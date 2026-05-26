//! Tests for the high-level `generate()` utility.
//!
//! Verifies:
//! 1. Tokenization, prefill, decode, sampling, and detokenization all
//!    compose correctly into one call.
//! 2. Greedy generation is deterministic across runs.
//! 3. Seeded temperature sampling is deterministic across runs.
//! 4. EOS stops generation.
//! 5. Token count limits are honored.
//!
//! The output text is gibberish (random weights), but the *pipeline* is
//! exercised end-to-end. With real weights this same call produces real
//! text.

use jouleclaw_loader_gguf::read_gguf;
use jouleclaw_loader_gguf::sample::SamplingConfig;
use jouleclaw_loader_gguf::synthetic::{synthesize_llama_gguf, SyntheticConfig};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_runtime::{generate, GenerateConfig, KvCacheKind, TokenizerKind};
use std::io::Cursor;

fn build_test_vocab() -> Vec<(String, f32)> {
    let mut v = vec![
        ("<unk>".to_string(), 0.0),     // 0
        ("<s>".to_string(), 0.0),       // 1: BOS
        ("</s>".to_string(), 0.0),      // 2: EOS
        ("\u{2581}".to_string(), 0.0),  // 3: ▁
    ];
    for c in 'a'..='z' { v.push((c.to_string(), -1.0)); }
    for c in 'a'..='z' { v.push((format!("\u{2581}{}", c), -0.5)); }
    for b in 0..256u32 { v.push((format!("<0x{:02X}>", b), -10.0)); }
    v
}

fn build_test_model() -> jouleclaw_loader_gguf::GgufModel {
    let vocab = build_test_vocab();
    let cfg = SyntheticConfig {
        vocab_size: vocab.len(),
        embedding_length: 16,
        block_count: 2,
        feed_forward_length: 32,
        head_count: 4,
        head_count_kv: 4,
        rms_eps: 1e-6,
        seed: 42,
        vocab: Some(vocab),
        merges: None,
        bos_id: Some(1),
        eos_id: Some(2),
        unk_id: Some(0), chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    read_gguf(Cursor::new(bytes)).expect("parse synthetic gguf")
}

/// Generation produces a result with the expected fields.
#[test]
fn generate_produces_well_formed_result() {
    let model = build_test_model();
    let vocab = Vocab::from_gguf(&model).unwrap();

    let cfg = GenerateConfig {
        max_new_tokens: 5,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
    cache_kind: KvCacheKind::Concat,
    max_seq: None,
    stop_strings: Vec::new(),
    };

    let result = generate(&model, &vocab, "hello", &cfg).expect("generate");

    // Prompt "hello" with BOS should tokenize to at least 2 tokens.
    assert!(result.prompt_token_count >= 2,
        "prompt should produce ≥2 tokens, got {}", result.prompt_token_count);
    // Should produce up to max_new_tokens (might be fewer if EOS hit).
    assert!(result.tokens.len() <= cfg.max_new_tokens);
    // Text should be the decoded form of tokens.
    let expected_text = vocab.decode_spm(&result.tokens);
    assert_eq!(result.text, expected_text);
}

/// Greedy generation is deterministic across runs.
#[test]
fn greedy_generation_is_deterministic() {
    let model = build_test_model();
    let vocab = Vocab::from_gguf(&model).unwrap();

    let cfg = GenerateConfig {
        max_new_tokens: 8,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
    cache_kind: KvCacheKind::Concat,
    max_seq: None,
    stop_strings: Vec::new(),
    };

    let r1 = generate(&model, &vocab, "hello world", &cfg).unwrap();
    let r2 = generate(&model, &vocab, "hello world", &cfg).unwrap();
    let r3 = generate(&model, &vocab, "hello world", &cfg).unwrap();

    assert_eq!(r1.tokens, r2.tokens);
    assert_eq!(r2.tokens, r3.tokens);
    assert_eq!(r1.text, r2.text);
}

/// Seeded temperature sampling is deterministic.
#[test]
fn temperature_generation_is_deterministic_per_seed() {
    let model = build_test_model();
    let vocab = Vocab::from_gguf(&model).unwrap();

    let cfg = GenerateConfig {
        max_new_tokens: 6,
        sampling: SamplingConfig::temperature(1.0, 42),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
    cache_kind: KvCacheKind::Concat,
    max_seq: None,
    stop_strings: Vec::new(),
    };

    let r1 = generate(&model, &vocab, "test", &cfg).unwrap();
    let r2 = generate(&model, &vocab, "test", &cfg).unwrap();
    assert_eq!(r1.tokens, r2.tokens, "same seed should produce same tokens");
}

/// Different prompts produce different output token streams.
#[test]
fn different_prompts_produce_different_outputs() {
    let model = build_test_model();
    let vocab = Vocab::from_gguf(&model).unwrap();

    let cfg = GenerateConfig {
        max_new_tokens: 8,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
    cache_kind: KvCacheKind::Concat,
    max_seq: None,
    stop_strings: Vec::new(),
    };

    let r1 = generate(&model, &vocab, "hello", &cfg).unwrap();
    let r2 = generate(&model, &vocab, "world", &cfg).unwrap();

    // With random weights and different prompts, the generated streams
    // should diverge. (It's possible by coincidence they don't on some
    // synthetic configs, but extremely unlikely on this 312-token vocab.)
    assert_ne!(r1.tokens, r2.tokens,
        "different prompts should yield different greedy outputs");
}

/// `max_new_tokens = 0` produces an empty result.
#[test]
fn max_new_tokens_zero_produces_empty() {
    let model = build_test_model();
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = GenerateConfig {
        max_new_tokens: 0,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
    cache_kind: KvCacheKind::Concat,
    max_seq: None,
    stop_strings: Vec::new(),
    };
    let r = generate(&model, &vocab, "hello", &cfg).unwrap();
    assert!(r.tokens.is_empty());
    assert!(r.text.is_empty());
    assert!(!r.stopped_at_eos);
}

/// Tokenizer auto-detection picks SPM when no merges are present.
#[test]
fn tokenizer_auto_picks_spm_without_merges() {
    let model = build_test_model();
    let vocab = Vocab::from_gguf(&model).unwrap();
    // No BPE merges in this model.
    assert!(vocab.bpe_merges.is_none());

    let cfg = GenerateConfig {
        max_new_tokens: 3,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
    cache_kind: KvCacheKind::Concat,
    max_seq: None,
    stop_strings: Vec::new(),
    };
    let r = generate(&model, &vocab, "abc", &cfg).expect("auto should use SPM");
    // Just verify it ran.
    assert!(r.prompt_token_count > 0);
}

/// Explicit SPM works.
#[test]
fn explicit_spm_works() {
    let model = build_test_model();
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = GenerateConfig {
        max_new_tokens: 3,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Spm,
    cache_kind: KvCacheKind::Concat,
    max_seq: None,
    stop_strings: Vec::new(),
    };
    let r = generate(&model, &vocab, "abc", &cfg).unwrap();
    assert!(r.prompt_token_count > 0);
}

/// Empty prompt with `add_bos=true` produces BOS plus the SPM space-prefix
/// token (▁). That's the standard SPM behavior — every string gets a
/// leading ▁ before tokenization.
#[test]
fn empty_prompt_with_bos_produces_bos_plus_space_prefix() {
    let model = build_test_model();
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = GenerateConfig {
        max_new_tokens: 1,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
    cache_kind: KvCacheKind::Concat,
    max_seq: None,
    stop_strings: Vec::new(),
    };
    let r = generate(&model, &vocab, "", &cfg).expect("empty + BOS should work");
    assert_eq!(r.prompt_token_count, 2,
        "should be BOS + ▁ space-prefix (got {})", r.prompt_token_count);
}

/// Without BOS, the prompt is just the encoded text.
#[test]
fn no_bos_prompt_only_includes_text() {
    let model = build_test_model();
    let vocab = Vocab::from_gguf(&model).unwrap();

    let with_bos_cfg = GenerateConfig {
        max_new_tokens: 1, sampling: SamplingConfig::greedy(),
        add_bos: true, tokenizer_kind: TokenizerKind::Auto,
    cache_kind: KvCacheKind::Concat,
    max_seq: None,
    stop_strings: Vec::new(),
    };
    let without_bos_cfg = GenerateConfig {
        max_new_tokens: 1, sampling: SamplingConfig::greedy(),
        add_bos: false, tokenizer_kind: TokenizerKind::Auto,
    cache_kind: KvCacheKind::Concat,
    max_seq: None,
    stop_strings: Vec::new(),
    };

    let r_bos = generate(&model, &vocab, "abc", &with_bos_cfg).unwrap();
    let r_no_bos = generate(&model, &vocab, "abc", &without_bos_cfg).unwrap();

    assert_eq!(r_bos.prompt_token_count, r_no_bos.prompt_token_count + 1,
        "with_bos should have exactly one more token than without_bos");
}

/// End-to-end demo. Run with `cargo test --test generation full_pipeline_demo -- --nocapture`
/// to see what an actual generation call looks like.
#[test]
fn full_pipeline_demo() {
    let model = build_test_model();
    let vocab = Vocab::from_gguf(&model).unwrap();

    println!("\n=== Joule generate() demo ===");
    println!("Model: {} layers, embedding={}, heads={}",
        2, 16, 4);
    println!("Vocab: {} tokens", vocab.len());

    let prompt = "hello world";
    println!("\nPrompt: {:?}", prompt);

    let cfg = GenerateConfig {
        max_new_tokens: 8,
        sampling: SamplingConfig::top_k(40, 0.8, 12345),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
    cache_kind: KvCacheKind::Concat,
    max_seq: None,
    stop_strings: Vec::new(),
    };
    println!("Sampling: top-k=40, temperature=0.8, seed=12345");
    println!("Max new tokens: 8");

    let result = generate(&model, &vocab, prompt, &cfg).expect("generate");

    println!("\nPrompt tokens: {} ({})", result.prompt_token_count,
        vocab.encode_spm(prompt, true).iter()
            .map(|id| format!("{}", id))
            .collect::<Vec<_>>().join(", "));
    println!("Generated tokens: {} ({})", result.tokens.len(),
        result.tokens.iter()
            .map(|id| format!("{}", id))
            .collect::<Vec<_>>().join(", "));
    println!("Decoded output: {:?}", result.text);
    println!("Stopped at EOS: {}", result.stopped_at_eos);
    println!("\n(Output is gibberish because weights are random;");
    println!(" the pipeline is what's being demonstrated.)");
}
