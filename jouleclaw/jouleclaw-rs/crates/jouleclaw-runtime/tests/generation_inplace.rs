//! Tests for `generate()` using the in-place KV cache.
//!
//! The defining property: `generate()` with the in-place cache produces
//! the same tokens (within sampling determinism) as `generate()` with the
//! concat cache. The in-place cache is now the default; this set of
//! tests verifies that change.

use jouleclaw_loader_gguf::read_gguf;
use jouleclaw_loader_gguf::sample::SamplingConfig;
use jouleclaw_loader_gguf::synthetic::{synthesize_llama_gguf, SyntheticConfig};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_runtime::{generate, GenerateConfig, KvCacheKind, TokenizerKind};
use std::io::Cursor;

fn build_test_vocab() -> Vec<(String, f32)> {
    let mut v = vec![
        ("<unk>".to_string(), 0.0),
        ("<s>".to_string(), 0.0),
        ("</s>".to_string(), 0.0),
        ("\u{2581}".to_string(), 0.0),
    ];
    for c in 'a'..='z' { v.push((c.to_string(), -1.0)); }
    for c in 'a'..='z' { v.push((format!("\u{2581}{}", c), -0.5)); }
    for b in 0..256u32 { v.push((format!("<0x{:02X}>", b), -10.0)); }
    v
}

fn build_test_model(head_count: usize, head_count_kv: usize) -> jouleclaw_loader_gguf::GgufModel {
    let vocab = build_test_vocab();
    let cfg = SyntheticConfig {
        vocab_size: vocab.len(),
        embedding_length: 16,
        block_count: 2,
        feed_forward_length: 32,
        head_count,
        head_count_kv,
        rms_eps: 1e-6,
        seed: 42,
        vocab: Some(vocab),
        merges: None,
        bos_id: Some(1), eos_id: Some(2), unk_id: Some(0), chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    read_gguf(Cursor::new(bytes)).expect("parse")
}

/// Default GenerateConfig uses the in-place cache.
#[test]
fn default_config_uses_inplace() {
    let cfg = GenerateConfig::default();
    assert_eq!(cfg.cache_kind, KvCacheKind::InPlace);
    assert!(cfg.max_seq.is_none(),
        "default max_seq should be None (auto-computed)");
}

/// In-place generation produces a well-formed result.
#[test]
fn inplace_generation_produces_well_formed_result() {
    let model = build_test_model(4, 4);  // MHA
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = GenerateConfig {
        max_new_tokens: 5,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::InPlace,
        max_seq: None,
    stop_strings: Vec::new(),
    };
    let result = generate(&model, &vocab, "hello", &cfg).expect("generate");
    assert!(result.prompt_token_count >= 2);
    assert!(result.tokens.len() <= cfg.max_new_tokens);
    assert_eq!(result.text, vocab.decode_spm(&result.tokens));
}

/// In-place greedy generation is deterministic across runs.
#[test]
fn inplace_greedy_is_deterministic() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = GenerateConfig {
        max_new_tokens: 8,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::InPlace,
        max_seq: Some(32),
    stop_strings: Vec::new(),
    };
    let r1 = generate(&model, &vocab, "hello world", &cfg).unwrap();
    let r2 = generate(&model, &vocab, "hello world", &cfg).unwrap();
    let r3 = generate(&model, &vocab, "hello world", &cfg).unwrap();
    assert_eq!(r1.tokens, r2.tokens);
    assert_eq!(r2.tokens, r3.tokens);
}

/// In-place and concat caches produce equivalent token streams under
/// greedy sampling (the most stringent test — any divergence shows up
/// as different argmax picks).
///
/// Note: this is a strong correctness claim. The two cache implementations
/// run different graph ops (Scatter+full-buffer attention vs.
/// Concat+truncated attention), so floating-point order differs slightly.
/// We allow that argmax could flip at a position where the top-2 logits
/// are within ~1e-3 of each other; in practice with small random-weight
/// models this is rare but possible. The test asserts at least 80% of
/// the generated tokens match.
#[test]
fn inplace_and_concat_produce_equivalent_greedy_streams() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();

    let mut cfg = GenerateConfig {
        max_new_tokens: 16,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::Concat,
        max_seq: None,
    stop_strings: Vec::new(),
    };
    let r_concat = generate(&model, &vocab, "hello world", &cfg).unwrap();
    cfg.cache_kind = KvCacheKind::InPlace;
    cfg.max_seq = Some(64);
    let r_inplace = generate(&model, &vocab, "hello world", &cfg).unwrap();

    assert_eq!(r_concat.tokens.len(), r_inplace.tokens.len());
    let matches = r_concat.tokens.iter()
        .zip(r_inplace.tokens.iter())
        .filter(|(a, b)| a == b).count();
    let total = r_concat.tokens.len();
    assert!(matches * 5 >= total * 4,
        "in-place and concat should agree on ≥80% of greedy tokens; \
         got {}/{} matches.\nconcat:   {:?}\ninplace:  {:?}",
        matches, total, r_concat.tokens, r_inplace.tokens);
}

/// In-place generation with GQA model.
#[test]
fn inplace_with_gqa_model() {
    let model = build_test_model(4, 2);  // GQA: group_size=2
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = GenerateConfig {
        max_new_tokens: 6,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::InPlace,
        max_seq: Some(32),
    stop_strings: Vec::new(),
    };
    let result = generate(&model, &vocab, "hello", &cfg).expect("GQA generate");
    assert!(!result.tokens.is_empty());
    assert_eq!(result.text, vocab.decode_spm(&result.tokens));
}

/// Auto-computed max_seq is large enough for prompt + max_new_tokens.
#[test]
fn auto_max_seq_handles_long_generation() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = GenerateConfig {
        max_new_tokens: 24,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::InPlace,
        max_seq: None,  // auto: prompt_len + max_new_tokens + 16
        stop_strings: Vec::new(),
    };
    let result = generate(&model, &vocab, "abc", &cfg).expect("auto max_seq");
    assert!(!result.tokens.is_empty());
}

/// Explicit max_seq that's too small returns an error.
#[test]
fn explicit_max_seq_too_small_errors() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = GenerateConfig {
        max_new_tokens: 100,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::InPlace,
        max_seq: Some(8),  // too small for 100 new tokens
        stop_strings: Vec::new(),
    };
    let r = generate(&model, &vocab, "hello", &cfg);
    assert!(r.is_err(),
        "should error when max_seq < prompt + max_new_tokens");
}

/// Explicit Concat cache still works (backward compatibility).
#[test]
fn explicit_concat_cache_works() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = GenerateConfig {
        max_new_tokens: 4,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::Concat,
        max_seq: None,  // ignored for Concat
        stop_strings: Vec::new(),
    };
    let result = generate(&model, &vocab, "test", &cfg).unwrap();
    assert!(!result.tokens.is_empty());
}

/// Demo: show both cache paths producing token streams.
#[test]
fn cache_kind_demo() {
    let model = build_test_model(4, 2);
    let vocab = Vocab::from_gguf(&model).unwrap();

    println!("\n=== Cache-kind comparison ===");
    println!("Model: 2 layers, embedding=16, heads=4, kv_heads=2 (GQA)");
    println!("Vocab: {} tokens", vocab.len());
    println!("Prompt: \"hello world\"");
    println!("Sampling: top-k=20, temperature=0.8, seed=999");

    let mut cfg = GenerateConfig {
        max_new_tokens: 8,
        sampling: SamplingConfig::top_k(20, 0.8, 999),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::Concat,
        max_seq: None,
    stop_strings: Vec::new(),
    };
    let r_concat = generate(&model, &vocab, "hello world", &cfg).unwrap();
    println!("\nConcat cache (grows per step):");
    println!("  prompt tokens: {}", r_concat.prompt_token_count);
    println!("  generated:     {} tokens — {:?}", r_concat.tokens.len(), r_concat.tokens);

    cfg.cache_kind = KvCacheKind::InPlace;
    cfg.max_seq = Some(64);
    let r_inplace = generate(&model, &vocab, "hello world", &cfg).unwrap();
    println!("\nIn-place cache (constant memory):");
    println!("  prompt tokens: {}", r_inplace.prompt_token_count);
    println!("  generated:     {} tokens — {:?}", r_inplace.tokens.len(), r_inplace.tokens);
    println!("  buffer size:   constant [n_heads_kv=2, max_seq=64, d_head=4]");
}
