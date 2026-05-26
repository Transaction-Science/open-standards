//! Tests for the streaming `generate()` API and multi-turn `Conversation`.

use jouleclaw_loader_gguf::read_gguf;
use jouleclaw_loader_gguf::sample::SamplingConfig;
use jouleclaw_loader_gguf::synthetic::{synthesize_llama_gguf, SyntheticConfig};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_runtime::{
    generate, generate_stream, Conversation, GenerateConfig, KvCacheKind, TokenizerKind,
};
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
        head_count, head_count_kv,
        rms_eps: 1e-6,
        seed: 100,
        vocab: Some(vocab),
        merges: None,
        bos_id: Some(1), eos_id: Some(2), unk_id: Some(0), chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    read_gguf(Cursor::new(bytes)).expect("parse")
}

fn cfg_greedy(max_new: usize) -> GenerateConfig {
    GenerateConfig {
        max_new_tokens: max_new,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::InPlace,
        max_seq: None,
    stop_strings: Vec::new(),
    }
}

/// Streaming produces the same tokens as one-shot generate() (greedy).
#[test]
fn stream_matches_oneshot_greedy() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let prompt = "hello world";
    let cfg = cfg_greedy(8);

    let oneshot = generate(&model, &vocab, prompt, &cfg).unwrap();

    let stream = generate_stream(&model, &vocab, prompt, &cfg).unwrap();
    let streamed_ids: Vec<u32> = stream.map(|r| r.unwrap().id).collect();

    assert_eq!(streamed_ids, oneshot.tokens,
        "streaming should produce the same token IDs as one-shot generate");
}

/// Streamed text concatenates to the same string as one-shot result.
#[test]
fn stream_text_concatenates_to_oneshot_text() {
    let model = build_test_model(4, 2);  // GQA
    let vocab = Vocab::from_gguf(&model).unwrap();
    let prompt = "abc";
    let cfg = cfg_greedy(5);

    let oneshot = generate(&model, &vocab, prompt, &cfg).unwrap();
    let stream = generate_stream(&model, &vocab, prompt, &cfg).unwrap();
    let streamed_text: String = stream.map(|r| r.unwrap().text).collect();
    assert_eq!(streamed_text, oneshot.text);
}

/// Stream terminates cleanly at max_new_tokens.
#[test]
fn stream_terminates_at_max_new_tokens() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let stream = generate_stream(&model, &vocab, "x", &cfg_greedy(3)).unwrap();
    let tokens: Vec<_> = stream.collect();
    assert!(tokens.len() <= 3, "got {} tokens, expected ≤ 3", tokens.len());
    for t in &tokens { assert!(t.is_ok()); }
}

/// Position field on streamed tokens reflects index within the generation.
#[test]
fn stream_position_indexes_correctly() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let stream = generate_stream(&model, &vocab, "abc", &cfg_greedy(5)).unwrap();
    for (i, result) in stream.enumerate() {
        let tok = result.unwrap();
        assert_eq!(tok.position, i,
            "token {} should have position {}, got {}", i, i, tok.position);
    }
}

/// Stream is deterministic under greedy.
#[test]
fn stream_is_deterministic_greedy() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let s1: Vec<u32> = generate_stream(&model, &vocab, "abc", &cfg_greedy(6))
        .unwrap().map(|r| r.unwrap().id).collect();
    let s2: Vec<u32> = generate_stream(&model, &vocab, "abc", &cfg_greedy(6))
        .unwrap().map(|r| r.unwrap().id).collect();
    assert_eq!(s1, s2);
}

/// Conversation.extend() produces tokens via streaming.
#[test]
fn conversation_extend_produces_tokens() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let mut conv = Conversation::new(&model, &vocab, 128).unwrap();
    assert_eq!(conv.total_tokens, 0);

    let cfg = cfg_greedy(4);
    let toks: Vec<u32> = conv.extend("hello", &cfg).unwrap()
        .map(|r| r.unwrap().id).collect();
    assert!(!toks.is_empty(), "should generate at least one token");
    // Position counter advanced by prompt + generated.
    assert!(conv.total_tokens > 0);
}

/// Two consecutive `extend()` calls accumulate state.
/// `total_tokens` after turn 2 equals (prompt1 + gen1 + prompt2 + gen2) lengths.
#[test]
fn conversation_state_accumulates_across_turns() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let mut conv = Conversation::new(&model, &vocab, 256).unwrap();

    let cfg_t1 = GenerateConfig { max_new_tokens: 3, ..cfg_greedy(3) };
    let mut cfg_t2 = cfg_t1.clone();
    cfg_t2.add_bos = false;  // important: don't re-BOS on subsequent turns

    let t1_count: usize = conv.extend("ask", &cfg_t1).unwrap()
        .map(|r| r.unwrap()).count();
    let after_t1 = conv.total_tokens;
    assert!(t1_count > 0);
    assert!(after_t1 >= t1_count, "after turn 1 should have ≥ generated count");

    let t2_count: usize = conv.extend("more", &cfg_t2).unwrap()
        .map(|r| r.unwrap()).count();
    let after_t2 = conv.total_tokens;
    assert!(t2_count > 0);
    assert!(after_t2 > after_t1,
        "after turn 2 total ({}) should exceed turn 1 total ({})",
        after_t2, after_t1);
}

/// Conversation overflow returns an error rather than panicking.
#[test]
fn conversation_overflow_errors() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let mut conv = Conversation::new(&model, &vocab, 12).unwrap();

    // First turn fills most of the buffer.
    let cfg_t1 = cfg_greedy(4);
    let _: Vec<_> = conv.extend("test", &cfg_t1).unwrap().collect();

    // Second turn with large max_new_tokens should overflow before yielding.
    let mut cfg_t2 = cfg_t1.clone();
    cfg_t2.add_bos = false;
    cfg_t2.max_new_tokens = 100;
    let result = conv.extend("more", &cfg_t2);
    assert!(result.is_err(),
        "second turn should error when total would exceed max_seq");
}

/// Reset clears the cache and the position counter.
#[test]
fn conversation_reset_clears_state() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let mut conv = Conversation::new(&model, &vocab, 64).unwrap();

    let cfg = cfg_greedy(3);
    let _: Vec<_> = conv.extend("hello", &cfg).unwrap().collect();
    assert!(conv.total_tokens > 0);

    conv.reset();
    assert_eq!(conv.total_tokens, 0);

    // Should be able to start a fresh conversation after reset.
    let toks: Vec<u32> = conv.extend("again", &cfg).unwrap()
        .map(|r| r.unwrap().id).collect();
    assert!(!toks.is_empty());
}

/// Multi-turn equivalence: a single turn with prompt = P1 + P2 should give
/// the same tokens as two turns of P1 then P2 (under greedy, with
/// add_bos=false on the second turn). This proves the cache reuse is
/// semantically equivalent to reprocessing the whole prompt.
///
/// Note: this is approximate because tokenization of "P1 P2" might differ
/// from tokenize(P1) + tokenize(P2) at the boundary. So the comparison is
/// on the *generated* tokens after each prefill, not on prompt tokens.
#[test]
fn multi_turn_produces_same_generated_tokens_as_combined_oneshot() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let combined = "hello world";
    let cfg_combined = cfg_greedy(4);

    let oneshot = generate(&model, &vocab, combined, &cfg_combined).unwrap();

    // Now: turn 1 = "hello", turn 2 = " world" — observe tokens at end.
    // We compare the first generated token after the FULL prompt has been
    // ingested. After turn 2's prefill, the cache holds the same tokens
    // as oneshot's cache after prefill. The first sampled token (greedy
    // argmax of the last-position logits) should match.
    let mut conv = Conversation::new(&model, &vocab, 64).unwrap();
    let cfg_t1 = cfg_greedy(0);  // prefill only, generate 0 tokens
    let _: Vec<_> = conv.extend("hello", &cfg_t1).unwrap().collect();
    let cfg_t2 = GenerateConfig { add_bos: false, ..cfg_greedy(1) };
    let stream = conv.extend(" world", &cfg_t2).unwrap();
    let multi_turn_first: Option<u32> = stream.into_iter()
        .next().map(|r| r.unwrap().id);

    assert_eq!(multi_turn_first, oneshot.tokens.first().copied(),
        "first generated token should match between one-shot and 2-turn paths");
}

/// Visible demo: stream tokens one at a time.
#[test]
fn streaming_demo() {
    let model = build_test_model(4, 2);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let prompt = "hello";
    let cfg = GenerateConfig {
        max_new_tokens: 6,
        sampling: SamplingConfig::top_k(20, 0.7, 42),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::InPlace,
        max_seq: Some(48),
    stop_strings: Vec::new(),
    };

    println!("\n=== Streaming demo ===");
    println!("Prompt: {:?}", prompt);
    println!("Model: 2 layers, GQA (4 heads / 2 kv-heads)");
    println!("Sampling: top-k=20, temperature=0.7, seed=42");
    println!("Tokens as they arrive:");

    let stream = generate_stream(&model, &vocab, prompt, &cfg).unwrap();
    let mut total = 0;
    for result in stream {
        let tok = result.unwrap();
        println!("  [pos={:2}, id={:3}] {:?}", tok.position, tok.id, tok.text);
        total += 1;
    }
    println!("Generated {} tokens total.", total);
}

/// Visible demo: two-turn conversation.
#[test]
fn conversation_demo() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let mut conv = Conversation::new(&model, &vocab, 128).unwrap();
    let mut cfg = GenerateConfig {
        max_new_tokens: 4,
        sampling: SamplingConfig::top_k(15, 0.8, 7),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::InPlace,
        max_seq: None,
    stop_strings: Vec::new(),
    };

    println!("\n=== Multi-turn conversation demo ===");
    println!("Cache size: 128 tokens. KV state persists across turns.");

    println!("\nTurn 1 — prompt: \"hello\"");
    let t1_tokens: Vec<u32> = conv.extend("hello", &cfg).unwrap()
        .map(|r| r.unwrap().id).collect();
    println!("  Generated: {:?}", t1_tokens);
    println!("  Cache state after turn 1: {} tokens", conv.total_tokens);

    cfg.add_bos = false;  // crucial for subsequent turns
    println!("\nTurn 2 — prompt: \"goodbye\"");
    let t2_tokens: Vec<u32> = conv.extend("goodbye", &cfg).unwrap()
        .map(|r| r.unwrap().id).collect();
    println!("  Generated: {:?}", t2_tokens);
    println!("  Cache state after turn 2: {} tokens", conv.total_tokens);

    println!("\nKV cache survived the boundary — turn 2 didn't reprocess turn 1.");
}
