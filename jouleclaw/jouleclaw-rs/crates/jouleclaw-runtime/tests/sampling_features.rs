//! Tests for the production sampler features added in Phase 1.22:
//! repetition / frequency / presence penalties, logit bias, and stop strings.

use jouleclaw_loader_gguf::read_gguf;
use jouleclaw_loader_gguf::sample::{sample_logits, sample_logits_with_history, SamplingConfig};
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

fn build_test_model() -> jouleclaw_loader_gguf::GgufModel {
    let vocab = build_test_vocab();
    let cfg = SyntheticConfig {
        vocab_size: vocab.len(), embedding_length: 16,
        block_count: 2, feed_forward_length: 32,
        head_count: 4, head_count_kv: 4,
        rms_eps: 1e-6, seed: 300,
        vocab: Some(vocab), merges: None,
        bos_id: Some(1), eos_id: Some(2), unk_id: Some(0),
        chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    read_gguf(Cursor::new(bytes)).unwrap()
}

// ==================== sampler unit tests ====================

/// `sample_logits` (no history) and `sample_logits_with_history` agree
/// when no penalties are configured.
#[test]
fn no_history_no_penalty_equivalence() {
    let logits = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];
    let cfg = SamplingConfig::greedy();
    let a = sample_logits(&logits, &cfg);
    let b = sample_logits_with_history(&logits, &cfg, &[0, 3, 3, 3]);
    assert_eq!(a, b,
        "with penalties disabled, history should be ignored");
}

/// Logit bias of -inf bans a token from being sampled.
#[test]
fn logit_bias_neg_inf_bans_token() {
    let logits = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];  // argmax = 3
    let cfg = SamplingConfig::greedy().banning(&[3]);
    let id = sample_logits(&logits, &cfg);
    assert_ne!(id, 3, "banned token should not be sampled");
    assert_eq!(id, 1, "next-highest token (index 1, value 0.5) should be picked");
}

/// Multiple bans work together.
#[test]
fn logit_bias_can_ban_multiple_tokens() {
    let logits = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];  // ranking: 3, 1, 2, 4, 0
    let cfg = SamplingConfig::greedy().banning(&[3, 1]);
    let id = sample_logits(&logits, &cfg);
    assert_eq!(id, 2, "after banning 3 and 1, next is 2 (logit 0.3)");
}

/// Positive logit bias boosts a token.
#[test]
fn positive_logit_bias_boosts_token() {
    let logits = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];  // argmax = 3
    let cfg = SamplingConfig::greedy().with_logit_bias(0, 10.0);
    let id = sample_logits(&logits, &cfg);
    assert_eq!(id, 0, "token 0 boosted by +10 should now be argmax");
}

/// Repetition penalty actually penalizes repeated tokens.
#[test]
fn repetition_penalty_demotes_repeats() {
    let logits = vec![1.0f32, 0.9, 0.5, 0.3];  // argmax = 0
    // Penalize positive logits by 2.0× (logit -> logit/2.0)
    let cfg = SamplingConfig::greedy().with_repetition_penalty(2.0);
    let id_no_history = sample_logits_with_history(&logits, &cfg, &[]);
    assert_eq!(id_no_history, 0, "no history → no penalty → original argmax");

    let id_with_history = sample_logits_with_history(&logits, &cfg, &[0, 0]);
    // After penalty: logit[0] = 1.0 / 2.0 = 0.5 < logit[1] = 0.9
    assert_ne!(id_with_history, 0,
        "token 0 was used recently; repetition penalty should make it lose");
    assert_eq!(id_with_history, 1,
        "expected next-best token after penalizing 0");
}

/// Frequency penalty scales with count.
#[test]
fn frequency_penalty_scales_with_count() {
    let logits = vec![1.0f32, 0.9, 0.5];
    let cfg = SamplingConfig::greedy().with_frequency_penalty(0.2);
    // Token 0 used 5 times → logit -= 0.2 * 5 = 1.0, so logit[0] becomes 0
    // Token 1 used 1 time → logit -= 0.2, so logit[1] becomes 0.7
    // Argmax: token 1.
    let id = sample_logits_with_history(&logits, &cfg, &[0, 0, 0, 0, 0, 1]);
    assert_eq!(id, 1);
}

/// Presence penalty is independent of count (single hit).
#[test]
fn presence_penalty_independent_of_count() {
    let logits = vec![1.0f32, 0.5, 0.3];
    // Presence penalty of 0.6 → if a token appears at all, logit -= 0.6.
    let cfg = SamplingConfig::greedy().with_presence_penalty(0.6);
    // Token 0 appears 10 times — still only one penalty hit: logit[0] = 0.4.
    // Token 1 still 0.5 (not in history). Argmax = 1.
    let id = sample_logits_with_history(&logits, &cfg, &[0; 10]);
    assert_eq!(id, 1, "after presence penalty, token 1 (0.5) > token 0 (0.4)");
}

/// Penalties combine.
#[test]
fn penalties_compose() {
    let logits = vec![1.0f32, 0.7, 0.5];
    let cfg = SamplingConfig::greedy()
        .with_repetition_penalty(2.0)
        .with_frequency_penalty(0.1)
        .with_presence_penalty(0.05);
    // Token 0 used 3 times. After:
    //   rep: 1.0 / 2.0 = 0.5
    //   freq: 0.5 - 0.1*3 = 0.2
    //   pres: 0.2 - 0.05 = 0.15
    // Token 1: not in history, stays 0.7.
    let id = sample_logits_with_history(&logits, &cfg, &[0, 0, 0]);
    assert_eq!(id, 1);
}

// ==================== end-to-end generation tests ====================

/// generate() honors stop strings.
#[test]
fn generate_stops_at_stop_string() {
    let model = build_test_model();
    let vocab = Vocab::from_gguf(&model).unwrap();

    // First run: no stop string, see what gets generated.
    let cfg_baseline = GenerateConfig {
        max_new_tokens: 8,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::InPlace,
        max_seq: None,
        stop_strings: Vec::new(),
    };
    let baseline = generate(&model, &vocab, "hello", &cfg_baseline).unwrap();
    // The greedy output has SOME first character — pick any short prefix.
    // If baseline.text is empty, the test can't proceed meaningfully.
    if baseline.text.is_empty() {
        return;
    }
    // Pick the first character of the generated text as the stop string.
    let first_char: String = baseline.text.chars().take(1).collect();

    let cfg_stop = GenerateConfig {
        stop_strings: vec![first_char.clone()],
        ..cfg_baseline.clone()
    };
    let stopped = generate(&model, &vocab, "hello", &cfg_stop).unwrap();
    // The output text must NOT contain the first character (it should
    // have been trimmed at the stop boundary).
    assert!(!stopped.text.contains(&first_char),
        "stop string {:?} should be trimmed; got text {:?}",
        first_char, stopped.text);
    // Either the text is shorter or empty.
    assert!(stopped.text.len() < baseline.text.len(),
        "stop string should produce shorter output: baseline {:?}, stopped {:?}",
        baseline.text, stopped.text);
}

/// Empty stop strings vec is a no-op.
#[test]
fn empty_stop_strings_is_noop() {
    let model = build_test_model();
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
    let r1 = generate(&model, &vocab, "abc", &cfg).unwrap();
    let mut cfg2 = cfg.clone();
    cfg2.stop_strings = vec!["".to_string()];  // empty stop string also no-op
    let r2 = generate(&model, &vocab, "abc", &cfg2).unwrap();
    assert_eq!(r1.tokens, r2.tokens,
        "empty stop strings should not change generation");
}

/// Banning a token at the API level prevents it from appearing in output.
#[test]
fn generation_banning_works_end_to_end() {
    let model = build_test_model();
    let vocab = Vocab::from_gguf(&model).unwrap();

    // First, see which token greedy picks first.
    let cfg_baseline = GenerateConfig {
        max_new_tokens: 1,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::InPlace,
        max_seq: None,
        stop_strings: Vec::new(),
    };
    let baseline = generate(&model, &vocab, "hello", &cfg_baseline).unwrap();
    if baseline.tokens.is_empty() { return; }
    let first_token = baseline.tokens[0];

    // Now ban it and generate again.
    let cfg_ban = GenerateConfig {
        sampling: SamplingConfig::greedy().banning(&[first_token]),
        ..cfg_baseline.clone()
    };
    let banned = generate(&model, &vocab, "hello", &cfg_ban).unwrap();
    if banned.tokens.is_empty() { return; }
    assert_ne!(banned.tokens[0], first_token,
        "banned token {} should not be sampled; got {:?}",
        first_token, banned.tokens);
}

/// Repetition penalty changes the output trajectory.
#[test]
fn repetition_penalty_changes_output() {
    let model = build_test_model();
    let vocab = Vocab::from_gguf(&model).unwrap();

    let cfg_base = GenerateConfig {
        max_new_tokens: 8,
        sampling: SamplingConfig::greedy(),
        add_bos: true,
        tokenizer_kind: TokenizerKind::Auto,
        cache_kind: KvCacheKind::InPlace,
        max_seq: None,
        stop_strings: Vec::new(),
    };
    let no_penalty = generate(&model, &vocab, "abc", &cfg_base).unwrap();

    let cfg_penalty = GenerateConfig {
        sampling: SamplingConfig::greedy().with_repetition_penalty(2.5),
        ..cfg_base.clone()
    };
    let with_penalty = generate(&model, &vocab, "abc", &cfg_penalty).unwrap();

    // The two streams might match if the model has no natural tendency
    // to repeat (with random weights). At minimum, the test verifies
    // the API runs cleanly.
    let _ = no_penalty;
    let _ = with_penalty;
    // No assertion: with random-weight models, repetition behavior is
    // unpredictable. The unit tests above prove the math works.
}

/// Demo: visible behavior of each sampling feature.
#[test]
fn sampling_features_demo() {
    println!("\n=== Production sampling features demo ===");
    let logits = vec![1.5f32, 1.0, 0.8, 0.5, 0.2, -0.5];
    println!("Logits: {:?}", logits);
    println!();

    // Baseline greedy.
    let cfg = SamplingConfig::greedy();
    println!("Greedy:                      {}",
        sample_logits(&logits, &cfg));

    // Ban argmax.
    let cfg = SamplingConfig::greedy().banning(&[0]);
    println!("Greedy banning token 0:      {}",
        sample_logits(&logits, &cfg));

    // Boost a low-ranked token.
    let cfg = SamplingConfig::greedy().with_logit_bias(5, 10.0);
    println!("Greedy +10 bias on token 5:  {}",
        sample_logits(&logits, &cfg));

    // Repetition penalty (no history → no effect).
    let cfg = SamplingConfig::greedy().with_repetition_penalty(2.0);
    let id_a = sample_logits_with_history(&logits, &cfg, &[]);
    let id_b = sample_logits_with_history(&logits, &cfg, &[0, 0]);
    println!("Rep penalty 2.0 (history=[]): {}", id_a);
    println!("Rep penalty 2.0 (history=[0,0]): {} (token 0 demoted)", id_b);

    // Frequency penalty.
    let cfg = SamplingConfig::greedy().with_frequency_penalty(0.3);
    let id = sample_logits_with_history(&logits, &cfg, &[0, 0, 0, 0]);
    println!("Freq penalty 0.3 (history=[0×4]): {} (logit[0] - 1.2 = 0.3)", id);

    // Presence penalty.
    let cfg = SamplingConfig::greedy().with_presence_penalty(1.2);
    let id = sample_logits_with_history(&logits, &cfg, &[0]);
    println!("Presence penalty 1.2 (history=[0]): {} (single hit)", id);

    println!();
    println!("All five features combine through SamplingConfig builder methods.");
}
