//! Sampling tests.

use jouleclaw_loader_gguf::sample::{sample_logits, SamplingConfig};

/// Greedy sampling picks the argmax with low-index tie-break.
#[test]
fn greedy_picks_argmax() {
    let logits = vec![0.1, -1.0, 0.5, 2.5, 1.7];
    let id = sample_logits(&logits, &SamplingConfig::greedy());
    assert_eq!(id, 3);
}

/// Greedy with ties: lower index wins.
#[test]
fn greedy_tie_break_lower_index() {
    let logits = vec![1.0, 1.0, 1.0, 0.5];
    let id = sample_logits(&logits, &SamplingConfig::greedy());
    assert_eq!(id, 0, "expected lowest index 0 on three-way tie");
}

/// Temperature sampling is deterministic given a seed.
#[test]
fn temperature_seeded_is_deterministic() {
    let logits = vec![0.5, 1.0, 1.5, 2.0, 1.8, 0.3];
    let cfg = SamplingConfig::temperature(0.8, 42);
    let id1 = sample_logits(&logits, &cfg);
    let id2 = sample_logits(&logits, &cfg);
    let id3 = sample_logits(&logits, &cfg);
    assert_eq!(id1, id2);
    assert_eq!(id2, id3);
}

/// Different seeds can produce different samples.
#[test]
fn different_seeds_can_diverge() {
    let logits = vec![0.5, 1.0, 1.5, 2.0, 1.8, 0.3];
    let mut samples = std::collections::HashSet::new();
    for seed in 0..50 {
        let cfg = SamplingConfig::temperature(2.0, seed);
        samples.insert(sample_logits(&logits, &cfg));
    }
    assert!(samples.len() > 1,
        "high-temperature sampling over 50 seeds should produce >1 distinct token");
}

/// Top-K filter restricts the selection set.
#[test]
fn top_k_only_samples_from_top_k() {
    // Logits: idx 0 has score 100 (so high it dominates anything past it
    // in any reasonable softmax), idx 1 = 50, rest negative.
    let mut logits = vec![-100.0_f32; 32];
    logits[0] = 100.0;
    logits[1] = 50.0;

    // top_k=2 with high temperature: must always sample 0 or 1.
    for seed in 0..30 {
        let cfg = SamplingConfig::top_k(2, 1.0, seed);
        let id = sample_logits(&logits, &cfg);
        assert!(id == 0 || id == 1,
            "top-k=2 should sample from idx 0 or 1, got {}", id);
    }
}

/// Top-K=1 collapses to greedy regardless of temperature/seed.
#[test]
fn top_k_one_is_greedy() {
    let logits = vec![0.5, 2.0, 1.0, 1.5];
    for seed in 0..10 {
        let cfg = SamplingConfig::top_k(1, 5.0, seed);
        let id = sample_logits(&logits, &cfg);
        assert_eq!(id, 1, "top-k=1 should always pick argmax (1)");
    }
}

/// Top-P (nucleus) filter takes the smallest set whose cumulative
/// probability covers the threshold.
#[test]
fn top_p_truncates_long_tail() {
    // Two dominant tokens that together cover ~95% of mass; rest are
    // long-tail noise.
    let mut logits = vec![-10.0_f32; 100];
    logits[0] = 5.0;
    logits[1] = 4.5;

    // top_p = 0.5 should always select from {0, 1}.
    for seed in 0..30 {
        let cfg = SamplingConfig::top_p(0.5, 1.0, seed);
        let id = sample_logits(&logits, &cfg);
        assert!(id == 0 || id == 1,
            "top-p=0.5 with two dominant tokens should sample from {{0, 1}}, got {}", id);
    }
}

/// Greedy is unaffected by temperature/seed since it short-circuits at T=0.
#[test]
fn greedy_ignores_seed() {
    let logits = vec![0.1, 0.5, 0.3, 0.9, 0.7];
    let id1 = sample_logits(&logits, &SamplingConfig {
        temperature: 0.0, top_k: 0, top_p: 1.0, seed: 1, ..Default::default()
    });
    let id2 = sample_logits(&logits, &SamplingConfig {
        temperature: 0.0, top_k: 0, top_p: 1.0, seed: 999999, ..Default::default()
    });
    assert_eq!(id1, id2);
    assert_eq!(id1, 3);
}
