//! Ebbinghaus forgetting-curve tests.

use eoc_memory::forget::{EbbinghausScorer, ForgetConfig};

#[test]
fn fresh_memory_has_full_retention() {
    let scorer = EbbinghausScorer::new(ForgetConfig::default());
    let r = scorer.retention(1_000, 1_000, 0);
    assert!((r - 1.0).abs() < 1e-9, "retention at t=0 must be 1.0; got {r}");
}

#[test]
fn retention_decays_with_time() {
    let cfg = ForgetConfig::new(1_000.0, 0.0, 0.2).expect("cfg");
    let scorer = EbbinghausScorer::new(cfg);
    let r_short = scorer.retention(0, 100, 0);
    let r_long = scorer.retention(0, 10_000, 0);
    assert!(r_short > r_long);
    assert!(r_long < 0.001);
}

#[test]
fn reinforcement_slows_decay() {
    let cfg = ForgetConfig::new(1_000.0, 1.0, 0.2).expect("cfg");
    let scorer = EbbinghausScorer::new(cfg);
    let r_unreviewed = scorer.retention(0, 5_000, 0);
    let r_reviewed = scorer.retention(0, 5_000, 3);
    assert!(
        r_reviewed > r_unreviewed,
        "reviewed retention {r_reviewed} must exceed unreviewed {r_unreviewed}"
    );
}

#[test]
fn is_forgotten_at_threshold() {
    let cfg = ForgetConfig::new(1_000.0, 0.0, 0.5).expect("cfg");
    let scorer = EbbinghausScorer::new(cfg);
    assert!(!scorer.is_forgotten(0, 0, 0));
    // After enough time retention will fall below 0.5.
    assert!(scorer.is_forgotten(0, 10_000, 0));
}

#[test]
fn invalid_config_is_rejected() {
    assert!(ForgetConfig::new(0.0, 1.0, 0.5).is_err());
    assert!(ForgetConfig::new(1.0, -0.1, 0.5).is_err());
    assert!(ForgetConfig::new(1.0, 1.0, 1.5).is_err());
}
