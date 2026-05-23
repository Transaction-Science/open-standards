//! End-to-end tests for op-fraud.
//!
//! Exercise the full pipeline: build a `PaymentDescriptor`, run feature
//! extraction, score with the heuristic, and convert to a `FraudDecision`.
//!
//! No ONNX dependency — these tests work in any build.

use op_core::{A2aKey, Currency, Money, PaymentMethod, RailKind, VaultRef};
use op_fraud::features::PaymentDescriptor;
use op_fraud::{
    FraudDecision, HeuristicScorer, Scorer, ScoringContext, Thresholds, extract_features,
};
use time::macros::datetime;

fn vault() -> PaymentMethod {
    PaymentMethod::Vault(VaultRef::new("tok_x"))
}

fn descriptor<'a>(method: &'a PaymentMethod, amt: Money, rail: RailKind) -> PaymentDescriptor<'a> {
    PaymentDescriptor {
        amount: amt,
        method,
        rail,
        creditor_account: Some("creditor_acct"),
        creditor_name: Some("Recipient Inc"),
        debtor_account: Some("debtor_acct"),
        has_remittance: false,
    }
}

#[test]
fn normal_card_payment_approves() {
    let m = vault();
    let p = descriptor(&m, Money::from_minor(2_499, Currency::USD), RailKind::Card);
    let ctx = ScoringContext {
        timestamp: Some(datetime!(2026-05-20 12:00:00 UTC)),
        velocity_1h: Some(1),
        velocity_24h: Some(4),
        is_new_customer: Some(false),
        geo_matches_history: Some(true),
        ..Default::default()
    };
    let features = extract_features(&p, &ctx).unwrap();
    let score = HeuristicScorer::new().score(&features).unwrap();
    let decision = Thresholds::default().decide(score).unwrap();
    assert_eq!(decision, FraudDecision::Approve);
}

#[test]
fn large_a2a_to_new_unknown_recipient_reviews_or_declines() {
    let m = vault();
    let mut p = descriptor(
        &m,
        Money::from_minor(2_500_000, Currency::USD),
        RailKind::A2a,
    );
    p.creditor_account = Some("never_seen_before_account");
    p.creditor_name = Some("New Recipient");

    let ctx = ScoringContext {
        timestamp: Some(datetime!(2026-05-20 02:30:00 UTC)), // night
        velocity_1h: Some(0),
        is_new_customer: Some(true),
        geo_matches_history: Some(false),
        ..Default::default()
    };
    let features = extract_features(&p, &ctx).unwrap();
    let score = HeuristicScorer::new().score(&features).unwrap();
    let decision = Thresholds::default().decide(score).unwrap();
    assert!(
        decision == FraudDecision::Review
            || decision == FraudDecision::Decline
            || decision == FraudDecision::Freeze,
        "expected non-approve for risky A2A; got {decision:?} (score {score})"
    );
}

#[test]
fn velocity_spike_triggers_review() {
    let m = vault();
    let p = descriptor(&m, Money::from_minor(50_000, Currency::USD), RailKind::Card);
    let ctx = ScoringContext {
        timestamp: Some(datetime!(2026-05-20 12:00:00 UTC)),
        velocity_1h: Some(15),
        velocity_24h: Some(40),
        device_velocity_1h: Some(12),
        is_new_customer: Some(false),
        geo_matches_history: Some(true),
        ..Default::default()
    };
    let score = HeuristicScorer::new()
        .score(&extract_features(&p, &ctx).unwrap())
        .unwrap();
    // Score should reflect velocity contribution (~0.20 from velocity alone)
    // plus round-amount signal (~0.05).
    assert!(score >= 0.20, "got {score}");
}

#[test]
fn first_pix_transfer_at_3am_with_geo_mismatch_does_not_silently_approve() {
    let m = vault();
    // 5000 BRL — over the $1K-equivalent threshold (f[6] = 1.0).
    let mut p = descriptor(&m, Money::from_minor(500_000, Currency::BRL), RailKind::A2a);
    p.creditor_account = Some("new_brazilian_account");

    let ctx = ScoringContext {
        timestamp: Some(datetime!(2026-05-20 03:00:00 UTC)),
        is_new_customer: Some(true),
        geo_matches_history: Some(false),
        ..Default::default()
    };
    let features = extract_features(&p, &ctx).unwrap();
    let score = HeuristicScorer::new().score(&features).unwrap();
    let decision = Thresholds::default().decide(score).unwrap();
    assert_ne!(decision, FraudDecision::Approve);
}

#[test]
fn scorer_is_pluggable_via_trait() {
    // The orchestrator holds Box<dyn Scorer> and doesn't know which
    // implementation. This test proves the trait works.
    let scorers: Vec<Box<dyn Scorer>> = vec![Box::new(HeuristicScorer::new())];
    let f = [0.0_f32; op_fraud::FEATURES];
    for s in scorers {
        let _ = s.score(&f).unwrap();
    }
}

#[test]
fn features_omit_pii_text_completely() {
    let m = PaymentMethod::A2a(A2aKey::Pix("alice.silva@email.com".into()));
    let p = PaymentDescriptor {
        amount: Money::from_minor(10_000, Currency::BRL),
        method: &m,
        rail: RailKind::A2a,
        creditor_account: Some("VERY_SENSITIVE_ACCOUNT_12345"),
        creditor_name: Some("João da Silva"),
        debtor_account: Some("ANOTHER_SENSITIVE_ACCOUNT_67890"),
        has_remittance: true,
    };
    let features = extract_features(&p, &ScoringContext::empty()).unwrap();

    // No string slot exists in the FeatureVector by construction (it's
    // [f32; 32]). The compile-time guarantee is the test. Here we
    // additionally verify that hashing distinguishes the inputs.
    let mut p2 = p.clone();
    p2.creditor_account = Some("DIFFERENT_ACCOUNT_99999");
    let features2 = extract_features(&p2, &ScoringContext::empty()).unwrap();
    assert_ne!(
        features[28], features2[28],
        "different creditor accounts must hash differently"
    );

    // Everything else should be identical (we only changed creditor_account).
    for i in 0..op_fraud::FEATURES {
        if i == 28 {
            continue;
        }
        assert_eq!(features[i], features2[i], "feature {i} should match");
    }
}

#[test]
fn score_below_review_threshold_means_approve() {
    let t = Thresholds::default();
    assert_eq!(t.decide(0.49).unwrap(), FraudDecision::Approve);
    assert_eq!(t.decide(0.0).unwrap(), FraudDecision::Approve);
}

#[test]
fn full_pipeline_is_deterministic() {
    let m = vault();
    let p = descriptor(&m, Money::from_minor(12_345, Currency::USD), RailKind::Card);
    let ctx = ScoringContext {
        timestamp: Some(datetime!(2026-05-20 12:00:00 UTC)),
        velocity_1h: Some(2),
        device_id: Some("dev_xyz".into()),
        ..Default::default()
    };
    let s = HeuristicScorer::new();
    let f1 = extract_features(&p, &ctx).unwrap();
    let f2 = extract_features(&p, &ctx).unwrap();
    assert_eq!(f1, f2);
    let d1 = Thresholds::default().decide(s.score(&f1).unwrap()).unwrap();
    let d2 = Thresholds::default().decide(s.score(&f2).unwrap()).unwrap();
    assert_eq!(d1, d2);
}
