//! Scorer trait and the pure-Rust [`HeuristicScorer`].

use crate::error::{Error, Result};
use crate::features::FeatureVector;

/// A fraud scorer. Takes a feature vector, returns a score in `[0.0, 1.0]`
/// where higher = more likely fraud.
///
/// Implementations must:
/// - Return `[0.0, 1.0]` for all valid inputs
/// - Be deterministic (same input → same output)
/// - Be fast: target p99 < 5ms on commodity merchant hardware
/// - Be pure: no I/O, no clock reads, no PII logging
pub trait Scorer: Send + Sync {
    /// Scorer name for telemetry (e.g. `"heuristic-v1"`, `"onnx-fraud-2026q1"`).
    fn name(&self) -> &str;

    /// Compute a fraud likelihood score in `[0.0, 1.0]`.
    fn score(&self, features: &FeatureVector) -> Result<f32>;
}

/// Pure-Rust rule-based scorer.
///
/// Useful as:
/// - A baseline before training models
/// - A fallback when an ONNX scorer fails to load
/// - A bound on minimum security (no fraud team should be worse than this)
///
/// The rules are intentionally conservative — they should produce
/// near-zero false-positive rate on legitimate payments and catch
/// obvious patterns:
///
/// - Very high amounts (>$10K equiv)
/// - High velocity (many transfers in a short window)
/// - Brand-new customers with large transfers
/// - Geo mismatch with history
///
/// Each rule contributes to the score; the final value is clamped to `[0,1]`.
#[derive(Debug, Clone)]
pub struct HeuristicScorer {
    /// Weight for "amount > $10K" signal. Default: 0.30
    pub w_high_amount: f32,
    /// Weight for "amount > $1K" signal. Default: 0.10
    pub w_medium_amount: f32,
    /// Weight for high velocity (>5 tx/h). Default: 0.20
    pub w_velocity: f32,
    /// Weight for new customer + medium amount. Default: 0.20
    pub w_new_customer_amount: f32,
    /// Weight for geo mismatch. Default: 0.20
    pub w_geo_mismatch: f32,
    /// Weight for night-time transfers. Default: 0.05
    pub w_night: f32,
    /// Weight for round amounts (often fraud). Default: 0.05
    pub w_round_amount: f32,
}

impl Default for HeuristicScorer {
    fn default() -> Self {
        Self {
            w_high_amount: 0.30,
            w_medium_amount: 0.10,
            w_velocity: 0.20,
            w_new_customer_amount: 0.20,
            w_geo_mismatch: 0.20,
            w_night: 0.05,
            w_round_amount: 0.05,
        }
    }
}

impl HeuristicScorer {
    /// Construct with default weights.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Scorer for HeuristicScorer {
    fn name(&self) -> &'static str {
        "heuristic-v1"
    }

    fn score(&self, f: &FeatureVector) -> Result<f32> {
        // Feature indices documented in features.rs.
        let high_amount = f[7]; // amount > $10K
        let medium_amount = f[6]; // amount > $1K
        let velocity_1h = f[16]; // log1p(velocity_1h)
        let new_customer = f[21]; // 1.0 if new, 0.5 if unknown, 0.0 if old
        let geo_match = f[22]; // 1.0 if matches, 0.5 if unknown, 0.0 if mismatch
        let is_night = f[11];
        let is_round = f[5];

        // Velocity contribution: log1p(velocity) > log1p(5) ≈ 1.79 is suspicious.
        // Map log1p(velocity) > log1p(5) → 1.0, < log1p(2) → 0.0.
        let velocity_signal =
            ((velocity_1h - 3.0_f32.ln()) / (6.0_f32.ln() - 3.0_f32.ln())).clamp(0.0, 1.0);

        // Geo signal: definitely-mismatched (0.0) → 1.0 risk
        //             unknown (0.5) → 0.5 risk
        //             match (1.0) → 0.0 risk
        let geo_signal = (1.0 - geo_match).clamp(0.0, 1.0);

        // New customer + medium-amount combo
        let new_customer_amount = new_customer.max(0.0) * medium_amount;

        let mut score = self.w_high_amount * high_amount
            + self.w_medium_amount * medium_amount
            + self.w_velocity * velocity_signal
            + self.w_new_customer_amount * new_customer_amount
            + self.w_geo_mismatch * geo_signal
            + self.w_night * is_night
            + self.w_round_amount * is_round;

        score = score.clamp(0.0, 1.0);
        if !score.is_finite() {
            return Err(Error::ModelOutput(format!(
                "heuristic produced non-finite score: {score}"
            )));
        }
        Ok(score)
    }
}

#[cfg(test)]
mod tests {
    // Scores clamp to exact 0.0/1.0 at saturation and the scorer is
    // deterministic (same features → same score), so `==` is the
    // correct assertion.
    #![allow(clippy::float_cmp)]
    use super::*;
    use crate::context::ScoringContext;
    use crate::features::{FEATURES, PaymentDescriptor, extract_features};
    use op_core::{Currency, Money, PaymentMethod, RailKind, VaultRef};
    use time::macros::datetime;

    fn make_method() -> PaymentMethod {
        PaymentMethod::Vault(VaultRef::new("tok_x"))
    }

    fn descriptor(method: &PaymentMethod, amount: Money, rail: RailKind) -> PaymentDescriptor<'_> {
        PaymentDescriptor {
            amount,
            method,
            rail,
            creditor_account: Some("creditor_a"),
            creditor_name: Some("Alice"),
            debtor_account: Some("debtor_b"),
            has_remittance: false,
        }
    }

    #[test]
    fn heuristic_score_is_bounded() {
        let m = make_method();
        let p = descriptor(&m, Money::from_minor(100, Currency::USD), RailKind::Card);
        let f = extract_features(&p, &ScoringContext::empty()).unwrap();
        let s = HeuristicScorer::new().score(&f).unwrap();
        assert!((0.0..=1.0).contains(&s), "score {s} out of range");
    }

    #[test]
    fn small_routine_payment_scores_low() {
        let m = make_method();
        // $4.99 (not round), daytime weekday, established customer, geo matches.
        let p = descriptor(&m, Money::from_minor(499, Currency::USD), RailKind::Card);
        let ctx = ScoringContext {
            timestamp: Some(datetime!(2026-05-20 14:30:00 UTC)), // Wed afternoon
            velocity_1h: Some(0),
            velocity_24h: Some(1),
            is_new_customer: Some(false),
            geo_matches_history: Some(true),
            ..Default::default()
        };
        let f = extract_features(&p, &ctx).unwrap();
        let s = HeuristicScorer::new().score(&f).unwrap();
        assert!(s < 0.05, "routine payment should score very low; got {s}");
    }

    #[test]
    fn very_high_amount_scores_higher() {
        let m = make_method();
        // $50K USD A2A from a new customer
        let p = descriptor(
            &m,
            Money::from_minor(5_000_000, Currency::USD),
            RailKind::A2a,
        );
        let ctx = ScoringContext {
            timestamp: Some(datetime!(2026-05-20 14:30:00 UTC)),
            is_new_customer: Some(true),
            geo_matches_history: Some(false),
            ..Default::default()
        };
        let f = extract_features(&p, &ctx).unwrap();
        let s = HeuristicScorer::new().score(&f).unwrap();
        assert!(
            s > 0.5,
            "high-amount new-customer geo-mismatch should score high; got {s}"
        );
    }

    #[test]
    fn high_velocity_increases_score() {
        let m = make_method();
        let p = descriptor(&m, Money::from_minor(50_000, Currency::USD), RailKind::Card);
        let ctx_low = ScoringContext {
            timestamp: Some(datetime!(2026-05-20 14:30:00 UTC)),
            velocity_1h: Some(0),
            ..Default::default()
        };
        let ctx_high = ScoringContext {
            timestamp: Some(datetime!(2026-05-20 14:30:00 UTC)),
            velocity_1h: Some(10),
            ..Default::default()
        };
        let s_low = HeuristicScorer::new()
            .score(&extract_features(&p, &ctx_low).unwrap())
            .unwrap();
        let s_high = HeuristicScorer::new()
            .score(&extract_features(&p, &ctx_high).unwrap())
            .unwrap();
        assert!(
            s_high > s_low,
            "high velocity should score higher: {s_low} vs {s_high}"
        );
    }

    #[test]
    fn night_transfers_score_slightly_higher() {
        let m = make_method();
        let p = descriptor(&m, Money::from_minor(50_000, Currency::USD), RailKind::A2a);
        let ctx_day = ScoringContext {
            timestamp: Some(datetime!(2026-05-20 14:00:00 UTC)),
            is_new_customer: Some(false),
            geo_matches_history: Some(true),
            ..Default::default()
        };
        let ctx_night = ScoringContext {
            timestamp: Some(datetime!(2026-05-20 03:00:00 UTC)),
            is_new_customer: Some(false),
            geo_matches_history: Some(true),
            ..Default::default()
        };
        let s_day = HeuristicScorer::new()
            .score(&extract_features(&p, &ctx_day).unwrap())
            .unwrap();
        let s_night = HeuristicScorer::new()
            .score(&extract_features(&p, &ctx_night).unwrap())
            .unwrap();
        assert!(
            s_night > s_day,
            "night should score >= day: {s_day} vs {s_night}"
        );
    }

    #[test]
    fn geo_mismatch_increases_score() {
        let m = make_method();
        let p = descriptor(&m, Money::from_minor(50_000, Currency::USD), RailKind::A2a);
        let ctx_match = ScoringContext {
            timestamp: Some(datetime!(2026-05-20 14:00:00 UTC)),
            geo_matches_history: Some(true),
            ..Default::default()
        };
        let ctx_mismatch = ScoringContext {
            timestamp: Some(datetime!(2026-05-20 14:00:00 UTC)),
            geo_matches_history: Some(false),
            ..Default::default()
        };
        let s_match = HeuristicScorer::new()
            .score(&extract_features(&p, &ctx_match).unwrap())
            .unwrap();
        let s_mismatch = HeuristicScorer::new()
            .score(&extract_features(&p, &ctx_mismatch).unwrap())
            .unwrap();
        assert!(s_mismatch > s_match);
    }

    #[test]
    fn extreme_features_clamp_to_one() {
        // Construct a feature vector that maxes out every signal.
        let mut f = [0.0_f32; FEATURES];
        f[5] = 1.0; // round
        f[6] = 1.0; // >$1K
        f[7] = 1.0; // >$10K
        f[11] = 1.0; // night
        f[16] = 100.0; // pathological velocity
        f[21] = 1.0; // new customer
        f[22] = 0.0; // geo mismatch
        let s = HeuristicScorer::new().score(&f).unwrap();
        assert_eq!(s, 1.0, "saturated features should clamp to 1.0");
    }

    #[test]
    fn empty_features_score_zero() {
        let f = [0.0_f32; FEATURES];
        // Unknown geo_match (0.0 here, not 0.5) means "definitely mismatched"
        // by our encoding — so we expect the geo weight to contribute.
        // The score should still be small (just the geo weight at most).
        let s = HeuristicScorer::new().score(&f).unwrap();
        // Geo weight default = 0.20. Geo signal = 1.0 - 0.0 = 1.0. Score = 0.20.
        assert!((s - 0.20).abs() < 1e-5, "got {s}");
    }

    #[test]
    fn neutral_features_score_below_threshold() {
        // Now use a vector where bools are 0.5 (unknown).
        let mut f = [0.0_f32; FEATURES];
        f[21] = 0.5;
        f[22] = 0.5;
        let s = HeuristicScorer::new().score(&f).unwrap();
        // Geo signal = 1.0 - 0.5 = 0.5. Contribution = 0.5 * 0.20 = 0.10.
        assert!(s <= 0.15, "neutral score should be low; got {s}");
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(HeuristicScorer::new().name(), "heuristic-v1");
    }

    #[test]
    fn scorer_is_object_safe() {
        let s: Box<dyn Scorer> = Box::new(HeuristicScorer::new());
        let f = [0.5_f32; FEATURES];
        let _ = s.score(&f);
    }

    #[test]
    fn determinism() {
        let f = [0.123_f32; FEATURES];
        let s = HeuristicScorer::new();
        let v1 = s.score(&f).unwrap();
        let v2 = s.score(&f).unwrap();
        assert_eq!(v1, v2);
    }
}
