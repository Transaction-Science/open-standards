//! Decision layer.
//!
//! Maps a raw score in `[0.0, 1.0]` to a [`FraudDecision`] using
//! calibrated thresholds. Operators tune thresholds to their own
//! false-positive / false-negative trade-off.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// What the orchestrator does with the payment.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FraudDecision {
    /// Score below the review threshold. Route to the chosen rail.
    Approve,
    /// Score above review threshold but below decline. The orchestrator
    /// should hold the payment, alert a human, optionally request
    /// step-up authentication.
    Review,
    /// Score above decline threshold. Refuse routing. The caller
    /// receives a `Payment<Failed>` with `FraudDecision` attached.
    Decline,
    /// Score above freeze threshold. Severe — the orchestrator should
    /// also flag the customer's account for review of all subsequent
    /// activity.
    Freeze,
}

impl FraudDecision {
    /// True if the payment should be sent to a rail.
    #[must_use]
    pub const fn is_approve(self) -> bool {
        matches!(self, Self::Approve)
    }

    /// True if the payment is permanently rejected (no retry).
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Decline | Self::Freeze)
    }

    /// True if the payment requires manual review.
    #[must_use]
    pub const fn needs_review(self) -> bool {
        matches!(self, Self::Review)
    }
}

/// Operator-tunable decision thresholds.
///
/// Defaults follow industry-standard calibrations for instant payments
/// where false negatives (letting fraud through) cost more than false
/// positives (annoying legitimate customers):
///
/// - **Review at 0.5**: 50%+ likelihood of fraud → human-in-the-loop
/// - **Decline at 0.80**: high confidence → block silently
/// - **Freeze at 0.95**: near-certain → block and escalate
///
/// These defaults are conservative for *A2A* rails (irrevocable). Card
/// deployments may safely loosen them since chargebacks provide recourse.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Thresholds {
    /// Score ≥ this triggers [`FraudDecision::Review`]. Default: 0.50.
    pub review: f32,
    /// Score ≥ this triggers [`FraudDecision::Decline`]. Default: 0.80.
    pub decline: f32,
    /// Score ≥ this triggers [`FraudDecision::Freeze`]. Default: 0.95.
    pub freeze: f32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            review: 0.50,
            decline: 0.80,
            freeze: 0.95,
        }
    }
}

impl Thresholds {
    /// Construct with custom thresholds. Validates order.
    ///
    /// # Errors
    /// `Error::Backend` if thresholds aren't monotonically increasing
    /// or are out of `[0.0, 1.0]`.
    pub fn new(review: f32, decline: f32, freeze: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&review)
            || !(0.0..=1.0).contains(&decline)
            || !(0.0..=1.0).contains(&freeze)
        {
            return Err(Error::Backend(format!(
                "thresholds out of [0,1]: review={review}, decline={decline}, freeze={freeze}"
            )));
        }
        if !(review <= decline && decline <= freeze) {
            return Err(Error::Backend(format!(
                "thresholds must be monotone: review={review}, decline={decline}, freeze={freeze}"
            )));
        }
        Ok(Self {
            review,
            decline,
            freeze,
        })
    }

    /// Convert a score to a decision.
    ///
    /// # Errors
    /// `Error::ScoreOutOfRange` if the score is outside `[0.0, 1.0]`.
    pub fn decide(&self, score: f32) -> Result<FraudDecision> {
        if !(0.0..=1.0).contains(&score) {
            return Err(Error::ScoreOutOfRange(score));
        }
        Ok(if score >= self.freeze {
            FraudDecision::Freeze
        } else if score >= self.decline {
            FraudDecision::Decline
        } else if score >= self.review {
            FraudDecision::Review
        } else {
            FraudDecision::Approve
        })
    }
}

#[cfg(test)]
mod tests {
    // Threshold constants are exact literals (0.50, 0.80, 0.95);
    // asserting them with `==` is correct, not float-fuzzy.
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn default_thresholds_are_sensible() {
        let t = Thresholds::default();
        assert_eq!(t.review, 0.50);
        assert_eq!(t.decline, 0.80);
        assert_eq!(t.freeze, 0.95);
        assert!(t.review < t.decline);
        assert!(t.decline < t.freeze);
    }

    #[test]
    fn approve_below_review_threshold() {
        let t = Thresholds::default();
        assert_eq!(t.decide(0.0).unwrap(), FraudDecision::Approve);
        assert_eq!(t.decide(0.49).unwrap(), FraudDecision::Approve);
        assert_eq!(t.decide(0.499).unwrap(), FraudDecision::Approve);
    }

    #[test]
    fn review_between_review_and_decline() {
        let t = Thresholds::default();
        assert_eq!(t.decide(0.50).unwrap(), FraudDecision::Review);
        assert_eq!(t.decide(0.70).unwrap(), FraudDecision::Review);
        assert_eq!(t.decide(0.79).unwrap(), FraudDecision::Review);
    }

    #[test]
    fn decline_between_decline_and_freeze() {
        let t = Thresholds::default();
        assert_eq!(t.decide(0.80).unwrap(), FraudDecision::Decline);
        assert_eq!(t.decide(0.90).unwrap(), FraudDecision::Decline);
        assert_eq!(t.decide(0.949).unwrap(), FraudDecision::Decline);
    }

    #[test]
    fn freeze_at_or_above_freeze_threshold() {
        let t = Thresholds::default();
        assert_eq!(t.decide(0.95).unwrap(), FraudDecision::Freeze);
        assert_eq!(t.decide(0.99).unwrap(), FraudDecision::Freeze);
        assert_eq!(t.decide(1.0).unwrap(), FraudDecision::Freeze);
    }

    #[test]
    fn score_out_of_range_errors() {
        let t = Thresholds::default();
        assert!(matches!(t.decide(-0.1), Err(Error::ScoreOutOfRange(_))));
        assert!(matches!(t.decide(1.1), Err(Error::ScoreOutOfRange(_))));
        assert!(matches!(t.decide(f32::NAN), Err(Error::ScoreOutOfRange(_))));
        assert!(matches!(
            t.decide(f32::INFINITY),
            Err(Error::ScoreOutOfRange(_))
        ));
    }

    #[test]
    fn custom_thresholds_validated() {
        // Valid
        assert!(Thresholds::new(0.3, 0.5, 0.7).is_ok());
        // Out of range
        assert!(Thresholds::new(-0.1, 0.5, 0.7).is_err());
        assert!(Thresholds::new(0.3, 1.1, 0.7).is_err());
        // Wrong order
        assert!(Thresholds::new(0.7, 0.5, 0.3).is_err());
        assert!(Thresholds::new(0.5, 0.5, 0.4).is_err());
        // Equal-adjacent OK
        assert!(Thresholds::new(0.5, 0.5, 0.5).is_ok());
    }

    #[test]
    fn decision_helpers() {
        assert!(FraudDecision::Approve.is_approve());
        assert!(!FraudDecision::Review.is_approve());

        assert!(FraudDecision::Decline.is_terminal());
        assert!(FraudDecision::Freeze.is_terminal());
        assert!(!FraudDecision::Review.is_terminal());
        assert!(!FraudDecision::Approve.is_terminal());

        assert!(FraudDecision::Review.needs_review());
        assert!(!FraudDecision::Approve.needs_review());
    }

    #[test]
    fn decision_round_trips_through_json() {
        for d in [
            FraudDecision::Approve,
            FraudDecision::Review,
            FraudDecision::Decline,
            FraudDecision::Freeze,
        ] {
            let s = serde_json::to_string(&d).unwrap();
            let back: FraudDecision = serde_json::from_str(&s).unwrap();
            assert_eq!(d, back);
        }
    }
}
