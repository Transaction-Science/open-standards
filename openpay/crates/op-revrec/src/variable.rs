//! Step 3 helper: variable consideration with the constraint of
//! ASC 606-10-32-11 ("highly probable that a significant reversal will
//! not occur").
//!
//! Variable consideration covers discounts, rebates, refund liabilities,
//! performance bonuses, contingent royalties, and any other amount that
//! is not fixed at contract inception. The standard requires the entity
//! to estimate the amount AND constrain it to the portion for which a
//! material reversal is unlikely.
//!
//! We model both estimation methods the standard recognises:
//!
//! - **Expected value** (ASC 606-10-32-8(a)) — probability-weighted sum
//!   over the distribution of possible outcomes. Best when the entity
//!   has many similar contracts.
//! - **Most likely amount** (ASC 606-10-32-8(b)) — the single most likely
//!   outcome. Best for binary outcomes (e.g. milestone bonus paid or not).
//!
//! The constraint is applied as an explicit ceiling — the auditor and
//! the entity agree on the amount that is "highly probable" of not
//! reversing, and that figure caps the recognized variable amount.

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};

/// One outcome and its probability for the expected-value method.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    /// The outcome amount in minor units (may be negative for a
    /// refund-liability or discount).
    pub amount_minor: i64,
    /// Probability of this outcome, 0..=1.
    pub probability: Decimal,
}

/// How the variable amount is estimated.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EstimationMethod {
    /// Probability-weighted sum over the outcome distribution.
    ExpectedValue {
        /// Discrete distribution; probabilities must sum to ~1.0.
        outcomes: Vec<Outcome>,
    },
    /// The single most likely outcome.
    MostLikelyAmount {
        /// The amount in minor units of the contract currency.
        amount_minor: i64,
    },
}

/// One variable-consideration component of a contract's transaction
/// price.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariableConsideration {
    /// Human-readable label (`"performance-bonus"`, `"volume-rebate"`).
    pub label: String,
    /// How the entity estimated the amount.
    pub method: EstimationMethod,
    /// The constraint ceiling in minor units — the largest amount the
    /// entity may recognize without breaching the "highly probable / no
    /// material reversal" test of ASC 606-10-32-11. Set to the
    /// unconstrained estimate to remove the constraint.
    pub constraint_ceiling_minor: i64,
}

impl VariableConsideration {
    /// Build an expected-value component with an explicit constraint.
    #[must_use]
    pub fn expected_value(
        label: impl Into<String>,
        outcomes: Vec<Outcome>,
        constraint_ceiling_minor: i64,
    ) -> Self {
        Self {
            label: label.into(),
            method: EstimationMethod::ExpectedValue { outcomes },
            constraint_ceiling_minor,
        }
    }

    /// Build a most-likely-amount component with an explicit constraint.
    #[must_use]
    pub fn most_likely(
        label: impl Into<String>,
        amount_minor: i64,
        constraint_ceiling_minor: i64,
    ) -> Self {
        Self {
            label: label.into(),
            method: EstimationMethod::MostLikelyAmount { amount_minor },
            constraint_ceiling_minor,
        }
    }

    /// The unconstrained estimate, in minor units. Negative for
    /// outcomes that reduce revenue (discounts, refunds).
    #[must_use]
    pub fn estimate_minor(&self) -> i64 {
        match &self.method {
            EstimationMethod::MostLikelyAmount { amount_minor } => *amount_minor,
            EstimationMethod::ExpectedValue { outcomes } => {
                // Probability-weighted sum. We compute in Decimal then
                // round to integer minor units (round-half-up).
                let mut acc = Decimal::ZERO;
                for o in outcomes {
                    let m = Decimal::from(o.amount_minor);
                    acc += m * o.probability;
                }
                // Round half-up; saturate on i64 overflow.
                acc.round_dp(0).to_i64().unwrap_or(i64::MAX)
            }
        }
    }

    /// Apply the constraint: the smaller of `|estimate|` and the
    /// ceiling, preserving sign. Returns the amount safe to recognize.
    #[must_use]
    pub fn constrained_amount_minor(&self) -> i64 {
        let est = self.estimate_minor();
        let sign = est.signum();
        // Take absolute value safely (i64::MIN.abs() would overflow;
        // saturate to i64::MAX to keep the constraint comparison sound).
        let abs = est.checked_abs().unwrap_or(i64::MAX);
        let capped = abs.min(self.constraint_ceiling_minor.unsigned_abs() as i64);
        sign.checked_mul(capped).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros_polyfill::dec;

    // Local dec! polyfill since rust_decimal_macros isn't a workspace dep.
    mod rust_decimal_macros_polyfill {
        macro_rules! dec {
            ($lit:literal) => {{
                let s = stringify!($lit);
                <rust_decimal::Decimal as core::str::FromStr>::from_str(s)
                    .unwrap_or(rust_decimal::Decimal::ZERO)
            }};
        }
        pub(crate) use dec;
    }

    #[test]
    fn most_likely_constrained_below_ceiling_returns_estimate() {
        let v = VariableConsideration::most_likely("bonus", 1_000, 5_000);
        assert_eq!(v.constrained_amount_minor(), 1_000);
    }

    #[test]
    fn most_likely_constrained_above_ceiling_returns_ceiling() {
        let v = VariableConsideration::most_likely("bonus", 10_000, 5_000);
        assert_eq!(v.constrained_amount_minor(), 5_000);
    }

    #[test]
    fn expected_value_weighted_sum() {
        let v = VariableConsideration::expected_value(
            "rebate",
            vec![
                Outcome { amount_minor: 0, probability: dec!(0.5) },
                Outcome { amount_minor: 1_000, probability: dec!(0.5) },
            ],
            1_000,
        );
        assert_eq!(v.estimate_minor(), 500);
        assert_eq!(v.constrained_amount_minor(), 500);
    }

    #[test]
    fn negative_outcome_preserves_sign() {
        let v = VariableConsideration::most_likely("discount", -2_000, 1_000);
        assert_eq!(v.constrained_amount_minor(), -1_000);
    }
}
