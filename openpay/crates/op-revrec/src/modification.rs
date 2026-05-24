//! Contract modifications per ASC 606-10-25-10 through 25-13.
//!
//! ASC 606 distinguishes three kinds of contract modification:
//!
//! - **Type I — Separate contract.** When the modification adds
//!   distinct goods/services AND the price reflects the entity's
//!   standalone selling price for those additions. Account for the
//!   modification as a brand-new contract; the original is unaffected.
//!   (ASC 606-10-25-12.)
//!
//! - **Type II — Termination + new contract.** When the remaining
//!   goods/services after the modification are distinct from those
//!   already transferred. Treat as if the original contract were
//!   terminated and a new contract started for the remaining +
//!   modified obligations. Allocation is reset prospectively.
//!   (ASC 606-10-25-13(a).)
//!
//! - **Type III — Cumulative catch-up.** When the remaining
//!   goods/services are NOT distinct from those already transferred
//!   (the modification affects an ongoing single performance
//!   obligation). Adjust revenue cumulatively at the modification
//!   date: recompute the schedule as if the new terms had always
//!   applied, post the catch-up. (ASC 606-10-25-13(b).)
//!
//! The classifier in this module ([`classify`]) implements the
//! decision tree from the standard given the modification's
//! characteristics. Operators with audit-ratified judgements may
//! bypass it and construct [`Modification`] variants directly.

use chrono::NaiveDate;
use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::contract::PerformanceObligation;

/// Properties of a proposed change to an active contract. Drive the
/// Type-I / Type-II / Type-III classification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModificationProposal {
    /// Date the modification becomes effective.
    pub effective_date: NaiveDate,
    /// Any new performance obligations the modification adds.
    pub added_obligations: Vec<PerformanceObligation>,
    /// Change to the transaction price (positive = price increase).
    pub price_change: Money,
    /// True if the entity assesses the added goods/services are
    /// distinct from those in the original contract.
    pub added_are_distinct: bool,
    /// True if the price change reflects the standalone selling
    /// prices of the additions (ASC 606-10-25-12(b)).
    pub price_reflects_ssp: bool,
    /// True if the remaining goods/services after the modification
    /// date are distinct from those already transferred. Drives the
    /// Type-II vs Type-III branch.
    pub remaining_are_distinct: bool,
}

/// Output of [`classify`]: which of the three ASC 606 paths to follow.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modification {
    /// Type I — separate contract. The original contract is unaffected;
    /// the additions go into a new [`crate::contract::Contract`] that
    /// the caller is expected to create.
    SeparateContract {
        /// Date the new contract is effective.
        effective_date: NaiveDate,
        /// Obligations on the new contract.
        new_obligations: Vec<PerformanceObligation>,
        /// Price of the new contract.
        price: Money,
    },
    /// Type II — terminate + new. Allocate the remaining transaction
    /// price (original + change) across remaining obligations
    /// prospectively from `effective_date`.
    TerminateAndNew {
        /// Effective date for the new prospective period.
        effective_date: NaiveDate,
        /// Net change to the transaction price.
        price_change: Money,
        /// New obligations introduced.
        new_obligations: Vec<PerformanceObligation>,
    },
    /// Type III — cumulative catch-up. Recompute the schedule for the
    /// affected obligation under the new terms; post the difference
    /// between previously recognized revenue and what would have been
    /// recognized under the new terms on `effective_date`.
    CumulativeCatchup {
        /// Catch-up date.
        effective_date: NaiveDate,
        /// Net change to the transaction price.
        price_change: Money,
    },
}

/// Apply the ASC 606-10-25-12 / -25-13 decision tree to a proposal.
///
/// 1. If the added obligations are distinct **and** the price reflects
///    SSP → Type I (separate contract).
/// 2. Otherwise, if the remaining obligations are distinct from those
///    already transferred → Type II (terminate + new).
/// 3. Otherwise → Type III (cumulative catch-up).
#[must_use]
pub fn classify(p: &ModificationProposal) -> Modification {
    if p.added_are_distinct && p.price_reflects_ssp {
        Modification::SeparateContract {
            effective_date: p.effective_date,
            new_obligations: p.added_obligations.clone(),
            price: p.price_change,
        }
    } else if p.remaining_are_distinct {
        Modification::TerminateAndNew {
            effective_date: p.effective_date,
            price_change: p.price_change,
            new_obligations: p.added_obligations.clone(),
        }
    } else {
        Modification::CumulativeCatchup {
            effective_date: p.effective_date,
            price_change: p.price_change,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{ObligationId, Presentation, RecognitionPattern};
    use op_core::Currency;

    fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap_or_default()
    }

    fn usd(n: i64) -> Money {
        Money::from_minor(n, Currency::USD)
    }

    fn ob() -> PerformanceObligation {
        PerformanceObligation {
            id: ObligationId::new("addon"),
            standalone_selling_price: usd(1_000),
            pattern: RecognitionPattern::PointInTime { date: ymd(2026, 7, 1) },
            presentation: Presentation::Gross,
        }
    }

    #[test]
    fn distinct_and_ssp_priced_is_type1() {
        let p = ModificationProposal {
            effective_date: ymd(2026, 7, 1),
            added_obligations: vec![ob()],
            price_change: usd(1_000),
            added_are_distinct: true,
            price_reflects_ssp: true,
            remaining_are_distinct: true,
        };
        assert!(matches!(classify(&p), Modification::SeparateContract { .. }));
    }

    #[test]
    fn distinct_remaining_but_not_ssp_priced_is_type2() {
        let p = ModificationProposal {
            effective_date: ymd(2026, 7, 1),
            added_obligations: vec![ob()],
            price_change: usd(500), // discounted
            added_are_distinct: true,
            price_reflects_ssp: false,
            remaining_are_distinct: true,
        };
        assert!(matches!(classify(&p), Modification::TerminateAndNew { .. }));
    }

    #[test]
    fn non_distinct_remaining_is_type3() {
        let p = ModificationProposal {
            effective_date: ymd(2026, 7, 1),
            added_obligations: vec![],
            price_change: usd(1_000),
            added_are_distinct: false,
            price_reflects_ssp: false,
            remaining_are_distinct: false,
        };
        assert!(matches!(classify(&p), Modification::CumulativeCatchup { .. }));
    }
}
