//! Step 1, Step 2 and Step 3 of the ASC 606 / IFRS 15 model:
//! identify the contract, identify the performance obligations,
//! determine the transaction price.
//!
//! - [`Contract`] — the contract envelope: customer, effective date,
//!   currency, the list of [`PerformanceObligation`]s, and the
//!   [`TransactionPrice`].
//! - [`PerformanceObligation`] — one promised good or service. Carries
//!   its standalone selling price (SSP) so [Step 4 — allocation] in
//!   [`crate::schedule`] can split the transaction price by relative SSP.
//! - [`TransactionPrice`] — the fixed + variable parts of what the
//!   customer will pay. Variable consideration is constrained per
//!   [`crate::variable`].

use chrono::NaiveDate;
use op_core::{Currency, Money};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::variable::VariableConsideration;

/// A stable identifier for a contract.
///
/// UUID v7 — time-sortable, so a `BTreeMap<ContractId, _>` orders by
/// creation time, useful for cohort reports.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ContractId(pub Uuid);

impl ContractId {
    /// Fresh time-sortable id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for ContractId {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable id for one performance obligation inside a contract.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ObligationId(pub String);

impl ObligationId {
    /// Build from any string-like; callers may use a meaningful slug
    /// (`"setup-fee"`, `"saas-2026"`) rather than a UUID for legibility.
    pub fn new<S: Into<String>>(s: S) -> Self {
        Self(s.into())
    }
}

/// How a single performance obligation transfers value to the customer.
///
/// ASC 606-10-25-27 distinguishes point-in-time from over-time
/// recognition. Over-time is then further parameterised by the
/// measurement method (input / output / time-elapsed).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecognitionPattern {
    /// Recognized at a single date: the date control transfers (e.g.
    /// hardware shipment, perpetual licence delivery, setup-completion).
    PointInTime {
        /// The single date on which revenue books.
        date: NaiveDate,
    },
    /// Straight-line over a service period (the textbook SaaS case).
    /// Recognition is `amount * elapsed_days / total_days` on each
    /// reporting period boundary.
    StraightLine {
        /// First day of the service period (inclusive).
        start: NaiveDate,
        /// Last day of the service period (inclusive).
        end: NaiveDate,
    },
    /// Output method: progress measured by units delivered / promised.
    /// Each milestone names its expected completion date and the
    /// fraction of progress it represents.
    OutputMilestones {
        /// Ordered milestones. Fractions must sum to 1.0 (validation in
        /// `schedule::generate`).
        milestones: Vec<Milestone>,
    },
    /// Input method: progress measured by costs incurred / total costs.
    /// We accept the calling system's already-computed percent-complete
    /// snapshots rather than tracking costs ourselves.
    InputPercentComplete {
        /// Ordered snapshots: at each date, the cumulative percent
        /// complete (0.0 to 1.0).
        snapshots: Vec<PercentCompleteSnapshot>,
    },
    /// Usage-based: recognized only when the customer consumes. The
    /// schedule has no entries until usage events arrive at runtime;
    /// see [`crate::schedule::usage_recognition`].
    Usage {
        /// Per-unit price (in minor units of the contract currency).
        unit_price_minor: i64,
    },
}

/// One milestone on an output-method obligation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Milestone {
    /// Human-readable label (`"phase-1-uat-signoff"`).
    pub label: String,
    /// Expected completion date.
    pub date: NaiveDate,
    /// Fraction of the obligation this milestone represents, scaled
    /// 0..=1. Stored as `Decimal` so the sum-to-one check is exact.
    pub fraction: Decimal,
}

/// One input-method progress snapshot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PercentCompleteSnapshot {
    /// Date of the snapshot.
    pub date: NaiveDate,
    /// Cumulative percent complete (0..=1).
    pub percent: Decimal,
}

/// Tax presentation under ASC 606-10-55-36 (principal-vs-agent) — see
/// [`crate::principal_agent`] for the indicator framework. A
/// performance obligation that is taxes-collected-as-agent should book
/// the *net* amount as revenue; principal-of-record obligations book
/// gross.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Presentation {
    /// Recognize gross of pass-through amounts (entity is principal).
    Gross,
    /// Recognize net (entity is agent).
    Net,
}

/// One promised good or service inside a contract.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformanceObligation {
    /// Caller-supplied identifier, unique within the contract.
    pub id: ObligationId,
    /// Standalone selling price — the price the entity would charge for
    /// this good/service if sold separately. Drives relative-SSP
    /// allocation of the transaction price (Step 4).
    pub standalone_selling_price: Money,
    /// How and when revenue books.
    pub pattern: RecognitionPattern,
    /// Gross or net presentation. Defaults to `Gross` for the common
    /// principal case.
    pub presentation: Presentation,
}

/// Step 3: the transaction price.
///
/// Split into a fixed component and a (possibly multiple) variable
/// component. The constrained variable amount is computed by
/// [`crate::variable::VariableConsideration::constrained_amount_minor`]
/// and added to the fixed amount to produce the recognizable total.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransactionPrice {
    /// Currency for every part of the transaction price.
    pub currency: Currency,
    /// Fixed consideration in minor units (e.g. the SaaS list price
    /// for the term).
    pub fixed_minor: i64,
    /// Zero or more variable-consideration components (discounts,
    /// rebates, performance bonuses, refund liabilities).
    pub variable: Vec<VariableConsideration>,
}

impl TransactionPrice {
    /// Construct a fixed-only transaction price.
    #[must_use]
    pub const fn fixed(amount: Money) -> Self {
        Self {
            currency: amount.currency,
            fixed_minor: amount.minor_units,
            variable: Vec::new(),
        }
    }

    /// Total recognizable amount = fixed + sum of constrained variables.
    ///
    /// # Errors
    /// - [`Error::Money`] on i64 overflow.
    pub fn total(&self) -> Result<Money> {
        let mut acc = self.fixed_minor;
        for v in &self.variable {
            acc = acc
                .checked_add(v.constrained_amount_minor())
                .ok_or_else(|| Error::Money(op_core::Error::Overflow))?;
        }
        Ok(Money::from_minor(acc, self.currency))
    }
}

/// Step 1: a contract with a customer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Contract {
    /// Stable contract id.
    pub id: ContractId,
    /// Opaque customer reference (UUID, CRM id, etc.).
    pub customer_ref: String,
    /// Date the contract becomes legally enforceable.
    pub effective_date: NaiveDate,
    /// Performance obligations promised under this contract.
    pub obligations: Vec<PerformanceObligation>,
    /// Total consideration the entity expects to be entitled to.
    pub transaction_price: TransactionPrice,
}

impl Contract {
    /// Validate that every obligation's currency matches the
    /// transaction-price currency.
    ///
    /// # Errors
    /// - [`Error::CurrencyMismatch`] if any obligation differs.
    pub fn validate_currencies(&self) -> Result<()> {
        for o in &self.obligations {
            if o.standalone_selling_price.currency != self.transaction_price.currency {
                return Err(Error::CurrencyMismatch);
            }
        }
        Ok(())
    }

    /// Sum of standalone selling prices across obligations, in minor
    /// units. Drives the relative-SSP allocation in
    /// [`crate::schedule::allocate_transaction_price`].
    ///
    /// # Errors
    /// - [`Error::Money`] on overflow.
    pub fn total_ssp_minor(&self) -> Result<i64> {
        let mut acc: i64 = 0;
        for o in &self.obligations {
            acc = acc
                .checked_add(o.standalone_selling_price.minor_units)
                .ok_or_else(|| Error::Money(op_core::Error::Overflow))?;
        }
        Ok(acc)
    }

    /// Lookup a performance obligation by id.
    ///
    /// # Errors
    /// - [`Error::UnknownObligation`] if not found.
    pub fn obligation(&self, id: &ObligationId) -> Result<&PerformanceObligation> {
        self.obligations
            .iter()
            .find(|o| &o.id == id)
            .ok_or_else(|| Error::UnknownObligation(id.0.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    fn usd(n: i64) -> Money {
        Money::from_minor(n, Currency::USD)
    }

    #[test]
    fn fixed_transaction_price_total_equals_fixed() {
        let tp = TransactionPrice::fixed(usd(10_000));
        assert_eq!(tp.total().unwrap(), usd(10_000));
    }

    #[test]
    fn currency_mismatch_detected() {
        let c = Contract {
            id: ContractId::new(),
            customer_ref: "cust_1".into(),
            effective_date: NaiveDate::from_ymd_opt(2026, 1, 1).unwrap_or_default(),
            obligations: vec![PerformanceObligation {
                id: ObligationId::new("saas"),
                standalone_selling_price: Money::from_minor(12_000, Currency::EUR),
                pattern: RecognitionPattern::StraightLine {
                    start: NaiveDate::from_ymd_opt(2026, 1, 1).unwrap_or_default(),
                    end: NaiveDate::from_ymd_opt(2026, 12, 31).unwrap_or_default(),
                },
                presentation: Presentation::Gross,
            }],
            transaction_price: TransactionPrice::fixed(usd(12_000)),
        };
        assert!(matches!(c.validate_currencies(), Err(Error::CurrencyMismatch)));
    }
}
