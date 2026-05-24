//! Typed fee accounting.
//!
//! Real-world payment fees fall into a small number of named buckets,
//! each governed by a different party with a different settlement
//! cadence. Conflating them ("just a fee") makes statement reconciliation
//! impossible; the merchant sees one PSP fee, the PSP owes 4 different
//! parties a piece of it, the regulator wants to see interchange broken
//! out separately, and the audit needs every bucket on its own line.

use op_core::{Money, Currency};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Canonical fee buckets.
///
/// Aligned with the way modern PSP statements (Stripe, Adyen,
/// Checkout.com) break out fees, plus the buckets specific to BNPL
/// providers (Klarna, Affirm) and FX intermediaries.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeeBucket {
    /// Interchange paid to the issuer (Visa / Mastercard rate sheets;
    /// EFTPOS equivalents). Largest component of card-payment fees.
    Interchange,
    /// Scheme fee paid to the card network (Visa, Mastercard, Amex,
    /// Discover) — assessment, network usage, cross-border.
    Scheme,
    /// Acquirer / processor markup — the PSP's own margin.
    Acquirer,
    /// FX spread / conversion fee on cross-currency transactions.
    Fx,
    /// Settlement-network fee (FedNow, RTP, SEPA Instant operator
    /// fees; ACH or wire ticket charges).
    SettlementNetwork,
    /// BNPL provider commission (Klarna, Affirm, Afterpay).
    Bnpl,
    /// Other / catch-all for rail-specific fees that don't slot
    /// elsewhere. Always provide a meaningful `code` when using.
    Other,
}

impl FeeBucket {
    /// Human-readable label for renderers.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Interchange => "Interchange",
            Self::Scheme => "Scheme",
            Self::Acquirer => "Acquirer",
            Self::Fx => "FX",
            Self::SettlementNetwork => "Settlement Network",
            Self::Bnpl => "BNPL",
            Self::Other => "Other",
        }
    }
}

/// One fee line.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeeLine {
    /// Which bucket.
    pub bucket: FeeBucket,
    /// A rail-specific code (e.g. Visa "CPS / Retail" interchange
    /// program code, or a PSP fee code).
    pub code: Option<String>,
    /// Magnitude amount in the line's currency.
    pub amount: Money,
    /// The underlying transaction id this fee accrued against
    /// (op-ledger external_id or rail reference).
    pub against_external_id: Option<String>,
}

impl FeeLine {
    /// Construct.
    #[must_use]
    pub fn new(bucket: FeeBucket, amount: Money) -> Self {
        Self {
            bucket,
            code: None,
            amount: Money {
                minor_units: amount.minor_units.abs(),
                currency: amount.currency,
            },
            against_external_id: None,
        }
    }

    /// Builder: rail-specific code.
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Builder: cross-reference to the underlying transaction.
    #[must_use]
    pub fn with_against(mut self, external_id: impl Into<String>) -> Self {
        self.against_external_id = Some(external_id.into());
        self
    }
}

/// A fee schedule: declarative rate sheet that turns a captured
/// transaction into a [`FeeLine`].
///
/// Each rule is `bucket × percent_bps + flat`. A captured transaction
/// runs through every rule whose `applies_when` predicate returns
/// `true`, accruing a [`FeeLine`] per matching rule.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeeSchedule {
    /// Currency this schedule's flat amounts are denominated in.
    pub currency: Currency,
    /// Rules in evaluation order.
    pub rules: Vec<FeeRule>,
}

/// One rule in a [`FeeSchedule`].
///
/// `percent_bps` is in basis points (1 bps = 0.01%). The fee on a
/// `gross` capture is `gross * percent_bps / 10_000 + flat`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeeRule {
    /// Which bucket this rule produces.
    pub bucket: FeeBucket,
    /// Rail-specific code (echoed onto the resulting [`FeeLine`]).
    pub code: Option<String>,
    /// Variable component in basis points (1/100 of a percent).
    pub percent_bps: u32,
    /// Flat component in the schedule's currency (minor units).
    pub flat_minor: i64,
}

impl FeeRule {
    /// Construct.
    #[must_use]
    pub const fn new(bucket: FeeBucket, percent_bps: u32, flat_minor: i64) -> Self {
        Self {
            bucket,
            code: None,
            percent_bps,
            flat_minor,
        }
    }

    /// Builder: attach a rail-specific code.
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Compute this rule's fee against a gross amount in
    /// `schedule_currency`. Returns the magnitude.
    fn compute(&self, gross_minor: i64) -> Result<i64> {
        let pct_part = i128::from(gross_minor.abs())
            .checked_mul(i128::from(self.percent_bps))
            .ok_or(Error::Overflow)?
            / 10_000;
        let total = pct_part
            .checked_add(i128::from(self.flat_minor))
            .ok_or(Error::Overflow)?;
        i64::try_from(total).map_err(|_| Error::Overflow)
    }
}

impl FeeSchedule {
    /// Empty schedule in the given currency.
    #[must_use]
    pub fn new(currency: Currency) -> Self {
        Self {
            currency,
            rules: Vec::new(),
        }
    }

    /// Builder: append a rule.
    #[must_use]
    pub fn with_rule(mut self, rule: FeeRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Apply this schedule to a captured gross amount, producing one
    /// [`FeeLine`] per rule. The gross amount's currency must match
    /// the schedule's currency.
    ///
    /// # Errors
    /// - [`Error::CurrencyMismatch`] on currency mismatch.
    /// - [`Error::Overflow`] on i64 saturation.
    pub fn accrue(
        &self,
        gross: Money,
        against_external_id: Option<&str>,
    ) -> Result<Vec<FeeLine>> {
        if gross.currency != self.currency {
            return Err(Error::CurrencyMismatch {
                line: gross.currency.code().to_owned(),
                statement: self.currency.code().to_owned(),
            });
        }
        let mut out = Vec::with_capacity(self.rules.len());
        for rule in &self.rules {
            let amount_minor = rule.compute(gross.minor_units)?;
            if amount_minor == 0 {
                continue;
            }
            let mut line = FeeLine::new(rule.bucket, Money::from_minor(amount_minor, self.currency));
            if let Some(code) = rule.code.clone() {
                line.code = Some(code);
            }
            if let Some(ext) = against_external_id {
                line.against_external_id = Some(ext.to_owned());
            }
            out.push(line);
        }
        Ok(out)
    }

    /// Sum the magnitude across a slice of fee lines, grouped by
    /// bucket. The returned vec is in declaration order of buckets the
    /// caller passed.
    #[must_use]
    pub fn group_by_bucket(lines: &[FeeLine]) -> Vec<(FeeBucket, i64)> {
        let mut out: Vec<(FeeBucket, i64)> = Vec::new();
        for line in lines {
            if let Some(slot) = out.iter_mut().find(|(b, _)| *b == line.bucket) {
                slot.1 = slot.1.saturating_add(line.amount.minor_units);
            } else {
                out.push((line.bucket, line.amount.minor_units));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interchange_2_9_pct_plus_30_cents() {
        // Stripe-classic: 2.9% + 30c on a $100 capture = $3.20.
        let sched = FeeSchedule::new(Currency::USD).with_rule(FeeRule::new(
            FeeBucket::Acquirer,
            290, // 2.90%
            30,  // 30 cents
        ));
        let fees = sched
            .accrue(Money::from_minor(10_000, Currency::USD), Some("ord-1"))
            .unwrap();
        assert_eq!(fees.len(), 1);
        assert_eq!(fees[0].amount.minor_units, 320);
        assert_eq!(fees[0].bucket, FeeBucket::Acquirer);
    }

    #[test]
    fn currency_mismatch_rejected() {
        let sched = FeeSchedule::new(Currency::USD).with_rule(FeeRule::new(FeeBucket::Scheme, 5, 0));
        let r = sched.accrue(Money::from_minor(100, Currency::EUR), None);
        assert!(matches!(r, Err(Error::CurrencyMismatch { .. })));
    }

    #[test]
    fn zero_amount_rule_is_skipped() {
        let sched = FeeSchedule::new(Currency::USD)
            .with_rule(FeeRule::new(FeeBucket::Scheme, 0, 0))
            .with_rule(FeeRule::new(FeeBucket::Acquirer, 100, 0)); // 1%
        let fees = sched
            .accrue(Money::from_minor(10_000, Currency::USD), None)
            .unwrap();
        assert_eq!(fees.len(), 1);
        assert_eq!(fees[0].bucket, FeeBucket::Acquirer);
    }

    #[test]
    fn group_by_bucket_sums() {
        let lines = vec![
            FeeLine::new(FeeBucket::Interchange, Money::from_minor(100, Currency::USD)),
            FeeLine::new(FeeBucket::Interchange, Money::from_minor(50, Currency::USD)),
            FeeLine::new(FeeBucket::Scheme, Money::from_minor(10, Currency::USD)),
        ];
        let g = FeeSchedule::group_by_bucket(&lines);
        assert_eq!(g.len(), 2);
        assert_eq!(g[0], (FeeBucket::Interchange, 150));
        assert_eq!(g[1], (FeeBucket::Scheme, 10));
    }
}
