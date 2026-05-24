//! Split-payments engine.
//!
//! A single inbound payment can be split into many outbound legs. The
//! canonical case for a marketplace: a $100 ticket sale splits into
//! $95 to the event organiser (sub-merchant) and $5 to the marketplace
//! (platform fee). Three-way splits add an affiliate / third-party
//! recipient leg.
//!
//! ## Invariants
//!
//! 1. `sum(legs.amount) == source.amount`.
//! 2. Every leg has the same currency as the source. Cross-currency
//!    splits require an FX leg upstream (out of scope for this module;
//!    see `op-fx`).
//! 3. No leg may have a negative amount.
//! 4. At most one leg per `destination` (operators wanting two pays
//!    to the same destination should aggregate first).
//!
//! Fee legs are tagged (`is_fee: true`) so downstream reporting
//! (1099-K, ledger entries, analytics) can separate platform revenue
//! from pass-through.

use std::collections::BTreeSet;

use op_core::{Currency, Money};
use serde::{Deserialize, Serialize};

use crate::account::AccountId;
use crate::error::{Error, Result};

/// Identifier for the source payment being split.
///
/// Local newtype — production callers pull this from `op-core::payment`
/// or their own payment-id namespace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PaymentId(pub String);

impl PaymentId {
    /// Borrow the inner string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A single outbound leg of a payment split.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitLeg {
    /// Connected account receiving this leg.
    pub destination: AccountId,
    /// Amount being routed to `destination`.
    pub amount: Money,
    /// Currency (denormalised from `amount.currency` for clearer JSON;
    /// validated to match at build time).
    pub currency: Currency,
    /// Free-text description (appears on the destination's ledger entry).
    pub description: String,
    /// True if this leg is platform revenue rather than pass-through.
    pub is_fee: bool,
    /// Optional grouping key for reporting (e.g. campaign id, batch id).
    pub transfer_group: Option<String>,
}

impl SplitLeg {
    /// Construct a leg, validating currency-coherence between
    /// `amount.currency` and the standalone `currency` field.
    ///
    /// # Errors
    /// [`Error::CurrencyMismatch`] if the two disagree.
    pub fn try_new(
        destination: AccountId,
        amount: Money,
        description: impl Into<String>,
        is_fee: bool,
        transfer_group: Option<String>,
    ) -> Result<Self> {
        let currency = amount.currency;
        Ok(Self {
            destination,
            amount,
            currency,
            description: description.into(),
            is_fee,
            transfer_group,
        })
    }
}

/// A complete split-payment plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaymentSplit {
    /// The inbound payment whose proceeds are being distributed.
    pub source_payment_id: PaymentId,
    /// Outbound legs.
    pub splits: Vec<SplitLeg>,
}

impl PaymentSplit {
    /// Validate the split against the source amount.
    ///
    /// # Errors
    /// - [`Error::InvalidSplit`] if `legs` is empty, contains negatives,
    ///   contains duplicate destinations, or doesn't sum to `source_amount`.
    /// - [`Error::CurrencyMismatch`] if any leg's currency disagrees with
    ///   `source_amount.currency`.
    /// - [`Error::Overflow`] if the leg amounts overflow on summation.
    pub fn validate(&self, source_amount: Money) -> Result<()> {
        if self.splits.is_empty() {
            return Err(Error::InvalidSplit {
                reason: "no legs".into(),
            });
        }

        // Negative legs?
        for (i, leg) in self.splits.iter().enumerate() {
            if leg.amount.minor_units < 0 {
                return Err(Error::InvalidSplit {
                    reason: format!("leg {i} has negative amount {}", leg.amount.minor_units),
                });
            }
            if leg.currency != source_amount.currency {
                return Err(Error::CurrencyMismatch(format!(
                    "leg {i} currency {} != source currency {}",
                    leg.currency, source_amount.currency
                )));
            }
            if leg.amount.currency != leg.currency {
                return Err(Error::CurrencyMismatch(format!(
                    "leg {i} self-inconsistent: amount.currency={} but leg.currency={}",
                    leg.amount.currency, leg.currency
                )));
            }
        }

        // Duplicate destinations?
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for leg in &self.splits {
            if !seen.insert(leg.destination.as_str()) {
                return Err(Error::InvalidSplit {
                    reason: format!(
                        "duplicate destination {} in split",
                        leg.destination.as_str()
                    ),
                });
            }
        }

        // Sum.
        let mut total = Money::zero(source_amount.currency);
        for leg in &self.splits {
            total = total
                .checked_add(leg.amount)
                .map_err(|_| Error::Overflow)?;
        }
        if total != source_amount {
            return Err(Error::InvalidSplit {
                reason: format!(
                    "split legs sum to {} but source is {}",
                    total, source_amount
                ),
            });
        }

        Ok(())
    }

    /// Sum of fee legs in `currency`.
    ///
    /// # Errors
    /// [`Error::Overflow`] if the sum overflows.
    pub fn fee_total(&self, currency: Currency) -> Result<Money> {
        let mut total = Money::zero(currency);
        for leg in &self.splits {
            if leg.is_fee && leg.currency == currency {
                total = total
                    .checked_add(leg.amount)
                    .map_err(|_| Error::Overflow)?;
            }
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acct(s: &str) -> AccountId {
        AccountId(s.into())
    }

    fn usd(minor: i64) -> Money {
        Money::from_minor(minor, Currency::USD)
    }

    #[test]
    fn even_split_validates() {
        let split = PaymentSplit {
            source_payment_id: PaymentId("pay_1".into()),
            splits: vec![
                SplitLeg::try_new(acct("acct_merchant"), usd(9500), "merchant payout", false, None)
                    .expect("ok"),
                SplitLeg::try_new(acct("acct_platform"), usd(500), "platform fee", true, None)
                    .expect("ok"),
            ],
        };
        split.validate(usd(10_000)).expect("validates");
        assert_eq!(split.fee_total(Currency::USD).expect("ok"), usd(500));
    }

    #[test]
    fn oversum_fails() {
        let split = PaymentSplit {
            source_payment_id: PaymentId("pay_2".into()),
            splits: vec![
                SplitLeg::try_new(acct("acct_merchant"), usd(9500), "m", false, None)
                    .expect("ok"),
                SplitLeg::try_new(acct("acct_platform"), usd(600), "p", true, None)
                    .expect("ok"),
            ],
        };
        let err = split.validate(usd(10_000)).expect_err("oversum");
        assert!(matches!(err, Error::InvalidSplit { .. }));
    }

    #[test]
    fn negative_leg_fails() {
        let split = PaymentSplit {
            source_payment_id: PaymentId("pay_3".into()),
            splits: vec![SplitLeg::try_new(
                acct("acct_x"),
                usd(-100),
                "bad",
                false,
                None,
            )
            .expect("ok")],
        };
        let err = split.validate(usd(-100)).expect_err("negative");
        assert!(matches!(err, Error::InvalidSplit { .. }));
    }

    #[test]
    fn duplicate_destinations_fail() {
        let split = PaymentSplit {
            source_payment_id: PaymentId("pay_4".into()),
            splits: vec![
                SplitLeg::try_new(acct("acct_m"), usd(5000), "a", false, None).expect("ok"),
                SplitLeg::try_new(acct("acct_m"), usd(5000), "b", false, None).expect("ok"),
            ],
        };
        let err = split.validate(usd(10_000)).expect_err("dupe");
        assert!(matches!(err, Error::InvalidSplit { .. }));
    }
}
