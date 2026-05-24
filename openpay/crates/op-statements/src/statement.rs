//! The merchant-facing [`Statement`] aggregate.
//!
//! A statement is a closed, immutable snapshot of money movement for a
//! merchant over a [`Period`](crate::cadence::Period). It is the
//! aggregate that every downstream serializer
//! ([`RenderTarget`](crate::render::RenderTarget),
//! [`Camt053Builder`](crate::iso20022::Camt053Builder),
//! [`Bai2Writer`](crate::bai2::Bai2Writer),
//! [`Mt940Writer`](crate::mt940::Mt940Writer)) walks.
//!
//! ## Stripe-style structure
//!
//! We model the same shape Stripe daily statements ship: a per-currency
//! aggregate (gross volume, refunds, chargebacks, fees, payouts) plus
//! the line-item detail behind it. Operators on other rails populate
//! the same shape; only the fee buckets differ.

use op_core::{Currency, Money};
use serde::{Deserialize, Serialize};

use crate::cadence::Period;
use crate::error::{Error, Result};
use crate::fees::FeeLine;

/// What category a [`StatementLine`] represents.
///
/// Stripe-style buckets that map cleanly onto bank-statement lines as
/// well as PSP daily reports. The kind is what distinguishes a refund
/// from a gross capture in the aggregate roll-up.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StatementLineKind {
    /// Captured payment from a customer (gross volume).
    GrossCapture,
    /// Refund paid back to a customer (negative against gross).
    Refund,
    /// Chargeback / dispute debit (negative against gross).
    Chargeback,
    /// Fee charged by the acquirer / scheme / BNPL provider. Carries a
    /// [`FeeLine`] attached.
    Fee,
    /// Payout sent to the merchant bank account (negative against
    /// retained balance).
    Payout,
    /// Adjustment / reserve / sundry. Sign determined by the line's
    /// amount.
    Adjustment,
}

impl StatementLineKind {
    /// Does this kind contribute to the merchant's gross volume tally?
    #[must_use]
    pub const fn is_gross_volume(self) -> bool {
        matches!(self, Self::GrossCapture)
    }

    /// Does this kind reduce the merchant's retained balance?
    #[must_use]
    pub const fn is_outflow(self) -> bool {
        matches!(
            self,
            Self::Refund | Self::Chargeback | Self::Fee | Self::Payout
        )
    }
}

/// One line on a statement.
///
/// Lines are signed-magnitude: `amount` is always non-negative,
/// `kind` carries the semantic direction. Downstream aggregation
/// flips signs per [`StatementLineKind::is_outflow`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatementLine {
    /// Stable id, unique within the statement.
    pub id: String,
    /// Kind / bucket.
    pub kind: StatementLineKind,
    /// Magnitude amount in some currency.
    pub amount: Money,
    /// Unix epoch seconds when the line posted.
    pub posted_at_unix_secs: u64,
    /// Optional cross-reference to an op-ledger transaction's
    /// `external_id` or a rail-level reference.
    pub external_id: Option<String>,
    /// If `kind == Fee`, the structured fee line.
    pub fee: Option<FeeLine>,
    /// Free-form passthrough preserved for the renderer.
    pub metadata: Vec<(String, String)>,
}

impl StatementLine {
    /// Construct a line. `amount` is normalized to its magnitude; the
    /// sign is carried by [`StatementLineKind`].
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        kind: StatementLineKind,
        amount: Money,
        posted_at_unix_secs: u64,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            amount: Money {
                minor_units: amount.minor_units.abs(),
                currency: amount.currency,
            },
            posted_at_unix_secs,
            external_id: None,
            fee: None,
            metadata: Vec::new(),
        }
    }

    /// Builder: attach external id.
    #[must_use]
    pub fn with_external_id(mut self, id: impl Into<String>) -> Self {
        self.external_id = Some(id.into());
        self
    }

    /// Builder: attach a structured fee line. Only meaningful when
    /// `kind == Fee`.
    #[must_use]
    pub fn with_fee(mut self, fee: FeeLine) -> Self {
        self.fee = Some(fee);
        self
    }

    /// Builder: attach a metadata pair.
    #[must_use]
    pub fn with_metadata(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.metadata.push((k.into(), v.into()));
        self
    }

    /// Signed contribution of this line to the ending balance.
    ///
    /// Gross captures increase balance; refunds, chargebacks, fees,
    /// and payouts decrease it. Adjustments use the caller-supplied
    /// sign embedded in the raw minor-units (`None` if the line was
    /// normalized to magnitude — adjustments should pre-set the sign).
    #[must_use]
    pub fn signed_minor_units(&self) -> i64 {
        let mag = self.amount.minor_units.abs();
        match self.kind {
            StatementLineKind::GrossCapture => mag,
            StatementLineKind::Refund
            | StatementLineKind::Chargeback
            | StatementLineKind::Fee
            | StatementLineKind::Payout => -mag,
            StatementLineKind::Adjustment => self.amount.minor_units,
        }
    }
}

/// FX rate carried per non-primary currency on a statement.
///
/// `minor_per_unit_q` is a fixed-point fraction: it expresses how many
/// minor units of the **primary** currency one whole unit of the
/// **foreign** currency converts to, scaled by `10^scale`. Using
/// scaled integers preserves the no-`f64`-money invariant.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FxRate {
    /// The foreign currency.
    pub foreign: Currency,
    /// The primary currency this rate converts INTO.
    pub primary: Currency,
    /// Numerator. `1 foreign` = `minor_per_unit_q / 10^scale` primary
    /// minor units.
    pub minor_per_unit_q: i128,
    /// Scale factor (power of ten).
    pub scale: u8,
    /// Rate-effective unix epoch seconds. Caller-supplied.
    pub as_of_unix_secs: u64,
}

impl FxRate {
    /// Convert a foreign-currency [`Money`] to the primary currency's
    /// minor units. Saturates on overflow (statement rendering must
    /// not panic on adversarial rates).
    #[must_use]
    pub fn convert_minor(&self, foreign_amount: Money) -> i64 {
        if foreign_amount.currency != self.foreign {
            return 0;
        }
        let foreign_minor = i128::from(foreign_amount.minor_units);
        let foreign_exp = i32::from(foreign_amount.currency.exponent());
        let primary_exp = i32::from(self.primary.exponent());

        // (foreign_minor / 10^foreign_exp) * (minor_per_unit_q / 10^scale)
        //   * (10^primary_exp / 1)
        // = foreign_minor * minor_per_unit_q
        //   / 10^(foreign_exp + scale - primary_exp)
        let numerator = match foreign_minor.checked_mul(self.minor_per_unit_q) {
            Some(n) => n,
            None => return 0,
        };
        let denom_exp = foreign_exp + i32::from(self.scale) - primary_exp;
        let result = if denom_exp >= 0 {
            let d = 10_i128.pow(u32::try_from(denom_exp).unwrap_or(0));
            numerator / d
        } else {
            let m = 10_i128.pow(u32::try_from(-denom_exp).unwrap_or(0));
            match numerator.checked_mul(m) {
                Some(n) => n,
                None => return 0,
            }
        };
        i64::try_from(result).unwrap_or(0)
    }
}

/// Per-currency rolled-up totals for a statement.
///
/// Computed from [`StatementLine`]s by [`Statement::aggregate`]. The
/// invariant is `ending = opening + gross_volume - refunds -
/// chargebacks - fees - payouts + adjustments`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrencyAggregate {
    /// The currency these totals are denominated in.
    pub currency: Currency,
    /// Opening balance for the period.
    pub opening: Money,
    /// Sum of [`StatementLineKind::GrossCapture`] line magnitudes.
    pub gross_volume: Money,
    /// Sum of refund line magnitudes.
    pub refunds: Money,
    /// Sum of chargeback line magnitudes.
    pub chargebacks: Money,
    /// Sum of fee line magnitudes.
    pub fees: Money,
    /// Sum of payout line magnitudes.
    pub payouts: Money,
    /// Net adjustments (signed).
    pub adjustments: Money,
    /// Ending balance = opening + gross - refunds - chargebacks - fees
    /// - payouts + adjustments.
    pub ending: Money,
}

impl CurrencyAggregate {
    /// Zero aggregate in the given currency.
    #[must_use]
    pub fn zero(currency: Currency) -> Self {
        let z = Money::zero(currency);
        Self {
            currency,
            opening: z,
            gross_volume: z,
            refunds: z,
            chargebacks: z,
            fees: z,
            payouts: z,
            adjustments: z,
            ending: z,
        }
    }
}

/// Opening / ending balance snapshot in the primary currency.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BalanceSnapshot {
    /// Opening balance in primary currency minor units.
    pub opening: Money,
    /// Ending balance in primary currency minor units.
    pub ending: Money,
}

/// A merchant statement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Statement {
    /// Stable id (caller-supplied; merchant statement number).
    pub id: String,
    /// Merchant account / merchant-of-record identifier.
    pub merchant_id: String,
    /// Reporting period.
    pub period: Period,
    /// Primary currency. All cross-currency totals are FX-adjusted to
    /// this currency in [`Self::primary_aggregate`].
    pub primary_currency: Currency,
    /// All line items.
    pub lines: Vec<StatementLine>,
    /// Per-currency aggregates, keyed by currency code.
    pub aggregates: Vec<CurrencyAggregate>,
    /// FX rates available for cross-currency aggregation.
    pub fx_rates: Vec<FxRate>,
    /// Free-form passthrough.
    pub metadata: Vec<(String, String)>,
}

impl Statement {
    /// Construct a statement skeleton. Aggregates start at zero in the
    /// primary currency; call [`Self::aggregate`] after attaching
    /// lines.
    ///
    /// # Errors
    /// [`Error::InvalidPeriod`] if `period.end < period.start`.
    pub fn new(
        id: impl Into<String>,
        merchant_id: impl Into<String>,
        period: Period,
        primary_currency: Currency,
    ) -> Result<Self> {
        if period.end_unix_secs < period.start_unix_secs {
            return Err(Error::InvalidPeriod {
                start: period.start_unix_secs,
                end: period.end_unix_secs,
            });
        }
        Ok(Self {
            id: id.into(),
            merchant_id: merchant_id.into(),
            period,
            primary_currency,
            lines: Vec::new(),
            aggregates: vec![CurrencyAggregate::zero(primary_currency)],
            fx_rates: Vec::new(),
            metadata: Vec::new(),
        })
    }

    /// Attach the opening balance in the primary currency. Idempotent.
    pub fn with_opening(mut self, opening: Money) -> Result<Self> {
        if opening.currency != self.primary_currency {
            return Err(Error::CurrencyMismatch {
                line: opening.currency.code().to_owned(),
                statement: self.primary_currency.code().to_owned(),
            });
        }
        // Ensure the primary aggregate exists and update opening.
        if let Some(agg) = self
            .aggregates
            .iter_mut()
            .find(|a| a.currency == self.primary_currency)
        {
            agg.opening = opening;
            agg.ending = opening;
        }
        Ok(self)
    }

    /// Builder: register an FX rate for cross-currency aggregation.
    #[must_use]
    pub fn with_fx_rate(mut self, rate: FxRate) -> Self {
        self.fx_rates.push(rate);
        self
    }

    /// Append a line. Rejects duplicate ids.
    ///
    /// # Errors
    /// [`Error::DuplicateLineId`] on collision.
    pub fn push_line(&mut self, line: StatementLine) -> Result<()> {
        if self.lines.iter().any(|l| l.id == line.id) {
            return Err(Error::DuplicateLineId(line.id));
        }
        self.lines.push(line);
        Ok(())
    }

    /// Recompute [`Self::aggregates`] from [`Self::lines`].
    ///
    /// Multi-currency safe: produces one [`CurrencyAggregate`] per
    /// distinct currency observed. The primary-currency aggregate is
    /// always present even if no primary-currency lines exist (its
    /// `gross_volume` etc. stay zero).
    ///
    /// # Errors
    /// [`Error::Overflow`] on i64 saturation during summation.
    pub fn aggregate(&mut self) -> Result<()> {
        // Preserve any caller-supplied opening on the primary
        // aggregate before we rebuild.
        let opening = self
            .aggregates
            .iter()
            .find(|a| a.currency == self.primary_currency)
            .map_or_else(|| Money::zero(self.primary_currency), |a| a.opening);

        let mut by_currency: Vec<CurrencyAggregate> = Vec::new();
        by_currency.push(CurrencyAggregate::zero(self.primary_currency));
        let primary_currency = self.primary_currency;

        for line in &self.lines {
            let cur = line.amount.currency;
            let idx = match by_currency.iter().position(|a| a.currency == cur) {
                Some(i) => i,
                None => {
                    by_currency.push(CurrencyAggregate::zero(cur));
                    by_currency.len() - 1
                }
            };
            let agg = &mut by_currency[idx];
            let mag = line.amount.minor_units.abs();
            match line.kind {
                StatementLineKind::GrossCapture => {
                    agg.gross_volume.minor_units = agg
                        .gross_volume
                        .minor_units
                        .checked_add(mag)
                        .ok_or(Error::Overflow)?;
                }
                StatementLineKind::Refund => {
                    agg.refunds.minor_units = agg
                        .refunds
                        .minor_units
                        .checked_add(mag)
                        .ok_or(Error::Overflow)?;
                }
                StatementLineKind::Chargeback => {
                    agg.chargebacks.minor_units = agg
                        .chargebacks
                        .minor_units
                        .checked_add(mag)
                        .ok_or(Error::Overflow)?;
                }
                StatementLineKind::Fee => {
                    agg.fees.minor_units = agg
                        .fees
                        .minor_units
                        .checked_add(mag)
                        .ok_or(Error::Overflow)?;
                }
                StatementLineKind::Payout => {
                    agg.payouts.minor_units = agg
                        .payouts
                        .minor_units
                        .checked_add(mag)
                        .ok_or(Error::Overflow)?;
                }
                StatementLineKind::Adjustment => {
                    agg.adjustments.minor_units = agg
                        .adjustments
                        .minor_units
                        .checked_add(line.amount.minor_units)
                        .ok_or(Error::Overflow)?;
                }
            }
        }

        // Now compute ending balances per currency.
        for agg in &mut by_currency {
            let starting = if agg.currency == primary_currency {
                opening.minor_units
            } else {
                0
            };
            agg.opening = Money::from_minor(starting, agg.currency);
            let mut ending = starting;
            ending = ending
                .checked_add(agg.gross_volume.minor_units)
                .ok_or(Error::Overflow)?;
            ending = ending
                .checked_sub(agg.refunds.minor_units)
                .ok_or(Error::Overflow)?;
            ending = ending
                .checked_sub(agg.chargebacks.minor_units)
                .ok_or(Error::Overflow)?;
            ending = ending
                .checked_sub(agg.fees.minor_units)
                .ok_or(Error::Overflow)?;
            ending = ending
                .checked_sub(agg.payouts.minor_units)
                .ok_or(Error::Overflow)?;
            ending = ending
                .checked_add(agg.adjustments.minor_units)
                .ok_or(Error::Overflow)?;
            agg.ending = Money::from_minor(ending, agg.currency);
        }

        self.aggregates = by_currency;
        Ok(())
    }

    /// The primary-currency aggregate (always present after
    /// [`Self::aggregate`]).
    #[must_use]
    pub fn primary_aggregate(&self) -> &CurrencyAggregate {
        // The constructor ensures this is always present. We fall back
        // to a defensive zero rather than panicking if a caller
        // hand-mutated `aggregates` to empty.
        self.aggregates
            .iter()
            .find(|a| a.currency == self.primary_currency)
            .unwrap_or_else(|| {
                // SAFETY: not reachable under documented invariants;
                // we deliberately leak a static zero here. The check
                // above keeps the happy path branch-free.
                debug_assert!(false, "primary aggregate missing");
                &PRIMARY_FALLBACK
            })
    }

    /// FX-adjust all non-primary aggregates to the primary currency,
    /// summing into one cross-currency [`BalanceSnapshot`]. Missing
    /// rates contribute zero (lossy, by design — the operator who
    /// hasn't supplied a rate has implicitly told us they don't want
    /// cross-currency totals).
    #[must_use]
    pub fn primary_snapshot(&self) -> BalanceSnapshot {
        let primary = self.primary_aggregate();
        let mut opening = primary.opening.minor_units;
        let mut ending = primary.ending.minor_units;
        for agg in &self.aggregates {
            if agg.currency == self.primary_currency {
                continue;
            }
            if let Some(rate) = self
                .fx_rates
                .iter()
                .find(|r| r.foreign == agg.currency && r.primary == self.primary_currency)
            {
                opening = opening.saturating_add(rate.convert_minor(agg.opening));
                ending = ending.saturating_add(rate.convert_minor(agg.ending));
            }
        }
        BalanceSnapshot {
            opening: Money::from_minor(opening, self.primary_currency),
            ending: Money::from_minor(ending, self.primary_currency),
        }
    }
}

// Defensive static fallback for `primary_aggregate()` — never
// observable under documented usage; exists so the public surface
// doesn't have to return `Option<&_>` for what is structurally
// guaranteed.
static PRIMARY_FALLBACK: CurrencyAggregate = CurrencyAggregate {
    currency: Currency::USD,
    opening: Money {
        minor_units: 0,
        currency: Currency::USD,
    },
    gross_volume: Money {
        minor_units: 0,
        currency: Currency::USD,
    },
    refunds: Money {
        minor_units: 0,
        currency: Currency::USD,
    },
    chargebacks: Money {
        minor_units: 0,
        currency: Currency::USD,
    },
    fees: Money {
        minor_units: 0,
        currency: Currency::USD,
    },
    payouts: Money {
        minor_units: 0,
        currency: Currency::USD,
    },
    adjustments: Money {
        minor_units: 0,
        currency: Currency::USD,
    },
    ending: Money {
        minor_units: 0,
        currency: Currency::USD,
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cadence::Period;
    use op_core::Currency;

    fn p() -> Period {
        Period::new(1_700_000_000, 1_700_086_400).unwrap()
    }

    #[test]
    fn statement_new_validates_period() {
        let r = Statement::new(
            "S1",
            "M1",
            Period {
                start_unix_secs: 100,
                end_unix_secs: 50,
            },
            Currency::USD,
        );
        assert!(matches!(r, Err(Error::InvalidPeriod { .. })));
    }

    #[test]
    fn aggregate_rolls_up_per_currency() {
        let mut s = Statement::new("S1", "M1", p(), Currency::USD).unwrap();
        s.push_line(StatementLine::new(
            "l1",
            StatementLineKind::GrossCapture,
            Money::from_minor(10_000, Currency::USD),
            p().start_unix_secs,
        ))
        .unwrap();
        s.push_line(StatementLine::new(
            "l2",
            StatementLineKind::Fee,
            Money::from_minor(290, Currency::USD),
            p().start_unix_secs,
        ))
        .unwrap();
        s.push_line(StatementLine::new(
            "l3",
            StatementLineKind::Refund,
            Money::from_minor(1_000, Currency::USD),
            p().start_unix_secs,
        ))
        .unwrap();
        s.aggregate().unwrap();
        let a = s.primary_aggregate();
        assert_eq!(a.gross_volume.minor_units, 10_000);
        assert_eq!(a.refunds.minor_units, 1_000);
        assert_eq!(a.fees.minor_units, 290);
        // 0 + 10000 - 1000 - 290 - 0 = 8710
        assert_eq!(a.ending.minor_units, 8_710);
    }

    #[test]
    fn duplicate_line_id_rejected() {
        let mut s = Statement::new("S1", "M1", p(), Currency::USD).unwrap();
        s.push_line(StatementLine::new(
            "dup",
            StatementLineKind::GrossCapture,
            Money::from_minor(1, Currency::USD),
            0,
        ))
        .unwrap();
        let r = s.push_line(StatementLine::new(
            "dup",
            StatementLineKind::Fee,
            Money::from_minor(1, Currency::USD),
            0,
        ));
        assert!(matches!(r, Err(Error::DuplicateLineId(_))));
    }

    #[test]
    fn opening_balance_carries_through() {
        let s = Statement::new("S1", "M1", p(), Currency::USD)
            .unwrap()
            .with_opening(Money::from_minor(5_000, Currency::USD))
            .unwrap();
        assert_eq!(s.primary_aggregate().opening.minor_units, 5_000);
    }

    #[test]
    fn fx_convert_usd_per_eur() {
        // 1 EUR = 1.10 USD. 100 EUR = 110 USD = 11000 minor USD.
        let rate = FxRate {
            foreign: Currency::EUR,
            primary: Currency::USD,
            minor_per_unit_q: 110, // 1.10 with scale = 2
            scale: 2,
            as_of_unix_secs: 0,
        };
        let converted = rate.convert_minor(Money::from_minor(10_000, Currency::EUR));
        // 10000 minor EUR = 100 EUR. 100 EUR * 1.10 = 110 USD = 11000 minor USD.
        assert_eq!(converted, 11_000);
    }

    #[test]
    fn primary_snapshot_sums_fx_adjusted() {
        let mut s = Statement::new("S1", "M1", p(), Currency::USD).unwrap();
        s.push_line(StatementLine::new(
            "l1",
            StatementLineKind::GrossCapture,
            Money::from_minor(10_000, Currency::EUR),
            0,
        ))
        .unwrap();
        s = s.with_fx_rate(FxRate {
            foreign: Currency::EUR,
            primary: Currency::USD,
            minor_per_unit_q: 110,
            scale: 2,
            as_of_unix_secs: 0,
        });
        s.aggregate().unwrap();
        let snap = s.primary_snapshot();
        // 0 USD + 100 EUR * 1.10 = 110 USD = 11_000 minor USD ending.
        assert_eq!(snap.ending.minor_units, 11_000);
    }
}
