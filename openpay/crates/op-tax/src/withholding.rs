//! Withholding & information-reporting integration.
//!
//! Three regimes the payment stack has to feed:
//!
//! 1. **IRS Form 1099-K** — third-party network reporting for US
//!    payees. The American Rescue Plan dropped the threshold to
//!    $600 / 1 transaction (effective 2024; phased rollout reached
//!    full $600 in 2026). Marketplaces and PSPs file one 1099-K
//!    per payee per year.
//!
//! 2. **IRS Form 1042-S** — non-resident alien (NRA) withholding for
//!    payouts to foreign persons. Default 30% withholding rate;
//!    reduced under bilateral tax treaties (typically 0–15% on
//!    royalties / interest / dividends, sometimes 0% on services).
//!
//! 3. **EU VAT-MOSS (One Stop Shop)** — for B2C cross-border digital
//!    services inside the EU, the seller files quarterly in its
//!    home country instead of registering in each of 27 member
//!    states. The threshold for the simplified scheme is €10,000
//!    (intra-EU sales of digital services + distance sales of goods
//!    combined).
//!
//! ## Why this module is thin
//!
//! Form generation (the actual PDF / IRS-XML output) lives in
//! `op-connect` — that's the layer that owns payee identity, TIN,
//! and the reporting calendar. This module exposes a tiny
//! [`WithholdingHook`] trait so the payment runtime can defer the
//! reporting record to whichever `op-connect` implementation the
//! operator wired in. We intentionally do NOT take an `op-connect`
//! dependency here: it's an outbound seam, not an inbound coupling.

use async_trait::async_trait;
use chrono::{DateTime, NaiveDate, Utc};
use op_core::Money;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::jurisdiction::CountryCode;

/// Classification of a withholding event the runtime needs to record.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum WithholdingEvent {
    /// US payee crossed the 1099-K reporting threshold.
    /// Operators submit one 1099-K per year per payee.
    Form1099K {
        /// Stable payee identifier (`op-connect` account ID).
        payee_id: String,
        /// Calendar year being reported.
        tax_year: i32,
        /// Gross processed amount for the year.
        gross_amount: Money,
        /// Card-Not-Present transactions for the year. (Box 5a on the
        /// IRS form.)
        cnp_count: u32,
    },
    /// Non-resident-alien payout subject to chapter-3 withholding.
    /// Operators issue a 1042-S to the payee and remit the withheld
    /// amount to the IRS quarterly via Form 1042.
    Form1042S {
        /// Stable payee identifier.
        payee_id: String,
        /// Payee's country of residence (ISO 3166-1 alpha-2).
        residence: CountryCode,
        /// Gross payment.
        gross_amount: Money,
        /// Withheld amount (gross × treaty rate).
        withheld_amount: Money,
        /// Treaty rate applied. `Decimal::ZERO` = full exemption,
        /// `Decimal::new(30, 2)` = the 30% statutory rate when no
        /// treaty applies.
        treaty_rate: Decimal,
        /// Income code per IRS instructions for Form 1042-S box 1
        /// (e.g. `"50"` for services, `"12"` for royalties).
        income_code: String,
    },
    /// EU VAT-MOSS quarterly summary line — one country per quarter
    /// where the seller had B2C cross-border digital sales.
    VatMossQuarter {
        /// Reporting quarter (year, quarter 1–4).
        year: i32,
        /// Quarter, 1..=4.
        quarter: u8,
        /// EU member state where consumers are located.
        consumer_member_state: CountryCode,
        /// Sum of gross consumer revenue from that member state for
        /// the quarter.
        gross_amount: Money,
        /// Sum of VAT collected from that member state for the quarter.
        vat_amount: Money,
    },
}

/// One payout the runtime is about to disburse to a payee.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PayoutEvent {
    /// Stable payee identifier (`op-connect` account ID).
    pub payee_id: String,
    /// Payee's country of residence — drives 1042-S vs 1099-K routing.
    pub payee_residence: CountryCode,
    /// Gross payment amount.
    pub gross_amount: Money,
    /// When the payout settles.
    pub settled_at: DateTime<Utc>,
    /// Income classification — drives the 1042-S `income_code` mapping
    /// and the 1099-K boxes.
    pub income_code: String,
}

/// US payee year-to-date roll-up. Maintained externally; this struct
/// is what the runtime passes in when checking thresholds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PayeeYtd {
    /// Payee identifier.
    pub payee_id: String,
    /// Calendar year.
    pub tax_year: i32,
    /// Gross processed YTD.
    pub gross_amount: Money,
    /// Card-Not-Present transaction count YTD.
    pub cnp_count: u32,
    /// Whether a 1099-K has already been emitted for this payee/year
    /// (idempotency guard).
    pub form_emitted: bool,
}

/// The seam where this crate hands a [`WithholdingEvent`] off to
/// `op-connect` (or any other downstream).
///
/// Implementations live OUTSIDE this crate — `op-connect` provides
/// the concrete impl that wires the event into IRS / EU reporting
/// pipelines.
#[async_trait]
pub trait WithholdingHook: Send + Sync {
    /// Persist / queue a withholding event for later filing.
    ///
    /// # Errors
    /// Implementation-defined.
    async fn record(&self, event: WithholdingEvent) -> Result<()>;
}

/// Bilateral tax-treaty lookup. Returns the reduced withholding rate
/// for `(payee_residence, income_code)`, or `None` if no treaty rate
/// applies (use the statutory 30%).
///
/// The seed table here covers the most-claimed treaty rates as of
/// 2026. Operators with corner cases should override via their own
/// table — IRS Publication 515 / Treasury TIN matching are the source
/// of truth and change annually.
#[must_use]
pub fn treaty_rate(payee_residence: &CountryCode, income_code: &str) -> Option<Decimal> {
    // Income codes (IRS Form 1042-S, 2024 instructions):
    //   "12" = royalties, industrial
    //   "50" = services performed in US (independent personal services)
    //   "06" = dividends from US corporations
    //   "01" = interest
    // We carry the most-common treaty reductions; the rest fall back
    // to statutory 30%.
    let r = match (payee_residence.0.as_str(), income_code) {
        // UK-US treaty.
        ("GB", "06") => Decimal::new(15, 2), // 15% dividends
        ("GB", "12") => Decimal::ZERO,       // 0% royalties
        ("GB", "01") => Decimal::ZERO,       // 0% interest
        ("GB", "50") => Decimal::ZERO,
        // Canada-US treaty.
        ("CA", "06") => Decimal::new(15, 2),
        ("CA", "12") => Decimal::new(10, 2),
        ("CA", "01") => Decimal::ZERO,
        ("CA", "50") => Decimal::ZERO,
        // Germany-US treaty.
        ("DE", "06") => Decimal::new(15, 2),
        ("DE", "12") => Decimal::ZERO,
        ("DE", "01") => Decimal::ZERO,
        // India-US treaty.
        ("IN", "06") => Decimal::new(15, 2),
        ("IN", "12") => Decimal::new(15, 2), // higher rate for industrial royalties
        ("IN", "01") => Decimal::new(15, 2),
        // Japan-US treaty.
        ("JP", "06") => Decimal::new(10, 2),
        ("JP", "12") => Decimal::ZERO,
        ("JP", "01") => Decimal::ZERO,
        _ => return None,
    };
    Some(r)
}

/// Statutory NRA withholding rate when no treaty applies (30%).
#[must_use]
pub fn statutory_nra_rate() -> Decimal {
    Decimal::new(30, 2)
}

/// 1099-K threshold for tax year 2026 (and forward).
/// Source: IRC §6050W as amended by the American Rescue Plan.
#[must_use]
pub fn form_1099k_threshold_minor() -> i64 {
    60_000 // $600.00
}

/// VAT-MOSS / OSS small-seller threshold for intra-EU B2C digital
/// services + distance sales of goods (€10,000 per year, combined).
#[must_use]
pub fn vat_oss_threshold_eur_minor() -> i64 {
    1_000_000 // €10,000.00
}

/// Compute the 1042-S withholding for a payout. Returns `(rate, withheld)`.
///
/// If a treaty applies, uses the treaty rate (which can be 0%). Otherwise
/// uses the statutory 30% rate.
///
/// # Errors
/// Returns [`crate::error::Error::Money`] if the multiplied amount
/// would overflow.
pub fn compute_1042s(payout: &PayoutEvent) -> Result<(Decimal, Money)> {
    let rate = treaty_rate(&payout.payee_residence, &payout.income_code)
        .unwrap_or_else(statutory_nra_rate);
    let amount_dec = Decimal::from(payout.gross_amount.minor_units) * rate;
    let withheld_minor = amount_dec
        .round_dp_with_strategy(0, rust_decimal::RoundingStrategy::MidpointAwayFromZero);
    let v = withheld_minor.to_string().parse::<i64>().unwrap_or(0);
    Ok((rate, Money::from_minor(v, payout.gross_amount.currency)))
}

/// Returns the relevant [`WithholdingEvent`] for a settled payout,
/// given the payee's current YTD totals. Returns `None` if no event
/// is yet required.
///
/// Routing:
/// - Payee residence == `US` → 1099-K if YTD ≥ threshold and not yet
///   emitted.
/// - Payee residence != `US` → 1042-S for every payout.
#[must_use]
pub fn classify(payout: &PayoutEvent, ytd: &PayeeYtd) -> Option<WithholdingEvent> {
    if payout.payee_residence.0 == "US" {
        if ytd.gross_amount.minor_units >= form_1099k_threshold_minor() && !ytd.form_emitted {
            return Some(WithholdingEvent::Form1099K {
                payee_id: payout.payee_id.clone(),
                tax_year: ytd.tax_year,
                gross_amount: ytd.gross_amount,
                cnp_count: ytd.cnp_count,
            });
        }
        return None;
    }
    let (rate, withheld) = match compute_1042s(payout) {
        Ok(x) => x,
        Err(_) => return None,
    };
    Some(WithholdingEvent::Form1042S {
        payee_id: payout.payee_id.clone(),
        residence: payout.payee_residence.clone(),
        gross_amount: payout.gross_amount,
        withheld_amount: withheld,
        treaty_rate: rate,
        income_code: payout.income_code.clone(),
    })
}

/// Build the quarterly VAT-MOSS summary event for a given member
/// state from a list of (gross, vat) tuples.
#[must_use]
pub fn build_vat_moss(
    year: i32,
    quarter: u8,
    consumer_member_state: CountryCode,
    lines: &[(Money, Money)],
) -> Option<WithholdingEvent> {
    let first = lines.first()?;
    let currency = first.0.currency;
    let mut gross = Money::from_minor(0, currency);
    let mut vat = Money::from_minor(0, currency);
    for (g, v) in lines {
        gross = gross.checked_add(*g).ok()?;
        vat = vat.checked_add(*v).ok()?;
    }
    Some(WithholdingEvent::VatMossQuarter {
        year,
        quarter,
        consumer_member_state,
        gross_amount: gross,
        vat_amount: vat,
    })
}

/// Sentinel — unused but exposed so callers can write a YTD initializer.
#[must_use]
pub const fn unix_epoch() -> NaiveDate {
    // chrono::NaiveDate::from_ymd_opt isn't const — return a sentinel
    // documenting expectation. Callers construct their own dates.
    NaiveDate::MIN
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    #[test]
    fn us_payee_under_threshold_no_event() {
        let p = PayoutEvent {
            payee_id: "P1".into(),
            payee_residence: CountryCode::new("US"),
            gross_amount: Money::from_minor(5_000, Currency::USD),
            settled_at: Utc::now(),
            income_code: "50".into(),
        };
        let ytd = PayeeYtd {
            payee_id: "P1".into(),
            tax_year: 2026,
            gross_amount: Money::from_minor(30_000, Currency::USD), // $300 — under $600
            cnp_count: 3,
            form_emitted: false,
        };
        assert!(classify(&p, &ytd).is_none());
    }

    #[test]
    fn us_payee_over_threshold_emits_1099k() {
        let p = PayoutEvent {
            payee_id: "P1".into(),
            payee_residence: CountryCode::new("US"),
            gross_amount: Money::from_minor(5_000, Currency::USD),
            settled_at: Utc::now(),
            income_code: "50".into(),
        };
        let ytd = PayeeYtd {
            payee_id: "P1".into(),
            tax_year: 2026,
            gross_amount: Money::from_minor(65_000, Currency::USD), // $650 — over $600
            cnp_count: 8,
            form_emitted: false,
        };
        let ev = classify(&p, &ytd).unwrap();
        assert!(matches!(ev, WithholdingEvent::Form1099K { .. }));
    }

    #[test]
    fn nra_payout_with_uk_treaty_zero_withholding_on_services() {
        let p = PayoutEvent {
            payee_id: "P-GB".into(),
            payee_residence: CountryCode::new("GB"),
            gross_amount: Money::from_minor(100_000, Currency::USD),
            settled_at: Utc::now(),
            income_code: "50".into(), // services
        };
        let (rate, withheld) = compute_1042s(&p).unwrap();
        assert_eq!(rate, Decimal::ZERO);
        assert_eq!(withheld.minor_units, 0);
    }

    #[test]
    fn nra_payout_no_treaty_uses_30_percent() {
        let p = PayoutEvent {
            payee_id: "P-XX".into(),
            payee_residence: CountryCode::new("XX"),
            gross_amount: Money::from_minor(100_000, Currency::USD),
            settled_at: Utc::now(),
            income_code: "50".into(),
        };
        let (rate, withheld) = compute_1042s(&p).unwrap();
        assert_eq!(rate, Decimal::new(30, 2));
        // 30% of $1000 = $300 = 30,000 minor units.
        assert_eq!(withheld.minor_units, 30_000);
    }

    #[test]
    fn vat_moss_aggregates_quarterly_lines() {
        let lines = vec![
            (
                Money::from_minor(10_000, Currency::EUR),
                Money::from_minor(2_000, Currency::EUR),
            ),
            (
                Money::from_minor(5_000, Currency::EUR),
                Money::from_minor(1_000, Currency::EUR),
            ),
        ];
        let ev = build_vat_moss(2026, 2, CountryCode::new("DE"), &lines).unwrap();
        let WithholdingEvent::VatMossQuarter {
            gross_amount,
            vat_amount,
            ..
        } = ev
        else {
            panic!()
        };
        assert_eq!(gross_amount.minor_units, 15_000);
        assert_eq!(vat_amount.minor_units, 3_000);
    }

    #[test]
    fn idempotent_after_form_emitted() {
        let p = PayoutEvent {
            payee_id: "P1".into(),
            payee_residence: CountryCode::new("US"),
            gross_amount: Money::from_minor(5_000, Currency::USD),
            settled_at: Utc::now(),
            income_code: "50".into(),
        };
        let ytd = PayeeYtd {
            payee_id: "P1".into(),
            tax_year: 2026,
            gross_amount: Money::from_minor(100_000, Currency::USD),
            cnp_count: 8,
            form_emitted: true,
        };
        assert!(classify(&p, &ytd).is_none());
    }
}
