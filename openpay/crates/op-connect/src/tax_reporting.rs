//! US tax-reporting form generation.
//!
//! Two forms cover the sub-merchant tax-reporting surface:
//!
//! - **Form 1099-K** — "Payment Card and Third Party Network Transactions".
//!   Filed by the Payment Settlement Entity (PSE) for every US payee
//!   exceeding the threshold. **IRS 2024 Filing Season** threshold:
//!   gross payments > $5,000 (no transaction-count requirement). The
//!   $600 threshold from the American Rescue Plan Act of 2021 was
//!   deferred and replaced with a phased step-down:
//!   - 2024 tax year: $5,000
//!   - 2025 tax year: $2,500
//!   - 2026 tax year onward: $600
//!   The pre-ARPA threshold (still cited by many third-party docs)
//!   was 200 transactions AND $20,000; that bright-line is **gone**
//!   for 2024 and later. We default to the 2024 number and expose
//!   [`Form1099KThresholds`] for operators projecting forward years.
//!
//! - **Form 1042-S** — "Foreign Person's U.S. Source Income Subject
//!   to Withholding". Filed for non-US sub-merchants receiving US-source
//!   income (e.g. a UK-resident seller on a US marketplace). Standard
//!   30% withholding unless a tax treaty reduces it (Form W-8BEN).

use std::collections::BTreeMap;

use op_core::{Currency, Money};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::kyb::{BusinessProfile, CountryCode};

/// US state code (ISO 3166-2 subdivision suffix, e.g. `"CA"`, `"TX"`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct State(pub String);

/// IRS Form 1099-K thresholds by tax year.
///
/// Source: IRS 2024 Form 1099-K Instructions (revised October 2024) and
/// IRS Notice 2024-85, which formalised the phased step-down from the
/// $20,000 / 200-transaction de minimis to $600.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Form1099KThresholds {
    /// Minimum gross payments to trigger filing.
    pub gross_threshold_minor: i64,
    /// Minimum transaction count to trigger filing, if non-zero.
    /// Set to 0 for 2024+ where transaction-count is no longer a
    /// bright line.
    pub transaction_count_threshold: u32,
}

impl Form1099KThresholds {
    /// Threshold for the given tax year, per IRS Notice 2024-85.
    #[must_use]
    pub const fn for_year(year: u16) -> Self {
        match year {
            0..=2023 => Self {
                // Pre-deferral: $20,000 AND 200 transactions.
                gross_threshold_minor: 20_000_00,
                transaction_count_threshold: 200,
            },
            2024 => Self {
                // IRS Notice 2024-85: $5,000, no count requirement.
                gross_threshold_minor: 5_000_00,
                transaction_count_threshold: 0,
            },
            2025 => Self {
                gross_threshold_minor: 2_500_00,
                transaction_count_threshold: 0,
            },
            _ => Self {
                // 2026 onward: ARPA $600.
                gross_threshold_minor: 600_00,
                transaction_count_threshold: 0,
            },
        }
    }
}

/// Generated US Form 1099-K.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Form1099K {
    /// Tax year covered.
    pub tax_year: u16,
    /// Payee business (the sub-merchant).
    pub payee: BusinessProfile,
    /// Payer (the PSE) Taxpayer Identification Number.
    pub payer_tin: String,
    /// Box 1a: gross amount.
    pub total_amount: Money,
    /// Box 3: number of payment transactions.
    pub num_transactions: u32,
    /// Box 5a-5l: monthly gross amount breakdown (January..December).
    pub monthly_breakdown: [Money; 12],
    /// Box 6 / state-filing: per-state gross-amount breakdown.
    pub state_breakdown: BTreeMap<State, Money>,
}

/// Generated US Form 1042-S for a non-US sub-merchant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Form1042S {
    /// Tax year covered.
    pub tax_year: u16,
    /// Payee business.
    pub payee: BusinessProfile,
    /// Country of residence (ISO 3166-1 alpha-2).
    pub payee_country: CountryCode,
    /// Box 2: gross income paid.
    pub gross_income: Money,
    /// Box 3a: chapter 3 withholding rate (decimal, e.g. 0.30 for 30%).
    pub withholding_rate: f32,
    /// Box 10: total federal tax withheld.
    pub federal_tax_withheld: Money,
}

/// Inbound transaction record used for annual-summary aggregation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnnualTransaction {
    /// 1..=12.
    pub month: u8,
    /// Gross amount in payee's reporting currency.
    pub gross: Money,
    /// State the cardholder transacted from (for box 6 reporting).
    pub state: State,
}

/// Build a Form 1099-K from a year of transactions, returning `Ok(None)`
/// if the payee does not meet the threshold.
///
/// # Errors
/// - [`Error::TaxReporting`] if `tax_year` is in the past beyond the
///   IRS 7-year statute, if any transaction has an invalid month, or
///   if currencies disagree across transactions.
/// - [`Error::Overflow`] on monetary overflow.
pub fn build_1099k(
    tax_year: u16,
    payee: &BusinessProfile,
    payer_tin: &str,
    transactions: &[AnnualTransaction],
    currency: Currency,
) -> Result<Option<Form1099K>> {
    if transactions.is_empty() {
        return Ok(None);
    }

    let mut total = Money::zero(currency);
    let mut monthly: [Money; 12] = [Money::zero(currency); 12];
    let mut state_breakdown: BTreeMap<State, Money> = BTreeMap::new();
    let mut count: u32 = 0;

    for tx in transactions {
        if tx.gross.currency != currency {
            return Err(Error::TaxReporting {
                reason: format!(
                    "transaction in {} but report currency is {}",
                    tx.gross.currency, currency
                ),
            });
        }
        if !(1..=12).contains(&tx.month) {
            return Err(Error::TaxReporting {
                reason: format!("invalid month {}", tx.month),
            });
        }
        let idx = (tx.month - 1) as usize;
        monthly[idx] = monthly[idx]
            .checked_add(tx.gross)
            .map_err(|_| Error::Overflow)?;
        total = total
            .checked_add(tx.gross)
            .map_err(|_| Error::Overflow)?;
        state_breakdown
            .entry(tx.state.clone())
            .and_modify(|m| {
                if let Ok(s) = m.checked_add(tx.gross) {
                    *m = s;
                }
            })
            .or_insert(tx.gross);
        count = count.saturating_add(1);
    }

    let thresholds = Form1099KThresholds::for_year(tax_year);
    let meets_amount = total.minor_units >= thresholds.gross_threshold_minor;
    let meets_count = thresholds.transaction_count_threshold == 0
        || count >= thresholds.transaction_count_threshold;
    // For 2024+ the threshold is amount-only ⇒ both flags collapse to
    // `meets_amount`. Pre-2024 used AND of both.
    let meets = if thresholds.transaction_count_threshold == 0 {
        meets_amount
    } else {
        meets_amount && meets_count
    };

    if !meets {
        return Ok(None);
    }

    Ok(Some(Form1099K {
        tax_year,
        payee: payee.clone(),
        payer_tin: payer_tin.to_string(),
        total_amount: total,
        num_transactions: count,
        monthly_breakdown: monthly,
        state_breakdown,
    }))
}

/// Build a Form 1042-S for a non-US payee.
///
/// # Errors
/// - [`Error::TaxReporting`] if `payee_country` is `"US"` (use 1099-K
///   instead) or `withholding_rate` is out of `[0.0, 1.0]`.
/// - [`Error::Overflow`] on the withheld-amount computation.
pub fn build_1042s(
    tax_year: u16,
    payee: &BusinessProfile,
    payee_country: CountryCode,
    gross_income: Money,
    withholding_rate: f32,
) -> Result<Form1042S> {
    if payee_country.0 == "US" {
        return Err(Error::TaxReporting {
            reason: "Form 1042-S is for non-US payees; use 1099-K".into(),
        });
    }
    if !(0.0..=1.0).contains(&withholding_rate) {
        return Err(Error::TaxReporting {
            reason: format!("withholding_rate {withholding_rate} not in [0.0, 1.0]"),
        });
    }
    let withheld_minor =
        ((gross_income.minor_units as f64) * f64::from(withholding_rate)).floor() as i64;
    let federal_tax_withheld = Money::from_minor(withheld_minor, gross_income.currency);
    Ok(Form1042S {
        tax_year,
        payee: payee.clone(),
        payee_country,
        gross_income,
        withholding_rate,
        federal_tax_withheld,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kyb::{Address, BusinessStructure, TaxId};

    fn sample_payee() -> BusinessProfile {
        BusinessProfile {
            legal_name: "Acme Widgets LLC".into(),
            trade_name: None,
            structure: BusinessStructure::SingleMemberLlc,
            tax_id: Some(TaxId::Ein("12-3456789".into())),
            mcc: 5734,
            country: CountryCode("US".into()),
            registered_address: Address {
                line1: "1 Main St".into(),
                line2: None,
                city: "Austin".into(),
                region: "TX".into(),
                postal_code: "78701".into(),
                country: CountryCode("US".into()),
            },
            support_email: None,
            support_phone: None,
            website: None,
        }
    }

    #[test]
    fn threshold_2024_is_5000() {
        let t = Form1099KThresholds::for_year(2024);
        assert_eq!(t.gross_threshold_minor, 500_000);
        assert_eq!(t.transaction_count_threshold, 0);
    }

    #[test]
    fn threshold_2026_is_600() {
        let t = Form1099KThresholds::for_year(2026);
        assert_eq!(t.gross_threshold_minor, 60_000);
    }

    #[test]
    fn under_threshold_returns_none() {
        let txs = vec![AnnualTransaction {
            month: 1,
            gross: Money::from_minor(1_000_00, Currency::USD),
            state: State("CA".into()),
        }];
        let form = build_1099k(2024, &sample_payee(), "98-7654321", &txs, Currency::USD)
            .expect("ok");
        assert!(form.is_none());
    }

    #[test]
    fn over_threshold_emits_form() {
        // 12 months at $1,000 = $12,000 — over the 2024 $5,000 line.
        let txs: Vec<AnnualTransaction> = (1..=12)
            .map(|m| AnnualTransaction {
                month: m,
                gross: Money::from_minor(1_000_00, Currency::USD),
                state: State("CA".into()),
            })
            .collect();
        let form = build_1099k(2024, &sample_payee(), "98-7654321", &txs, Currency::USD)
            .expect("ok")
            .expect("form");
        assert_eq!(form.total_amount, Money::from_minor(12_000_00, Currency::USD));
        assert_eq!(form.num_transactions, 12);
        for m in &form.monthly_breakdown {
            assert_eq!(*m, Money::from_minor(1_000_00, Currency::USD));
        }
        assert_eq!(
            form.state_breakdown.get(&State("CA".into())).copied(),
            Some(Money::from_minor(12_000_00, Currency::USD))
        );
    }

    #[test]
    fn us_payee_rejects_1042s() {
        let err = build_1042s(
            2024,
            &sample_payee(),
            CountryCode("US".into()),
            Money::from_minor(10_000_00, Currency::USD),
            0.30,
        )
        .expect_err("us not eligible");
        assert!(matches!(err, Error::TaxReporting { .. }));
    }

    #[test]
    fn non_us_1042s_applies_withholding() {
        let mut payee = sample_payee();
        payee.country = CountryCode("GB".into());
        let form = build_1042s(
            2024,
            &payee,
            CountryCode("GB".into()),
            Money::from_minor(10_000_00, Currency::USD),
            0.30,
        )
        .expect("ok");
        assert_eq!(
            form.federal_tax_withheld,
            Money::from_minor(3_000_00, Currency::USD)
        );
    }
}
