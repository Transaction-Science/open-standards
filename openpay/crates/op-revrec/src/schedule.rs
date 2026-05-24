//! Step 4 (allocate transaction price) and Step 5 (recognize revenue
//! as obligations are satisfied) of the ASC 606 / IFRS 15 model.
//!
//! - [`allocate_transaction_price`] performs the relative-SSP split
//!   per ASC 606-10-32-31 — allocated_i = total_price * ssp_i / Σ ssp.
//! - [`generate`] produces the [`RecognitionSchedule`] for one
//!   performance obligation, given its allocation: a series of
//!   [`ScheduleEntry`]s, each booking some portion of the obligation
//!   on a specific date.
//! - [`usage_recognition`] is the runtime helper for usage-based
//!   obligations — it does not produce a schedule up front; instead the
//!   caller emits one entry per usage event.

use chrono::{Datelike, NaiveDate};
use op_core::{Currency, Money};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::contract::{
    Contract, Milestone, ObligationId, PercentCompleteSnapshot, PerformanceObligation,
    RecognitionPattern,
};
use crate::error::{Error, Result};

/// One row of a recognition schedule. Each row books a portion of one
/// performance obligation on a specific date.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleEntry {
    /// Obligation this entry recognizes against.
    pub obligation_id: ObligationId,
    /// Date the entry books.
    pub date: NaiveDate,
    /// Amount booked on this date in minor units.
    pub amount_minor: i64,
    /// Currency of the amount.
    pub currency: Currency,
}

/// The schedule for one performance obligation. The sum of entries'
/// `amount_minor` equals the allocated transaction price for that
/// obligation (modulo a one-minor-unit reconciliation row that
/// [`generate`] adds to the last entry to absorb integer rounding).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecognitionSchedule {
    /// Obligation id this schedule belongs to.
    pub obligation_id: ObligationId,
    /// Ordered list of entries.
    pub entries: Vec<ScheduleEntry>,
}

impl RecognitionSchedule {
    /// Total scheduled amount, in minor units.
    #[must_use]
    pub fn total_minor(&self) -> i64 {
        self.entries.iter().map(|e| e.amount_minor).sum()
    }
}

/// Step 4: allocate the contract's transaction price across its
/// performance obligations in proportion to standalone selling price.
///
/// Returns a map from `ObligationId` to the allocated amount in minor
/// units of the contract currency. Allocated amounts sum to the
/// transaction-price total; the last obligation absorbs the
/// integer-division remainder so the sum is exact.
///
/// # Errors
/// - [`Error::ZeroStandaloneSellingPrice`] when Σ SSP is zero.
/// - [`Error::CurrencyMismatch`] when an obligation's SSP currency
///   differs from the contract transaction-price currency.
/// - [`Error::Money`] on i64 overflow.
pub fn allocate_transaction_price(contract: &Contract) -> Result<Vec<(ObligationId, i64)>> {
    contract.validate_currencies()?;
    let total_ssp = contract.total_ssp_minor()?;
    if total_ssp == 0 {
        return Err(Error::ZeroStandaloneSellingPrice);
    }
    let total_tp = contract.transaction_price.total()?;
    let total_tp_minor = total_tp.minor_units;

    let mut out: Vec<(ObligationId, i64)> = Vec::with_capacity(contract.obligations.len());
    let mut assigned: i64 = 0;
    let last_idx = contract.obligations.len().saturating_sub(1);

    for (i, o) in contract.obligations.iter().enumerate() {
        if i == last_idx {
            // Plug to make the allocation sum exactly to the transaction
            // price. Avoids the classic integer-division leak.
            let remainder = total_tp_minor
                .checked_sub(assigned)
                .ok_or_else(|| Error::Money(op_core::Error::Overflow))?;
            out.push((o.id.clone(), remainder));
        } else {
            // amount = total_tp * ssp / total_ssp, computed as i128 to
            // avoid overflow on intermediate product.
            let ssp = i128::from(o.standalone_selling_price.minor_units);
            let tp = i128::from(total_tp_minor);
            let denom = i128::from(total_ssp);
            let alloc_i128 = tp.saturating_mul(ssp) / denom;
            let alloc: i64 = i64::try_from(alloc_i128)
                .map_err(|_| Error::Money(op_core::Error::Overflow))?;
            assigned = assigned
                .checked_add(alloc)
                .ok_or_else(|| Error::Money(op_core::Error::Overflow))?;
            out.push((o.id.clone(), alloc));
        }
    }
    Ok(out)
}

/// Step 5: build the recognition schedule for one performance
/// obligation given its allocated amount.
///
/// For each pattern:
///
/// - `PointInTime`: a single entry on `date`.
/// - `StraightLine`: monthly entries at month-end through the period,
///   each prorated by days-in-month / total-days. Last entry takes any
///   rounding plug.
/// - `OutputMilestones`: one entry per milestone, sized by its fraction.
/// - `InputPercentComplete`: one entry per snapshot, sized by the delta
///   between consecutive percent-complete values.
/// - `Usage`: no schedule up front; use [`usage_recognition`] at runtime.
///   Returns an empty schedule.
///
/// # Errors
/// - [`Error::InvalidPeriod`] if `end < start` on a straight-line obligation.
/// - [`Error::Invariant`] if milestones do not sum to 1.0 or snapshots
///   are not monotonically non-decreasing.
pub fn generate(
    obligation: &PerformanceObligation,
    allocated_minor: i64,
    currency: Currency,
) -> Result<RecognitionSchedule> {
    let entries = match &obligation.pattern {
        RecognitionPattern::PointInTime { date } => {
            vec![ScheduleEntry {
                obligation_id: obligation.id.clone(),
                date: *date,
                amount_minor: allocated_minor,
                currency,
            }]
        }
        RecognitionPattern::StraightLine { start, end } => {
            straight_line(&obligation.id, *start, *end, allocated_minor, currency)?
        }
        RecognitionPattern::OutputMilestones { milestones } => {
            output_milestones(&obligation.id, milestones, allocated_minor, currency)?
        }
        RecognitionPattern::InputPercentComplete { snapshots } => {
            input_percent_complete(&obligation.id, snapshots, allocated_minor, currency)?
        }
        RecognitionPattern::Usage { .. } => Vec::new(),
    };
    Ok(RecognitionSchedule {
        obligation_id: obligation.id.clone(),
        entries,
    })
}

/// Straight-line monthly recognition over `[start, end]` inclusive.
///
/// Days-in-month / total-days proration. The last entry plugs the
/// rounding remainder so the schedule sums to `allocated_minor`.
fn straight_line(
    id: &ObligationId,
    start: NaiveDate,
    end: NaiveDate,
    allocated_minor: i64,
    currency: Currency,
) -> Result<Vec<ScheduleEntry>> {
    if end < start {
        return Err(Error::InvalidPeriod {
            start: start.to_string(),
            end: end.to_string(),
        });
    }
    let total_days = (end - start).num_days() + 1; // inclusive
    if total_days <= 0 {
        return Err(Error::InvalidPeriod {
            start: start.to_string(),
            end: end.to_string(),
        });
    }

    // Compute month boundaries. We book at the last day of each month
    // that falls within [start, end], plus the start month's last day
    // and the period end if it doesn't align with a month-end.
    let mut entries: Vec<ScheduleEntry> = Vec::new();
    let mut cursor = start;
    let mut assigned: i64 = 0;

    while cursor <= end {
        let month_end = last_day_of_month(cursor);
        let entry_date = if month_end < end { month_end } else { end };

        let days_this_period = (entry_date - cursor).num_days() + 1;
        // amount = allocated * days_this_period / total_days, i128 to
        // avoid overflow.
        let alloc_i128 = i128::from(allocated_minor)
            .saturating_mul(i128::from(days_this_period))
            / i128::from(total_days);
        let alloc: i64 = i64::try_from(alloc_i128)
            .map_err(|_| Error::Money(op_core::Error::Overflow))?;
        entries.push(ScheduleEntry {
            obligation_id: id.clone(),
            date: entry_date,
            amount_minor: alloc,
            currency,
        });
        assigned = assigned
            .checked_add(alloc)
            .ok_or_else(|| Error::Money(op_core::Error::Overflow))?;

        // Advance to first day of next month after entry_date.
        cursor = match entry_date.succ_opt() {
            Some(d) => d,
            None => break,
        };
        if cursor > end {
            break;
        }
    }

    // Plug rounding into the last entry.
    if let Some(last) = entries.last_mut() {
        let remainder = allocated_minor
            .checked_sub(assigned)
            .ok_or_else(|| Error::Money(op_core::Error::Overflow))?;
        last.amount_minor = last
            .amount_minor
            .checked_add(remainder)
            .ok_or_else(|| Error::Money(op_core::Error::Overflow))?;
    }
    Ok(entries)
}

/// Last calendar day of the month containing `d`.
fn last_day_of_month(d: NaiveDate) -> NaiveDate {
    let (y, m) = (d.year(), d.month());
    let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
    let first_of_next = NaiveDate::from_ymd_opt(ny, nm, 1).unwrap_or(d);
    first_of_next.pred_opt().unwrap_or(d)
}

fn output_milestones(
    id: &ObligationId,
    milestones: &[Milestone],
    allocated_minor: i64,
    currency: Currency,
) -> Result<Vec<ScheduleEntry>> {
    // Validate fractions sum to ~1.0.
    let sum: Decimal = milestones.iter().map(|m| m.fraction).sum();
    let one = Decimal::ONE;
    let eps = Decimal::new(1, 4); // 0.0001 tolerance
    if (sum - one).abs() > eps {
        return Err(Error::Invariant(format!(
            "milestone fractions sum to {sum}, expected 1.0"
        )));
    }
    let mut entries = Vec::with_capacity(milestones.len());
    let mut assigned: i64 = 0;
    for (i, m) in milestones.iter().enumerate() {
        let amount: i64 = if i + 1 == milestones.len() {
            // last absorbs the plug
            allocated_minor
                .checked_sub(assigned)
                .ok_or_else(|| Error::Money(op_core::Error::Overflow))?
        } else {
            let f_minor = Decimal::from(allocated_minor) * m.fraction;
            let f_int = f_minor.round_dp(0).to_i64().unwrap_or(0);
            assigned = assigned
                .checked_add(f_int)
                .ok_or_else(|| Error::Money(op_core::Error::Overflow))?;
            f_int
        };
        entries.push(ScheduleEntry {
            obligation_id: id.clone(),
            date: m.date,
            amount_minor: amount,
            currency,
        });
    }
    Ok(entries)
}

fn input_percent_complete(
    id: &ObligationId,
    snapshots: &[PercentCompleteSnapshot],
    allocated_minor: i64,
    currency: Currency,
) -> Result<Vec<ScheduleEntry>> {
    // Validate monotonic non-decreasing.
    let mut prev = Decimal::ZERO;
    for s in snapshots {
        if s.percent < prev {
            return Err(Error::Invariant(format!(
                "percent-complete decreased from {prev} to {} on {}",
                s.percent, s.date
            )));
        }
        prev = s.percent;
    }
    let mut entries = Vec::with_capacity(snapshots.len());
    let mut last_percent = Decimal::ZERO;
    let mut assigned: i64 = 0;
    for (i, s) in snapshots.iter().enumerate() {
        let delta = s.percent - last_percent;
        last_percent = s.percent;
        let amount: i64 = if i + 1 == snapshots.len() {
            allocated_minor
                .checked_sub(assigned)
                .ok_or_else(|| Error::Money(op_core::Error::Overflow))?
        } else {
            let d = Decimal::from(allocated_minor) * delta;
            let int = d.round_dp(0).to_i64().unwrap_or(0);
            assigned = assigned
                .checked_add(int)
                .ok_or_else(|| Error::Money(op_core::Error::Overflow))?;
            int
        };
        entries.push(ScheduleEntry {
            obligation_id: id.clone(),
            date: s.date,
            amount_minor: amount,
            currency,
        });
    }
    Ok(entries)
}

/// Build one schedule entry for a usage event on a usage-based
/// obligation. Returns `None` if the obligation pattern is not
/// `Usage`. Caller is expected to post the entry through the ledger.
#[must_use]
pub fn usage_recognition(
    obligation: &PerformanceObligation,
    units: u64,
    date: NaiveDate,
    currency: Currency,
) -> Option<ScheduleEntry> {
    if let RecognitionPattern::Usage { unit_price_minor } = &obligation.pattern {
        let units_i: i64 = i64::try_from(units).unwrap_or(i64::MAX);
        let amount = unit_price_minor.checked_mul(units_i).unwrap_or(i64::MAX);
        Some(ScheduleEntry {
            obligation_id: obligation.id.clone(),
            date,
            amount_minor: amount,
            currency,
        })
    } else {
        None
    }
}

/// Convenience: convert a [`ScheduleEntry`] into an [`op_core::Money`].
#[must_use]
pub fn entry_money(e: &ScheduleEntry) -> Money {
    Money::from_minor(e.amount_minor, e.currency)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{Presentation, TransactionPrice};
    use op_core::Currency;

    fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap_or_default()
    }

    fn usd(n: i64) -> Money {
        Money::from_minor(n, Currency::USD)
    }

    #[test]
    fn straight_line_annual_sums_to_total() {
        let o = PerformanceObligation {
            id: ObligationId::new("saas"),
            standalone_selling_price: usd(12_000_00),
            pattern: RecognitionPattern::StraightLine {
                start: ymd(2026, 1, 1),
                end: ymd(2026, 12, 31),
            },
            presentation: Presentation::Gross,
        };
        let sched = generate(&o, 12_000_00, Currency::USD).expect("schedule");
        assert_eq!(sched.entries.len(), 12);
        assert_eq!(sched.total_minor(), 12_000_00);
    }

    #[test]
    fn allocation_sums_to_transaction_price() {
        let contract = Contract {
            id: crate::contract::ContractId::new(),
            customer_ref: "c".into(),
            effective_date: ymd(2026, 1, 1),
            obligations: vec![
                PerformanceObligation {
                    id: ObligationId::new("a"),
                    standalone_selling_price: usd(1_000),
                    pattern: RecognitionPattern::PointInTime { date: ymd(2026, 1, 1) },
                    presentation: Presentation::Gross,
                },
                PerformanceObligation {
                    id: ObligationId::new("b"),
                    standalone_selling_price: usd(3_000),
                    pattern: RecognitionPattern::PointInTime { date: ymd(2026, 1, 1) },
                    presentation: Presentation::Gross,
                },
            ],
            transaction_price: TransactionPrice::fixed(usd(3_999)), // odd so we test the plug
        };
        let alloc = allocate_transaction_price(&contract).expect("alloc");
        let sum: i64 = alloc.iter().map(|(_, n)| n).sum();
        assert_eq!(sum, 3_999);
    }

    #[test]
    fn milestones_sum_to_allocated() {
        let o = PerformanceObligation {
            id: ObligationId::new("impl"),
            standalone_selling_price: usd(100_000),
            pattern: RecognitionPattern::OutputMilestones {
                milestones: vec![
                    Milestone {
                        label: "kickoff".into(),
                        date: ymd(2026, 2, 1),
                        fraction: "0.25".parse().unwrap_or_default(),
                    },
                    Milestone {
                        label: "uat".into(),
                        date: ymd(2026, 5, 1),
                        fraction: "0.50".parse().unwrap_or_default(),
                    },
                    Milestone {
                        label: "golive".into(),
                        date: ymd(2026, 8, 1),
                        fraction: "0.25".parse().unwrap_or_default(),
                    },
                ],
            },
            presentation: Presentation::Gross,
        };
        let s = generate(&o, 100_000, Currency::USD).expect("ok");
        assert_eq!(s.total_minor(), 100_000);
        assert_eq!(s.entries.len(), 3);
    }

    #[test]
    fn invalid_period_rejected() {
        let o = PerformanceObligation {
            id: ObligationId::new("bad"),
            standalone_selling_price: usd(100),
            pattern: RecognitionPattern::StraightLine {
                start: ymd(2026, 12, 31),
                end: ymd(2026, 1, 1),
            },
            presentation: Presentation::Gross,
        };
        assert!(matches!(
            generate(&o, 100, Currency::USD),
            Err(Error::InvalidPeriod { .. })
        ));
    }
}
