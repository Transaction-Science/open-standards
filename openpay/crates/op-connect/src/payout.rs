//! Sub-merchant payout orchestration.
//!
//! Payouts are the outbound leg of the platform model: after a
//! sub-merchant's collected balance crosses the schedule's trigger
//! point (daily / weekly / monthly / on-demand), funds move from the
//! platform's pooled account to the sub-merchant's external bank
//! account.
//!
//! Actual rail submission is delegated to `op-batch` (NACHA / SEPA /
//! BACS) or `op-rails-a2a` (FedNow / RTP) — this module computes
//! schedules and reserves and produces the [`PayoutInstruction`] the
//! rail driver consumes.
//!
//! ## Holdback / rolling reserve
//!
//! Per-account `reserve_pct` (typically 5-15%) is withheld on each
//! payout to cover chargebacks. The reserve releases after a 180-day
//! window (the Visa / Mastercard chargeback dispute deadline). This
//! crate exposes the reserve computation; the actual escrow / release
//! machinery lives one layer up.

use chrono::{DateTime, Datelike, Days, NaiveDate, Utc, Weekday};
use op_core::{Currency, Money};
use serde::{Deserialize, Serialize};

use crate::account::AccountId;
use crate::error::{Error, Result};

/// Cadence at which payouts fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PayoutMode {
    /// Real-time / minutes-grain, as soon as funds are eligible.
    Instant,
    /// Once per business day.
    Daily,
    /// Once per week on `day_of_week`.
    Weekly,
    /// Once per month on `day_of_month`.
    Monthly,
    /// No automatic schedule — sub-merchant pulls funds manually.
    OnDemand,
}

/// Payout schedule attached to a connected account.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PayoutSchedule {
    /// Cadence.
    pub mode: PayoutMode,
    /// Required when `mode == Weekly`.
    pub day_of_week: Option<Weekday>,
    /// Required when `mode == Monthly`. 1..=28 (avoid month-end edge cases).
    pub day_of_month: Option<u8>,
    /// Settlement delay in days (e.g. T+2 for card payments).
    pub delay_days: u8,
    /// True if the sub-merchant is approved for instant payouts where eligible.
    pub instant_eligible: bool,
    /// Fraction of every payout to withhold as a rolling reserve.
    /// Range: `0.0..=1.0` (validated at construction).
    pub reserve_pct: f32,
}

impl PayoutSchedule {
    /// Construct a schedule and validate it.
    ///
    /// # Errors
    /// [`Error::InvalidPayoutSchedule`] on any out-of-range field.
    pub fn try_new(
        mode: PayoutMode,
        day_of_week: Option<Weekday>,
        day_of_month: Option<u8>,
        delay_days: u8,
        instant_eligible: bool,
        reserve_pct: f32,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&reserve_pct) {
            return Err(Error::InvalidPayoutSchedule {
                reason: format!("reserve_pct {reserve_pct} not in [0.0, 1.0]"),
            });
        }
        if delay_days > 30 {
            return Err(Error::InvalidPayoutSchedule {
                reason: format!("delay_days {delay_days} > 30 (no real schedule needs this)"),
            });
        }
        if matches!(mode, PayoutMode::Weekly) && day_of_week.is_none() {
            return Err(Error::InvalidPayoutSchedule {
                reason: "weekly schedule requires day_of_week".into(),
            });
        }
        if matches!(mode, PayoutMode::Monthly) {
            match day_of_month {
                Some(d) if (1..=28).contains(&d) => {}
                Some(d) => {
                    return Err(Error::InvalidPayoutSchedule {
                        reason: format!("day_of_month {d} not in 1..=28"),
                    });
                }
                None => {
                    return Err(Error::InvalidPayoutSchedule {
                        reason: "monthly schedule requires day_of_month".into(),
                    });
                }
            }
        }
        Ok(Self {
            mode,
            day_of_week,
            day_of_month,
            delay_days,
            instant_eligible,
            reserve_pct,
        })
    }

    /// Compute the next payout date given the date funds became
    /// available.
    ///
    /// Adds `delay_days` to `available_on`, then rounds up to the next
    /// scheduled cadence point.
    ///
    /// `OnDemand` and `Instant` return `available_on + delay_days`.
    #[must_use]
    pub fn next_payout_date(&self, available_on: NaiveDate) -> NaiveDate {
        let base = available_on
            .checked_add_days(Days::new(u64::from(self.delay_days)))
            .unwrap_or(available_on);
        match self.mode {
            PayoutMode::Instant | PayoutMode::OnDemand | PayoutMode::Daily => base,
            PayoutMode::Weekly => {
                let target = self.day_of_week.unwrap_or(Weekday::Mon);
                // Walk forward at most 6 days.
                let mut d = base;
                for _ in 0..=6 {
                    if d.weekday() == target {
                        return d;
                    }
                    d = d.checked_add_days(Days::new(1)).unwrap_or(d);
                }
                d
            }
            PayoutMode::Monthly => {
                let dom = u32::from(self.day_of_month.unwrap_or(1));
                // Same-month if `base.day() <= dom`, else next month.
                if base.day() <= dom {
                    base.with_day(dom).unwrap_or(base)
                } else {
                    let (year, month) = if base.month() == 12 {
                        (base.year() + 1, 1)
                    } else {
                        (base.year(), base.month() + 1)
                    };
                    NaiveDate::from_ymd_opt(year, month, dom).unwrap_or(base)
                }
            }
        }
    }

    /// Split a gross payout amount into `(payout, reserve)`.
    ///
    /// Reserve is computed via integer rounding on minor units to keep
    /// money exact.
    ///
    /// # Errors
    /// [`Error::Overflow`] if the reserve computation overflows.
    pub fn apply_reserve(&self, gross: Money) -> Result<(Money, Money)> {
        let cur = gross.currency;
        // reserve_minor = gross.minor * reserve_pct, rounded down.
        let reserve_minor = ((gross.minor_units as f64) * f64::from(self.reserve_pct)).floor() as i64;
        let reserve = Money::from_minor(reserve_minor, cur);
        let payout = gross.checked_sub(reserve).map_err(|_| Error::Overflow)?;
        Ok((payout, reserve))
    }
}

/// An instruction to the rail driver (`op-batch` / `op-rails-a2a`) to
/// submit a payout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PayoutInstruction {
    /// Sub-merchant receiving the funds.
    pub destination: AccountId,
    /// Net amount paid out (after reserve withholding).
    pub net_amount: Money,
    /// Amount withheld to reserve, if any.
    pub reserve_amount: Money,
    /// Scheduled disbursement date.
    pub scheduled_for: NaiveDate,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
}

impl PayoutInstruction {
    /// Build a payout instruction from a schedule + gross amount +
    /// availability date.
    ///
    /// # Errors
    /// Bubbles validation failures from [`PayoutSchedule::apply_reserve`].
    pub fn build(
        destination: &AccountId,
        gross: Money,
        available_on: NaiveDate,
        schedule: &PayoutSchedule,
    ) -> Result<Self> {
        if gross.currency != Currency::USD
            && gross.currency != Currency::EUR
            && gross.currency != Currency::GBP
            && gross.currency != Currency::INR
            && gross.currency != Currency::BRL
            && gross.currency != Currency::JPY
            && gross.currency != Currency::CNY
        {
            // Allow but log — exotic currency, operator should know.
            tracing::debug!(currency = %gross.currency, "payout in non-mainline currency");
        }
        let (net, reserve) = schedule.apply_reserve(gross)?;
        Ok(Self {
            destination: destination.clone(),
            net_amount: net,
            reserve_amount: reserve,
            scheduled_for: schedule.next_payout_date(available_on),
            created_at: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_reserve_pct_rejected() {
        let err = PayoutSchedule::try_new(PayoutMode::Daily, None, None, 2, false, 1.5)
            .expect_err("out of range");
        assert!(matches!(err, Error::InvalidPayoutSchedule { .. }));
    }

    #[test]
    fn weekly_without_dow_rejected() {
        let err = PayoutSchedule::try_new(PayoutMode::Weekly, None, None, 2, false, 0.0)
            .expect_err("weekly needs dow");
        assert!(matches!(err, Error::InvalidPayoutSchedule { .. }));
    }

    #[test]
    fn daily_two_day_delay_lands_t_plus_2() {
        let s = PayoutSchedule::try_new(PayoutMode::Daily, None, None, 2, false, 0.0)
            .expect("ok");
        let available = NaiveDate::from_ymd_opt(2026, 1, 10).expect("date");
        let next = s.next_payout_date(available);
        assert_eq!(next, NaiveDate::from_ymd_opt(2026, 1, 12).expect("date"));
    }

    #[test]
    fn weekly_rolls_to_target_day() {
        // Target = Friday. Start = Wednesday + 0-day delay = Wednesday.
        // Next payout should be that Friday.
        let s = PayoutSchedule::try_new(
            PayoutMode::Weekly,
            Some(Weekday::Fri),
            None,
            0,
            false,
            0.0,
        )
        .expect("ok");
        // 2026-01-14 is a Wednesday.
        let wed = NaiveDate::from_ymd_opt(2026, 1, 14).expect("date");
        assert_eq!(wed.weekday(), Weekday::Wed);
        let next = s.next_payout_date(wed);
        // Friday is 2026-01-16.
        assert_eq!(next, NaiveDate::from_ymd_opt(2026, 1, 16).expect("date"));
    }

    #[test]
    fn monthly_rolls_to_next_month_after_dom() {
        let s = PayoutSchedule::try_new(
            PayoutMode::Monthly,
            None,
            Some(15),
            0,
            false,
            0.0,
        )
        .expect("ok");
        let nov_20 = NaiveDate::from_ymd_opt(2026, 11, 20).expect("date");
        let next = s.next_payout_date(nov_20);
        assert_eq!(next, NaiveDate::from_ymd_opt(2026, 12, 15).expect("date"));
    }

    #[test]
    fn reserve_split_is_exact_in_minor_units() {
        let s = PayoutSchedule::try_new(PayoutMode::Daily, None, None, 0, false, 0.10)
            .expect("ok");
        let gross = Money::from_minor(10_000, Currency::USD);
        let (net, reserve) = s.apply_reserve(gross).expect("ok");
        assert_eq!(reserve, Money::from_minor(1_000, Currency::USD));
        assert_eq!(net, Money::from_minor(9_000, Currency::USD));
    }
}
