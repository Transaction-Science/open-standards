//! `BillingScheduler` — pure period math and due-now classification.
//!
//! Day / Week math is simple arithmetic on unix seconds.
//! Month / Year math is calendar-aware (Jan 31 → Feb 28/29 → Mar 31)
//! and uses the `time` crate for safe date manipulation. All
//! periods are `[start, end)`: `start` inclusive, `end` exclusive.

use time::{Duration, OffsetDateTime};

use crate::error::Result;
use crate::plan::{Interval, Plan};
use crate::subscription::{Status, Subscription};

/// Compute the first period bounds for a fresh subscription
/// starting at `now_unix_secs`. Trials don't get a separate
/// "trial period"; the first billing period runs from `now` to
/// `next` — the scheduler later checks `Subscription::status` to
/// decide whether to actually bill.
#[must_use]
pub fn first_period(plan: &Plan, now_unix_secs: u64) -> (u64, u64) {
    let start = match plan.trial_days {
        Some(d) if d > 0 => now_unix_secs + (u64::from(d) * 86_400),
        _ => now_unix_secs,
    };
    let end = advance(start, plan.interval, plan.interval_count);
    (start, end)
}

/// Advance `from` by `count × interval`. Calendar-aware for
/// `Month` / `Year` (Jan 31 → Feb 28).
#[must_use]
pub fn advance(from_unix_secs: u64, interval: Interval, count: u32) -> u64 {
    match interval {
        Interval::Day => from_unix_secs + (u64::from(count) * 86_400),
        Interval::Week => from_unix_secs + (u64::from(count) * 7 * 86_400),
        Interval::Month => add_months(from_unix_secs, i64::from(count)),
        Interval::Year => add_months(from_unix_secs, i64::from(count) * 12),
    }
}

/// Add `months` to a unix timestamp, clamping the day to the
/// destination month's length (Jan 31 + 1 month = Feb 28/29).
fn add_months(from_unix_secs: u64, months: i64) -> u64 {
    let Ok(secs_i64) = i64::try_from(from_unix_secs) else {
        // Far-future timestamp — saturating fallback.
        return from_unix_secs;
    };
    let Ok(dt) = OffsetDateTime::from_unix_timestamp(secs_i64) else {
        return from_unix_secs;
    };
    let mut year = dt.year();
    let mut month = i32::from(u8::from(dt.month()));
    let delta = i32::try_from(months.rem_euclid(12)).unwrap_or(0);
    let year_delta = months.div_euclid(12);

    month += delta;
    if month > 12 {
        month -= 12;
        year += 1;
    }
    year += i32::try_from(year_delta).unwrap_or(0);

    let month_u8 = u8::try_from(month).unwrap_or(u8::from(dt.month()));
    let month_enum = time::Month::try_from(month_u8).unwrap_or(dt.month());
    let max_day = days_in_month(year, month_enum);
    let day = u8::min(dt.day(), max_day);
    let date = time::Date::from_calendar_date(year, month_enum, day).unwrap_or_else(|_| dt.date());
    let new_dt = dt.replace_date(date);
    u64::try_from(new_dt.unix_timestamp()).unwrap_or(from_unix_secs)
}

fn days_in_month(year: i32, month: time::Month) -> u8 {
    let next_month_year = if matches!(month, time::Month::December) {
        year + 1
    } else {
        year
    };
    let next_month = month.next();
    let first_of_next =
        time::Date::from_calendar_date(next_month_year, next_month, 1).expect("valid date");
    let first_of_this = time::Date::from_calendar_date(year, month, 1).expect("valid date");
    let diff: Duration = first_of_next - first_of_this;
    u8::try_from(diff.whole_days()).unwrap_or(28)
}

/// What the scheduler thinks should happen next for a given
/// subscription at `now_unix_secs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DueState {
    /// Current period still open; nothing to do.
    NotDue,
    /// Trial has ended; transition to `Active` and bill the first
    /// real period.
    TrialEnded,
    /// Current period has closed; roll to the next period and
    /// emit a billing attempt.
    PeriodRollover,
    /// `cancel_at_period_end` was set and the period rolled —
    /// transition to `Canceled` rather than rolling.
    CancelAtPeriodEnd,
}

/// Pure-function classifier: compare `now` against the
/// subscription's bounds and decide what should happen.
#[derive(Debug, Clone, Copy)]
pub struct BillingScheduler;

impl BillingScheduler {
    /// Classify a subscription's status relative to `now`.
    #[must_use]
    pub fn classify(sub: &Subscription, now_unix_secs: u64) -> DueState {
        if sub.status.is_terminal() {
            return DueState::NotDue;
        }
        // Trial promotion check.
        if let Status::Trialing {
            trial_end_unix_secs,
        } = sub.status
            && now_unix_secs >= trial_end_unix_secs
        {
            return DueState::TrialEnded;
        }
        if matches!(sub.status, Status::Paused { .. }) {
            return DueState::NotDue;
        }
        if now_unix_secs < sub.current_period_end_unix_secs {
            return DueState::NotDue;
        }
        if sub.cancel_at_period_end {
            DueState::CancelAtPeriodEnd
        } else {
            DueState::PeriodRollover
        }
    }

    /// Apply the classification's transition to the subscription.
    /// Mutates `sub` in place. Returns `Ok(state)` so callers can
    /// fan out on the resulting `DueState` (e.g. trigger a billing
    /// attempt on `PeriodRollover`).
    ///
    /// # Errors
    /// Bubbles up state-machine errors. Should not happen with a
    /// well-formed subscription.
    pub fn tick(sub: &mut Subscription, now_unix_secs: u64) -> Result<DueState> {
        let state = Self::classify(sub, now_unix_secs);
        match state {
            DueState::NotDue => {}
            DueState::TrialEnded => {
                sub.promote_from_trial()?;
            }
            DueState::PeriodRollover => {
                let next_start = sub.current_period_end_unix_secs;
                let next_end = advance(next_start, sub.plan.interval, sub.plan.interval_count);
                sub.current_period_start_unix_secs = next_start;
                sub.current_period_end_unix_secs = next_end;
            }
            DueState::CancelAtPeriodEnd => {
                sub.cancel_now(now_unix_secs);
            }
        }
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Plan;
    use crate::subscription::Subscription;
    use op_core::{Currency, Money, PaymentMethod, VaultRef};

    fn plan(interval: Interval, count: u32) -> Plan {
        Plan::new("p", Money::from_minor(1000, Currency::USD), interval, count).unwrap()
    }

    fn fresh_sub(plan: Plan, start: u64) -> Subscription {
        Subscription::new("c", plan, PaymentMethod::Vault(VaultRef::new("tok")), start).unwrap()
    }

    #[test]
    fn day_advance_is_exact_seconds() {
        let now = 1_700_000_000;
        assert_eq!(advance(now, Interval::Day, 5), now + 5 * 86_400);
    }

    #[test]
    fn week_advance_seven_days_each() {
        let now = 1_700_000_000;
        assert_eq!(advance(now, Interval::Week, 2), now + 14 * 86_400);
    }

    #[test]
    fn month_advance_calendar_aware() {
        // 2024-01-31 00:00:00 UTC = 1706659200.
        let jan_31 = 1_706_659_200_u64;
        let advanced = advance(jan_31, Interval::Month, 1);
        let feb_29 = OffsetDateTime::from_unix_timestamp(i64::try_from(advanced).unwrap()).unwrap();
        assert_eq!(feb_29.year(), 2024);
        assert_eq!(u8::from(feb_29.month()), 2);
        // 2024 is a leap year so Feb has 29 days.
        assert_eq!(feb_29.day(), 29);
    }

    #[test]
    fn year_advance_is_twelve_months() {
        let jan_15_2024 = 1_705_276_800_u64; // 2024-01-15 00:00:00 UTC
        let advanced = advance(jan_15_2024, Interval::Year, 1);
        let next_year =
            OffsetDateTime::from_unix_timestamp(i64::try_from(advanced).unwrap()).unwrap();
        assert_eq!(next_year.year(), 2025);
        assert_eq!(u8::from(next_year.month()), 1);
        assert_eq!(next_year.day(), 15);
    }

    #[test]
    fn classify_not_due_within_period() {
        let sub = fresh_sub(plan(Interval::Month, 1), 1_700_000_000);
        let mid = sub.current_period_start_unix_secs + 100;
        assert_eq!(BillingScheduler::classify(&sub, mid), DueState::NotDue);
    }

    #[test]
    fn classify_rollover_at_period_end() {
        let sub = fresh_sub(plan(Interval::Day, 1), 1_700_000_000);
        let after = sub.current_period_end_unix_secs + 1;
        assert_eq!(
            BillingScheduler::classify(&sub, after),
            DueState::PeriodRollover
        );
    }

    #[test]
    fn classify_trial_ended() {
        let p = plan(Interval::Month, 1).with_trial_days(7);
        let sub = fresh_sub(p, 1_700_000_000);
        let after_trial = 1_700_000_000 + 8 * 86_400;
        assert_eq!(
            BillingScheduler::classify(&sub, after_trial),
            DueState::TrialEnded
        );
    }

    #[test]
    fn tick_advances_period() {
        let mut sub = fresh_sub(plan(Interval::Day, 1), 1_700_000_000);
        let original_end = sub.current_period_end_unix_secs;
        let next = BillingScheduler::tick(&mut sub, original_end + 10).unwrap();
        assert_eq!(next, DueState::PeriodRollover);
        assert_eq!(sub.current_period_start_unix_secs, original_end);
        assert_eq!(
            sub.current_period_end_unix_secs,
            advance(original_end, Interval::Day, 1)
        );
    }

    #[test]
    fn tick_cancels_at_period_end() {
        let mut sub = fresh_sub(plan(Interval::Day, 1), 1_700_000_000);
        sub.schedule_cancel_at_period_end();
        let after = sub.current_period_end_unix_secs + 1;
        let state = BillingScheduler::tick(&mut sub, after).unwrap();
        assert_eq!(state, DueState::CancelAtPeriodEnd);
        assert!(sub.status.is_terminal());
    }

    #[test]
    fn paused_subscriptions_dont_roll() {
        let mut sub = fresh_sub(plan(Interval::Day, 1), 1_700_000_000);
        sub.pause(1_700_000_500).unwrap();
        let way_after = sub.current_period_end_unix_secs + 86_400 * 30;
        assert_eq!(
            BillingScheduler::classify(&sub, way_after),
            DueState::NotDue
        );
    }

    #[test]
    fn canceled_subscriptions_never_due() {
        let mut sub = fresh_sub(plan(Interval::Day, 1), 1_700_000_000);
        sub.cancel_now(1_700_000_100);
        let after = sub.current_period_end_unix_secs + 86_400;
        assert_eq!(BillingScheduler::classify(&sub, after), DueState::NotDue);
    }
}
