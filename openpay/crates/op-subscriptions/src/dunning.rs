//! Dunning: retry policy for failed billing attempts.
//!
//! Stripe and Adyen both ship configurable retry schedules with
//! decreasing-frequency backoffs (1d → 3d → 7d → cancel). We
//! expose the same shape as a pluggable [`DunningPolicy`] enum,
//! and a [`decide`](DunningPolicy::decide) function that maps
//! `(retry_count, last_failure_at, now)` → next action.

use serde::{Deserialize, Serialize};

/// Dunning policy. The retry schedule is encoded as a list of
/// per-retry delays in days. The N-th failure waits
/// `schedule[N]` days before the next attempt; once `N >=
/// schedule.len()` the policy gives up and the subscription
/// cancels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DunningPolicy {
    /// Per-retry delay in days. `schedule.len()` is the maximum
    /// retry count before cancellation.
    pub schedule_days: Vec<u32>,
}

impl Default for DunningPolicy {
    fn default() -> Self {
        // Stripe's "Smart Retries" default ladder, simplified.
        Self {
            schedule_days: vec![1, 3, 5, 7],
        }
    }
}

impl DunningPolicy {
    /// Construct.
    #[must_use]
    pub fn new(schedule_days: Vec<u32>) -> Self {
        Self { schedule_days }
    }

    /// Aggressive policy: retry every day for a week then cancel.
    #[must_use]
    pub fn aggressive_daily() -> Self {
        Self {
            schedule_days: vec![1; 7],
        }
    }

    /// Conservative policy: 1d, 7d, 30d, then cancel.
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            schedule_days: vec![1, 7, 30],
        }
    }

    /// Decide what to do given the current retry count and the
    /// first-failure timestamp.
    ///
    /// - `retry_count` — how many retries have happened so far
    ///   (0 = the original failure; 1+ = subsequent retries).
    /// - `failed_at_unix_secs` — when the first failure occurred.
    /// - `now_unix_secs` — current time.
    #[must_use]
    pub fn decide(
        &self,
        retry_count: u32,
        failed_at_unix_secs: u64,
        now_unix_secs: u64,
    ) -> DunningOutcome {
        let n = retry_count as usize;
        if n >= self.schedule_days.len() {
            return DunningOutcome::Cancel;
        }
        let delay = u64::from(self.schedule_days[n]) * 86_400;
        let next_attempt_at = failed_at_unix_secs + delay;
        if now_unix_secs >= next_attempt_at {
            DunningOutcome::RetryNow
        } else {
            DunningOutcome::Wait { next_attempt_at }
        }
    }
}

/// Output of [`DunningPolicy::decide`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DunningOutcome {
    /// Retry the billing attempt now.
    RetryNow,
    /// Wait until `next_attempt_at` before retrying.
    Wait {
        /// Unix epoch seconds.
        next_attempt_at: u64,
    },
    /// Retry budget exhausted; cancel the subscription.
    Cancel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_schedule_is_1_3_5_7() {
        assert_eq!(DunningPolicy::default().schedule_days, vec![1, 3, 5, 7]);
    }

    #[test]
    fn retry_now_when_delay_elapsed() {
        let p = DunningPolicy::default();
        let outcome = p.decide(0, 1_000, 1_000 + 86_400 + 100);
        assert_eq!(outcome, DunningOutcome::RetryNow);
    }

    #[test]
    fn wait_when_delay_not_elapsed() {
        let p = DunningPolicy::default();
        let outcome = p.decide(0, 1_000, 1_000 + 1_000);
        assert!(matches!(outcome, DunningOutcome::Wait { .. }));
    }

    #[test]
    fn cancel_when_budget_exhausted() {
        let p = DunningPolicy::default();
        // 4 retries scheduled; the 4th tries `schedule[3]`. The
        // 5th probe (retry_count=4) is past the end.
        let outcome = p.decide(4, 1_000, u64::MAX);
        assert_eq!(outcome, DunningOutcome::Cancel);
    }

    #[test]
    fn aggressive_daily_runs_seven_days() {
        let p = DunningPolicy::aggressive_daily();
        for n in 0..7 {
            assert_ne!(
                p.decide(n, 1_000, 1_000 + 86_400 * (u64::from(n) + 1) + 1),
                DunningOutcome::Cancel
            );
        }
        assert_eq!(p.decide(7, 1_000, u64::MAX), DunningOutcome::Cancel);
    }

    #[test]
    fn conservative_has_three_attempts() {
        let p = DunningPolicy::conservative();
        assert_eq!(p.schedule_days, vec![1, 7, 30]);
        assert_eq!(p.decide(3, 1_000, u64::MAX), DunningOutcome::Cancel);
    }
}
