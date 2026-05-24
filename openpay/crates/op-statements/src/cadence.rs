//! Recurring statement cadences.
//!
//! A cadence enumerates a sequence of half-open [`Period`]s
//! `[start, end)` along the unix-epoch-seconds timeline. Every period
//! is deterministic given an anchor and a length: no calendar
//! semantics, no time zones, no DST surprises. Operators who need
//! "first business day of the month" semantics build that on top.
//!
//! Cadences:
//! - [`Cadence::Daily`] — 86,400-second windows.
//! - [`Cadence::Weekly`] — 7 * 86,400.
//! - [`Cadence::Monthly`] — 30 * 86,400 (a "banking month"). Real
//!   calendar months are caller-supplied via [`Cadence::Custom`].
//! - [`Cadence::Custom`] — arbitrary period length in seconds.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

const SECS_PER_DAY: u64 = 86_400;
const SECS_PER_WEEK: u64 = 7 * SECS_PER_DAY;
const SECS_PER_BANKING_MONTH: u64 = 30 * SECS_PER_DAY;

/// A closed period on the unix timeline.
///
/// `[start_unix_secs, end_unix_secs]` with `end >= start`. Statement
/// generators interpret this half-open in practice; the field naming
/// is intentionally inclusive so caller code reads naturally.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Period {
    /// Start (unix epoch seconds).
    pub start_unix_secs: u64,
    /// End (unix epoch seconds). Must be >= `start_unix_secs`.
    pub end_unix_secs: u64,
}

impl Period {
    /// Construct.
    ///
    /// # Errors
    /// [`Error::InvalidPeriod`] if `end < start`.
    pub fn new(start_unix_secs: u64, end_unix_secs: u64) -> Result<Self> {
        if end_unix_secs < start_unix_secs {
            return Err(Error::InvalidPeriod {
                start: start_unix_secs,
                end: end_unix_secs,
            });
        }
        Ok(Self {
            start_unix_secs,
            end_unix_secs,
        })
    }

    /// Length in seconds.
    #[must_use]
    pub const fn length_secs(self) -> u64 {
        self.end_unix_secs.saturating_sub(self.start_unix_secs)
    }
}

/// Recurrence schedule for [`Statement`](crate::Statement) generation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Cadence {
    /// 86,400-second windows.
    Daily,
    /// Seven-day windows.
    Weekly,
    /// 30-day windows ("banking month").
    Monthly,
    /// Arbitrary period length in seconds.
    Custom {
        /// Period length in seconds. Must be non-zero.
        period_secs: u64,
    },
}

impl Cadence {
    /// Period length in seconds.
    ///
    /// # Errors
    /// [`Error::DegenerateCadence`] when `Custom { period_secs: 0 }`.
    pub fn period_secs(self) -> Result<u64> {
        match self {
            Self::Daily => Ok(SECS_PER_DAY),
            Self::Weekly => Ok(SECS_PER_WEEK),
            Self::Monthly => Ok(SECS_PER_BANKING_MONTH),
            Self::Custom { period_secs } => {
                if period_secs == 0 {
                    Err(Error::DegenerateCadence(0))
                } else {
                    Ok(period_secs)
                }
            }
        }
    }

    /// Enumerate all periods of this cadence anchored at
    /// `anchor_unix_secs` that fall fully within
    /// `[from, to]`. The first period starts at the smallest multiple
    /// of `period_secs` from the anchor that is `>= from`.
    ///
    /// # Errors
    /// [`Error::DegenerateCadence`] / [`Error::InvalidPeriod`] from
    /// constituent constructors.
    pub fn enumerate(
        self,
        anchor_unix_secs: u64,
        from_unix_secs: u64,
        to_unix_secs: u64,
    ) -> Result<Vec<Period>> {
        let period = self.period_secs()?;
        if to_unix_secs < from_unix_secs {
            return Err(Error::InvalidPeriod {
                start: from_unix_secs,
                end: to_unix_secs,
            });
        }
        // Smallest `anchor + k * period >= from`.
        let mut start = if from_unix_secs <= anchor_unix_secs {
            anchor_unix_secs
        } else {
            let delta = from_unix_secs - anchor_unix_secs;
            let k = delta.div_ceil(period);
            anchor_unix_secs.saturating_add(k.saturating_mul(period))
        };
        let mut out = Vec::new();
        while start.saturating_add(period) <= to_unix_secs.saturating_add(1) {
            let end = start.saturating_add(period).saturating_sub(1);
            out.push(Period::new(start, end)?);
            start = start.saturating_add(period);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daily_enumerates_three_periods() {
        let anchor = 1_700_000_000;
        let periods = Cadence::Daily
            .enumerate(anchor, anchor, anchor + 3 * SECS_PER_DAY - 1)
            .unwrap();
        assert_eq!(periods.len(), 3);
        assert_eq!(periods[0].start_unix_secs, anchor);
        assert_eq!(periods[0].end_unix_secs, anchor + SECS_PER_DAY - 1);
        assert_eq!(periods[1].start_unix_secs, anchor + SECS_PER_DAY);
        assert_eq!(periods[2].start_unix_secs, anchor + 2 * SECS_PER_DAY);
    }

    #[test]
    fn weekly_period_length() {
        let p = Cadence::Weekly
            .enumerate(0, 0, SECS_PER_WEEK - 1)
            .unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].length_secs(), SECS_PER_WEEK - 1);
    }

    #[test]
    fn monthly_30_day_window() {
        let p = Cadence::Monthly
            .enumerate(0, 0, SECS_PER_BANKING_MONTH - 1)
            .unwrap();
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn custom_zero_rejected() {
        let r = Cadence::Custom { period_secs: 0 }.enumerate(0, 0, 100);
        assert!(matches!(r, Err(Error::DegenerateCadence(0))));
    }

    #[test]
    fn enumerate_with_from_after_anchor_aligns_to_next_boundary() {
        // Anchor at t=0, daily cadence, query starts mid-day -> first
        // emitted period starts at the next day boundary.
        let from = SECS_PER_DAY / 2;
        let to = SECS_PER_DAY * 3 - 1;
        let p = Cadence::Daily.enumerate(0, from, to).unwrap();
        // Expect periods [day1, day2] (day0 is partially in the past).
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].start_unix_secs, SECS_PER_DAY);
    }

    #[test]
    fn invalid_range_rejected() {
        let r = Cadence::Daily.enumerate(0, 100, 50);
        assert!(matches!(r, Err(Error::InvalidPeriod { .. })));
    }
}
