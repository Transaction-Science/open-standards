//! Settlement cutoff schedules.
//!
//! A cutoff is the rule that decides *when* a batch closes. The
//! settlement engine ticks the cutoff with a wall-clock timestamp;
//! the cutoff says whether to close the currently-open batch.
//!
//! We keep the model deliberately small (three variants — `Daily`,
//! `MultiDaily`, `Manual`) because almost every real-world payment
//! processor uses one of these patterns. Operators with stranger
//! schedules construct a `Manual` cutoff and drive it themselves.

use serde::{Deserialize, Serialize};

/// How frequently the batch closes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cutoff {
    /// Once a day at the given UTC hour `[0, 23]`. Most card
    /// processors run nightly cutoffs around 02:00 ET — operators
    /// pass `7` (UTC) for that.
    Daily {
        /// UTC hour of day, `[0, 23]`.
        hour_utc: u8,
    },

    /// Multiple times a day. Common for instant-rail operators
    /// (`FedNow`, RTP) who pay out same-day multiple times. Hours
    /// must be in ascending order and each in `[0, 23]`.
    MultiDaily {
        /// Sorted ascending UTC hours.
        hours_utc: Vec<u8>,
    },

    /// Never auto-closes. Operator calls
    /// [`crate::SettlementEngine::close_batch`] explicitly. Useful
    /// for tests and operators with bespoke schedules (e.g. weekly,
    /// hourly, signal-driven).
    Manual,
}

impl Cutoff {
    /// Daily at `hour_utc` (`[0, 23]`).
    ///
    /// # Errors
    /// [`crate::Error::Invalid`] if `hour_utc >= 24`.
    pub fn daily(hour_utc: u8) -> crate::Result<Self> {
        if hour_utc >= 24 {
            return Err(crate::Error::Invalid(format!(
                "hour_utc {hour_utc} not in [0,23]"
            )));
        }
        Ok(Self::Daily { hour_utc })
    }

    /// Multiple times a day. Hours must be ascending and all
    /// `[0, 23]`. Empty list is rejected.
    ///
    /// # Errors
    /// [`crate::Error::Invalid`] on empty / out-of-range / unsorted.
    pub fn multi_daily(hours_utc: Vec<u8>) -> crate::Result<Self> {
        if hours_utc.is_empty() {
            return Err(crate::Error::Invalid("multi_daily needs ≥1 hour".into()));
        }
        let mut last: i16 = -1;
        for h in &hours_utc {
            if *h >= 24 {
                return Err(crate::Error::Invalid(format!("hour {h} not in [0,23]")));
            }
            if i16::from(*h) <= last {
                return Err(crate::Error::Invalid(
                    "multi_daily hours must be strictly ascending".into(),
                ));
            }
            last = i16::from(*h);
        }
        Ok(Self::MultiDaily { hours_utc })
    }

    /// Returns `true` if the cutoff fires for the second-since-epoch
    /// `now_unix_secs`, given the previous tick observed at
    /// `last_tick_unix_secs` (the batch's `opened_at` or the last
    /// cutoff fire). The semantics: did a scheduled cutoff hour
    /// occur in `(last_tick, now_unix_secs]`?
    ///
    /// `Manual` never fires automatically.
    #[must_use]
    pub fn should_close(&self, last_tick_unix_secs: u64, now_unix_secs: u64) -> bool {
        if now_unix_secs <= last_tick_unix_secs {
            return false;
        }
        match self {
            Self::Manual => false,
            Self::Daily { hour_utc } => hour_crossed(last_tick_unix_secs, now_unix_secs, *hour_utc),
            Self::MultiDaily { hours_utc } => hours_utc
                .iter()
                .any(|h| hour_crossed(last_tick_unix_secs, now_unix_secs, *h)),
        }
    }
}

/// True iff the `(last, now]` window covers a UTC-`hour:00:00`
/// boundary on any calendar day.
///
/// Implementation walks at most a handful of days — the engine
/// ticks at least daily in practice, and even months of arrears
/// settle in `<366` iterations.
fn hour_crossed(last_unix_secs: u64, now_unix_secs: u64, hour_utc: u8) -> bool {
    const DAY: u64 = 86_400;
    if now_unix_secs <= last_unix_secs {
        return false;
    }
    let hour_offset = u64::from(hour_utc) * 3_600;
    // Find the first cutoff timestamp strictly after `last`.
    let day_floor = (last_unix_secs / DAY) * DAY;
    let mut candidate = day_floor + hour_offset;
    if candidate <= last_unix_secs {
        candidate += DAY;
    }
    candidate <= now_unix_secs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daily_rejects_bad_hour() {
        assert!(Cutoff::daily(24).is_err());
        assert!(Cutoff::daily(7).is_ok());
    }

    #[test]
    fn multi_daily_rejects_empty() {
        assert!(Cutoff::multi_daily(vec![]).is_err());
    }

    #[test]
    fn multi_daily_rejects_unsorted() {
        assert!(Cutoff::multi_daily(vec![10, 4, 16]).is_err());
    }

    #[test]
    fn multi_daily_rejects_duplicate() {
        assert!(Cutoff::multi_daily(vec![6, 6]).is_err());
    }

    #[test]
    fn manual_never_fires() {
        assert!(!Cutoff::Manual.should_close(0, u64::from(u32::MAX)));
    }

    #[test]
    fn daily_fires_when_hour_crossed() {
        // 1700000000 = Tue Nov 14 22:13:20 UTC 2023
        // Daily(07) -> next 07:00 is Nov 15 07:00:00 = 1700031600 + ?
        let last = 1_700_000_000;
        let same_day = last + 3_600; // still Nov 14 23:13:20 UTC
        let next_day_after_07 = 1_700_032_000; // Nov 15 ~07:06 UTC
        let c = Cutoff::daily(7).unwrap();
        assert!(!c.should_close(last, same_day));
        assert!(c.should_close(last, next_day_after_07));
    }

    #[test]
    fn daily_idempotent_within_same_window() {
        // Once we've ticked past 07:00, ticking again at 08:00
        // (`last=07:30, now=08:00`) should not re-fire — the
        // 07:00 boundary is already behind `last`.
        let last_after_fire = 1_700_032_200; // ~07:10
        let later_same_day = 1_700_036_000; // ~08:13
        let c = Cutoff::daily(7).unwrap();
        assert!(!c.should_close(last_after_fire, later_same_day));
    }

    #[test]
    fn multi_daily_fires_on_either_hour() {
        let c = Cutoff::multi_daily(vec![7, 19]).unwrap();
        // 1700060400 = Nov 15 14:20:00, after the 07:00 cutoff.
        let last = 1_700_060_400;
        let after_19 = 1_700_080_000; // ~19:46 UTC
        assert!(c.should_close(last, after_19));
    }
}
