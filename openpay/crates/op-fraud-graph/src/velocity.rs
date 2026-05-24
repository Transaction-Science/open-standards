//! Per-entity / per-edge rolling **velocity counters**.
//!
//! Per-card and per-IP transaction-rate counters are the simplest, most
//! effective fraud feature in the playbook. We give them a fixed time
//! window with second-resolution bucketing, so memory is bounded
//! regardless of traffic.
//!
//! ## Memory
//!
//! Each [`VelocityCounter`] stores `window_buckets` `u32` counters per
//! tracked entity, plus a `last_bucket` timestamp. At default 60s
//! window and 1s bucket resolution: 240 bytes per entity.

use std::collections::HashMap;

use crate::entity::EntityKey;
use crate::error::{Error, Result};

/// Window configuration.
#[derive(Debug, Clone, Copy)]
pub struct VelocityWindow {
    /// Window length in seconds (e.g. 60).
    pub window_secs: u32,
    /// Bucket resolution in seconds (e.g. 1 for per-second). Must
    /// divide `window_secs` evenly.
    pub bucket_secs: u32,
}

impl VelocityWindow {
    /// Number of buckets in the window.
    pub fn bucket_count(&self) -> u32 {
        self.window_secs / self.bucket_secs.max(1)
    }
}

impl Default for VelocityWindow {
    fn default() -> Self {
        Self {
            window_secs: 60,
            bucket_secs: 1,
        }
    }
}

/// Per-entity sliding-window counter.
#[derive(Debug, Clone)]
pub struct VelocityCounter {
    window: VelocityWindow,
    /// `EntityKey` → ring-buffer state.
    state: HashMap<EntityKey, EntityState>,
}

#[derive(Debug, Clone)]
struct EntityState {
    /// Bucket counts. Length = `window.bucket_count()`.
    buckets: Vec<u32>,
    /// Last absolute bucket index we wrote to.
    last_bucket: i64,
    /// Running sum of `buckets[]` to avoid re-summing on every read.
    running_sum: u64,
}

impl VelocityCounter {
    /// Build a counter for the given window.
    pub fn new(window: VelocityWindow) -> Result<Self> {
        if window.window_secs == 0 || window.bucket_secs == 0 {
            return Err(Error::InvalidConfig(
                "window_secs and bucket_secs must be > 0",
            ));
        }
        if window.window_secs % window.bucket_secs != 0 {
            return Err(Error::InvalidConfig(
                "window_secs must be a multiple of bucket_secs",
            ));
        }
        Ok(Self {
            window,
            state: HashMap::new(),
        })
    }

    /// Record one event for `key` at `ts_unix`.
    pub fn record(&mut self, key: EntityKey, ts_unix: i64) {
        let bucket_idx = ts_unix / i64::from(self.window.bucket_secs);
        let n_buckets = self.window.bucket_count() as usize;
        let st = self.state.entry(key).or_insert_with(|| EntityState {
            buckets: vec![0; n_buckets],
            last_bucket: bucket_idx,
            running_sum: 0,
        });
        Self::roll(st, bucket_idx, n_buckets);
        let slot = (bucket_idx.rem_euclid(n_buckets as i64)) as usize;
        st.buckets[slot] = st.buckets[slot].saturating_add(1);
        st.running_sum = st.running_sum.saturating_add(1);
    }

    /// Current count over the trailing window as of `ts_unix`.
    pub fn count(&mut self, key: EntityKey, ts_unix: i64) -> u64 {
        let n_buckets = self.window.bucket_count() as usize;
        let bucket_idx = ts_unix / i64::from(self.window.bucket_secs);
        if let Some(st) = self.state.get_mut(&key) {
            Self::roll(st, bucket_idx, n_buckets);
            st.running_sum
        } else {
            0
        }
    }

    /// Eject any entities whose entire window is stale as of `ts_unix`.
    /// Call periodically to keep memory bounded for one-shot identifiers
    /// (single-use cards, transient IPs).
    pub fn evict_stale(&mut self, ts_unix: i64) {
        let n_buckets = self.window.bucket_count() as i64;
        let bucket_idx = ts_unix / i64::from(self.window.bucket_secs);
        self.state
            .retain(|_, st| bucket_idx - st.last_bucket < n_buckets);
    }

    /// Advance `st` to `bucket_idx`, zeroing buckets that scrolled out.
    fn roll(st: &mut EntityState, bucket_idx: i64, n_buckets: usize) {
        let delta = bucket_idx - st.last_bucket;
        if delta <= 0 {
            return;
        }
        let n = n_buckets as i64;
        if delta >= n {
            // Entire window has expired.
            st.buckets.iter_mut().for_each(|b| *b = 0);
            st.running_sum = 0;
        } else {
            // Zero each scrolled-out bucket.
            for step in 1..=delta {
                let slot_to_clear =
                    ((st.last_bucket + step).rem_euclid(n)) as usize;
                st.running_sum =
                    st.running_sum.saturating_sub(u64::from(st.buckets[slot_to_clear]));
                st.buckets[slot_to_clear] = 0;
            }
        }
        st.last_bucket = bucket_idx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityKind;

    fn key(s: &str) -> EntityKey {
        EntityKey::from_raw(EntityKind::Account, s)
    }

    #[test]
    fn rejects_bad_config() {
        assert!(VelocityCounter::new(VelocityWindow {
            window_secs: 0,
            bucket_secs: 1
        })
        .is_err());
        assert!(VelocityCounter::new(VelocityWindow {
            window_secs: 60,
            bucket_secs: 7
        })
        .is_err());
    }

    #[test]
    fn basic_record_and_count() {
        let mut c = VelocityCounter::new(VelocityWindow::default()).expect("ok");
        let k = key("acc-1");
        for t in 0..10 {
            c.record(k, t);
        }
        assert_eq!(c.count(k, 10), 10);
    }

    #[test]
    fn rolls_off_after_window() {
        let mut c = VelocityCounter::new(VelocityWindow {
            window_secs: 5,
            bucket_secs: 1,
        })
        .expect("ok");
        let k = key("acc-2");
        for t in 0..5 {
            c.record(k, t);
        }
        assert_eq!(c.count(k, 4), 5);
        // Advance well past the window.
        assert_eq!(c.count(k, 100), 0);
    }
}
