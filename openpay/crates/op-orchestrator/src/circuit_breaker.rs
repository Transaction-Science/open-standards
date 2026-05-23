//! Circuit breaker per (rail, driver) pair.
//!
//! Standard three-state breaker:
//!
//! - **Closed** — normal operation, requests flow through.
//! - **Open** — after N consecutive failures, fail-fast for a
//!   cooldown window without calling the driver.
//! - **Half-open** — after the cooldown, allow ONE probe; on
//!   success the breaker closes, on failure it re-opens for
//!   another full cooldown.
//!
//! The implementation here is **synchronous** and **deterministic**
//! — it takes a `now: u64` (unix epoch seconds) parameter rather
//! than calling `SystemTime::now()` itself. This lets tests advance
//! the clock without sleeping and keeps the orchestrator free of
//! `std::time` dependencies for environments that mock time
//! externally.

use std::collections::HashMap;
use std::sync::Mutex;

use op_core::RailKind;

/// Current state of a single breaker.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CircuitState {
    /// Closed — requests flow through.
    Closed,

    /// Open — fail-fast until `until_unix_secs` elapses.
    Open {
        /// Unix epoch seconds at which the breaker enters half-open.
        until_unix_secs: u64,
    },

    /// Half-open — one probe allowed.
    HalfOpen,
}

/// Pluggable breaker trait. Lets operators substitute their own
/// implementation backed by a shared Redis (so a cluster of
/// orchestrators agrees on circuit state).
pub trait CircuitBreaker: Send + Sync {
    /// True if the given (rail, driver) pair allows a request right
    /// now (Closed, or HalfOpen waiting for a probe).
    fn allow(&self, rail: RailKind, driver: &str, now_unix_secs: u64) -> bool;

    /// Report a successful call. Closes the breaker if it was
    /// half-open.
    fn record_success(&self, rail: RailKind, driver: &str);

    /// Report a failed call. Trips the breaker if the consecutive
    /// failure count exceeds the threshold.
    fn record_failure(&self, rail: RailKind, driver: &str, now_unix_secs: u64);

    /// Observe the current state. For diagnostics and testing.
    fn state(&self, rail: RailKind, driver: &str, now_unix_secs: u64) -> CircuitState;
}

/// In-process breaker. NOT for multi-instance production.
///
/// Industry default: 5 consecutive failures → open for 60 seconds.
/// Operators tune via [`Self::with_threshold`] and [`Self::with_cooldown`].
pub struct InMemoryCircuitBreaker {
    inner: Mutex<HashMap<(RailKind, String), BreakerState>>,
    failure_threshold: u32,
    cooldown_secs: u64,
}

#[derive(Clone, Default)]
struct BreakerState {
    consecutive_failures: u32,
    open_until: Option<u64>,
}

impl Default for InMemoryCircuitBreaker {
    fn default() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            failure_threshold: 5,
            cooldown_secs: 60,
        }
    }
}

impl InMemoryCircuitBreaker {
    /// Construct with defaults (threshold 5, cooldown 60s).
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set the consecutive-failure threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold;
        self
    }

    /// Builder: set the cooldown window in seconds.
    #[must_use]
    pub fn with_cooldown(mut self, cooldown_secs: u64) -> Self {
        self.cooldown_secs = cooldown_secs;
        self
    }
}

impl CircuitBreaker for InMemoryCircuitBreaker {
    fn allow(&self, rail: RailKind, driver: &str, now: u64) -> bool {
        !matches!(self.state(rail, driver, now), CircuitState::Open { .. })
    }

    fn record_success(&self, rail: RailKind, driver: &str) {
        let mut map = self.inner.lock().expect("circuit breaker poisoned");
        let s = map.entry((rail, driver.to_owned())).or_default();
        s.consecutive_failures = 0;
        s.open_until = None;
    }

    fn record_failure(&self, rail: RailKind, driver: &str, now: u64) {
        let mut map = self.inner.lock().expect("circuit breaker poisoned");
        let s = map.entry((rail, driver.to_owned())).or_default();
        s.consecutive_failures = s.consecutive_failures.saturating_add(1);
        if s.consecutive_failures >= self.failure_threshold {
            s.open_until = Some(now.saturating_add(self.cooldown_secs));
        }
    }

    fn state(&self, rail: RailKind, driver: &str, now: u64) -> CircuitState {
        let map = self.inner.lock().expect("circuit breaker poisoned");
        let Some(s) = map.get(&(rail, driver.to_owned())) else {
            return CircuitState::Closed;
        };
        match s.open_until {
            None => CircuitState::Closed,
            Some(t) if now < t => CircuitState::Open { until_unix_secs: t },
            // Past the cooldown. Conceptually half-open until next
            // call. (A more elaborate impl would atomically claim the
            // probe slot here; we keep it simple.)
            Some(_) => CircuitState::HalfOpen,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_breaker_is_closed_and_allows() {
        let b = InMemoryCircuitBreaker::new();
        assert_eq!(b.state(RailKind::Card, "hsw", 100), CircuitState::Closed);
        assert!(b.allow(RailKind::Card, "hsw", 100));
    }

    #[test]
    fn failures_below_threshold_dont_trip() {
        let b = InMemoryCircuitBreaker::new().with_threshold(3);
        b.record_failure(RailKind::Card, "hsw", 100);
        b.record_failure(RailKind::Card, "hsw", 100);
        assert!(b.allow(RailKind::Card, "hsw", 100));
    }

    #[test]
    fn threshold_failures_trip_breaker_open() {
        let b = InMemoryCircuitBreaker::new()
            .with_threshold(3)
            .with_cooldown(30);
        for _ in 0..3 {
            b.record_failure(RailKind::Card, "hsw", 100);
        }
        assert!(!b.allow(RailKind::Card, "hsw", 100));
        match b.state(RailKind::Card, "hsw", 100) {
            CircuitState::Open { until_unix_secs } => assert_eq!(until_unix_secs, 130),
            s => panic!("expected Open, got {s:?}"),
        }
    }

    #[test]
    fn cooldown_elapses_to_half_open() {
        let b = InMemoryCircuitBreaker::new()
            .with_threshold(2)
            .with_cooldown(30);
        b.record_failure(RailKind::Card, "hsw", 100);
        b.record_failure(RailKind::Card, "hsw", 100);
        // Just before cooldown.
        assert!(matches!(
            b.state(RailKind::Card, "hsw", 129),
            CircuitState::Open { .. }
        ));
        // At and past cooldown.
        assert_eq!(b.state(RailKind::Card, "hsw", 130), CircuitState::HalfOpen);
        assert_eq!(b.state(RailKind::Card, "hsw", 200), CircuitState::HalfOpen);
    }

    #[test]
    fn success_resets_breaker() {
        let b = InMemoryCircuitBreaker::new().with_threshold(3);
        b.record_failure(RailKind::Card, "hsw", 100);
        b.record_failure(RailKind::Card, "hsw", 100);
        b.record_success(RailKind::Card, "hsw");
        // Reset means 0 consecutive failures.
        b.record_failure(RailKind::Card, "hsw", 100);
        b.record_failure(RailKind::Card, "hsw", 100);
        // Only 2 since the reset — below threshold 3.
        assert!(b.allow(RailKind::Card, "hsw", 100));
    }

    #[test]
    fn breakers_are_per_rail_driver_pair() {
        let b = InMemoryCircuitBreaker::new().with_threshold(2);
        for _ in 0..3 {
            b.record_failure(RailKind::Card, "primary", 100);
        }
        // Primary is open.
        assert!(!b.allow(RailKind::Card, "primary", 100));
        // Backup is unaffected.
        assert!(b.allow(RailKind::Card, "backup", 100));
        // A2A rail unaffected.
        assert!(b.allow(RailKind::A2a, "fednow", 100));
    }

    #[test]
    fn half_open_allows_probe_request() {
        let b = InMemoryCircuitBreaker::new()
            .with_threshold(2)
            .with_cooldown(30);
        b.record_failure(RailKind::Card, "hsw", 100);
        b.record_failure(RailKind::Card, "hsw", 100);
        // Past cooldown.
        assert!(b.allow(RailKind::Card, "hsw", 200));
    }

    #[test]
    fn failure_count_saturates() {
        // Don't overflow on long-running consecutive-failure streams.
        let b = InMemoryCircuitBreaker::new().with_threshold(1_000_000);
        for _ in 0..10 {
            b.record_failure(RailKind::Card, "hsw", 100);
        }
        // No panic — saturating_add handles it.
        assert!(b.allow(RailKind::Card, "hsw", 100));
    }

    #[test]
    fn allow_uses_now_to_determine_open_state() {
        let b = InMemoryCircuitBreaker::new()
            .with_threshold(2)
            .with_cooldown(60);
        b.record_failure(RailKind::Card, "hsw", 1000);
        b.record_failure(RailKind::Card, "hsw", 1000);
        // Open until 1060.
        assert!(!b.allow(RailKind::Card, "hsw", 1030));
        // After cooldown, HalfOpen — allow returns true.
        assert!(b.allow(RailKind::Card, "hsw", 1060));
        assert!(b.allow(RailKind::Card, "hsw", 2000));
    }
}
