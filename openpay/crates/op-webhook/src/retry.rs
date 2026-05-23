//! Retry scheduling.
//!
//! Exponential backoff with **full jitter** per AWS's "Exponential
//! Backoff and Jitter" guidance (2015) and current Stripe/GitHub
//! practice. The policy is fully deterministic given an injected
//! RNG, which makes tests reproducible.
//!
//! ## Why full jitter and not "equal jitter" or "decorrelated"?
//!
//! Three options are commonly cited:
//!
//! - **Full jitter**: `sleep = rand(0, base * 2^n)`. Maximum
//!   spread; best at avoiding thundering herd.
//! - **Equal jitter**: `sleep = base * 2^(n-1) + rand(0, base * 2^(n-1))`.
//!   Guarantees some minimum delay; smaller spread.
//! - **Decorrelated jitter**: `sleep = rand(base, prev * 3)`.
//!   Couples to previous sleep; useful for adaptive systems.
//!
//! For webhook delivery, full jitter is the AWS-recommended default
//! and matches what Stripe/GitHub appear to do. The OpenPay
//! reference implementation uses full jitter; operators can plug
//! their own [`RetryPolicy`] implementation if they need a
//! different strategy.
//!
//! ## What counts as a retry?
//!
//! Only transient failures: timeouts, transport errors, 5xx HTTP
//! responses, 429s. Permanent failures (signature configured wrong,
//! 4xx other than 408/425/429) should NOT be retried; they're
//! escalated to the dead-letter state immediately. The
//! [`RetryPolicy::should_retry`] method classifies these.

use std::sync::Mutex;

/// Trait an operator can implement to plug in a custom retry
/// strategy. The reference impl is [`ExponentialBackoffPolicy`].
pub trait RetryPolicy: Send + Sync {
    /// Should an HTTP status code or transport error trigger a
    /// retry?
    ///
    /// - `Some(status)`: HTTP response received. 5xx, 408, 425, 429
    ///   → retry. Anything else → no retry.
    /// - `None`: transport-level failure (network, timeout). Always
    ///   retry.
    fn should_retry(&self, http_status: Option<u16>) -> bool;

    /// Delay in seconds before the Nth attempt (0-indexed).
    ///
    /// Caller passes the retry attempt number and the number of
    /// elapsed seconds since the original event was created. The
    /// policy returns `None` if the retry budget is exhausted (no
    /// further attempts).
    fn next_delay_secs(&self, attempt_number: u32, elapsed_secs: u64) -> Option<u64>;

    /// After how many *consecutive* delivery failures (across all
    /// events for one endpoint) the endpoint is auto-disabled.
    fn disable_after_consecutive_failures(&self) -> u32;
}

/// Exponential-backoff-with-full-jitter retry policy.
///
/// Defaults match industry practice: base 1s, cap 1h, total window
/// 72h (Stripe), auto-disable at 10 consecutive failures.
pub struct ExponentialBackoffPolicy {
    /// Base delay in seconds. The unjittered Nth delay is
    /// `base_secs * 2^attempt_number`.
    base_secs: u64,
    /// Maximum single delay in seconds (caps the exponential
    /// growth).
    max_delay_secs: u64,
    /// Total time budget in seconds: if `elapsed > max_age_secs`,
    /// `next_delay_secs` returns `None`.
    max_age_secs: u64,
    /// Auto-disable threshold.
    disable_after: u32,
    /// Pluggable RNG so tests can be deterministic. Wrapped in a
    /// Mutex so the policy is `Send + Sync`.
    rng: Mutex<Box<dyn JitterRng>>,
}

/// A trait for the jitter RNG. Tests inject `FixedJitter(x)` so
/// timings are deterministic; production uses [`SystemJitter`]
/// which calls `std::time` for entropy (good enough for jitter —
/// we're not generating crypto keys).
pub trait JitterRng: Send + Sync {
    /// Return a value uniformly distributed in `[0, upper)`.
    /// `upper > 0` always.
    fn uniform(&mut self, upper: u64) -> u64;
}

/// Production jitter source. Uses `SystemTime`'s nanoseconds as a
/// linear-congruential-style mixer. **Not cryptographically
/// random**, but webhook jitter doesn't need it.
pub struct SystemJitter {
    state: u64,
}

impl SystemJitter {
    /// Construct, seeding from the current time.
    #[must_use]
    pub fn new() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xC0FFEE_DEADBEEF);
        Self { state: nanos | 1 } // never zero (LCG hates 0)
    }
}

impl Default for SystemJitter {
    fn default() -> Self {
        Self::new()
    }
}

impl JitterRng for SystemJitter {
    fn uniform(&mut self, upper: u64) -> u64 {
        // Numerical Recipes LCG constants.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        if upper == 0 { 0 } else { self.state % upper }
    }
}

/// Fully deterministic jitter source for tests. Returns `value` (or
/// `upper - 1` if `value >= upper`) every call.
pub struct FixedJitter(pub u64);

impl JitterRng for FixedJitter {
    fn uniform(&mut self, upper: u64) -> u64 {
        if upper == 0 {
            0
        } else if self.0 >= upper {
            upper - 1
        } else {
            self.0
        }
    }
}

impl ExponentialBackoffPolicy {
    /// Construct with custom parameters.
    ///
    /// Defaults are returned by [`Self::stripe_like`].
    #[must_use]
    pub fn new(
        base_secs: u64,
        max_delay_secs: u64,
        max_age_secs: u64,
        disable_after: u32,
        rng: Box<dyn JitterRng>,
    ) -> Self {
        Self {
            base_secs: base_secs.max(1),
            max_delay_secs: max_delay_secs.max(1),
            max_age_secs,
            disable_after: disable_after.max(1),
            rng: Mutex::new(rng),
        }
    }

    /// Stripe-like defaults: base 1s, cap 1h, total 72h, disable
    /// after 10 consecutive failures. Uses [`SystemJitter`].
    #[must_use]
    pub fn stripe_like() -> Self {
        Self::new(1, 3600, 72 * 3600, 10, Box::new(SystemJitter::new()))
    }

    /// Construct with a [`FixedJitter`] for deterministic tests.
    #[must_use]
    pub fn deterministic(
        base_secs: u64,
        max_delay_secs: u64,
        max_age_secs: u64,
        disable_after: u32,
        fixed_jitter: u64,
    ) -> Self {
        Self::new(
            base_secs,
            max_delay_secs,
            max_age_secs,
            disable_after,
            Box::new(FixedJitter(fixed_jitter)),
        )
    }
}

impl RetryPolicy for ExponentialBackoffPolicy {
    fn should_retry(&self, http_status: Option<u16>) -> bool {
        match http_status {
            // Transport-level failure (timeout, DNS, etc.).
            None => true,
            // 5xx, 408 (timeout), 425 (too early), 429 (too many
            // requests) → retry.
            Some(s) if (500..600).contains(&s) => true,
            Some(408) | Some(425) | Some(429) => true,
            // 2xx, 3xx, other 4xx → no retry.
            Some(_) => false,
        }
    }

    fn next_delay_secs(&self, attempt_number: u32, elapsed_secs: u64) -> Option<u64> {
        if elapsed_secs >= self.max_age_secs {
            return None;
        }
        // Exponential: base * 2^attempt, capped.
        // Saturate the exponent to avoid overflow at e.g. 2^64.
        let exponent = attempt_number.min(62);
        let computed = self
            .base_secs
            .saturating_mul(1u64 << exponent)
            .min(self.max_delay_secs);
        // Full jitter: pick uniformly in [0, computed].
        let upper = computed.saturating_add(1);
        let jittered = self
            .rng
            .lock()
            .expect("poisoned jitter mutex")
            .uniform(upper);
        Some(jittered)
    }

    fn disable_after_consecutive_failures(&self) -> u32 {
        self.disable_after
    }
}

/// Convenience function for callers that want the full-jitter math
/// without constructing a policy: given a base, cap, and attempt,
/// returns `[0, min(base*2^n, cap)]` with `rng_value` modulating.
#[must_use]
pub fn jitter_full(base_secs: u64, max_delay_secs: u64, attempt: u32, rng_value: u64) -> u64 {
    let exponent = attempt.min(62);
    let computed = base_secs
        .saturating_mul(1u64 << exponent)
        .min(max_delay_secs);
    let upper = computed.saturating_add(1);
    if upper == 0 { 0 } else { rng_value % upper }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_retry_transport_failure() {
        let p = ExponentialBackoffPolicy::stripe_like();
        assert!(p.should_retry(None));
    }

    #[test]
    fn should_retry_5xx() {
        let p = ExponentialBackoffPolicy::stripe_like();
        assert!(p.should_retry(Some(500)));
        assert!(p.should_retry(Some(502)));
        assert!(p.should_retry(Some(503)));
        assert!(p.should_retry(Some(599)));
    }

    #[test]
    fn should_retry_specific_4xx() {
        let p = ExponentialBackoffPolicy::stripe_like();
        assert!(p.should_retry(Some(408)));
        assert!(p.should_retry(Some(425)));
        assert!(p.should_retry(Some(429)));
    }

    #[test]
    fn should_not_retry_2xx() {
        // Success itself doesn't trigger a retry — defensive check;
        // dispatcher only calls should_retry on non-success.
        let p = ExponentialBackoffPolicy::stripe_like();
        assert!(!p.should_retry(Some(200)));
        assert!(!p.should_retry(Some(201)));
    }

    #[test]
    fn should_not_retry_most_4xx() {
        let p = ExponentialBackoffPolicy::stripe_like();
        assert!(!p.should_retry(Some(400)));
        assert!(!p.should_retry(Some(401)));
        assert!(!p.should_retry(Some(403)));
        assert!(!p.should_retry(Some(404)));
        assert!(!p.should_retry(Some(410)));
        assert!(!p.should_retry(Some(422)));
    }

    #[test]
    fn next_delay_is_zero_when_jitter_is_zero() {
        let p = ExponentialBackoffPolicy::deterministic(1, 3600, 72 * 3600, 10, 0);
        // Full jitter picks in [0, base*2^n]. With FixedJitter(0) =>
        // always 0.
        assert_eq!(p.next_delay_secs(0, 0).unwrap(), 0);
        assert_eq!(p.next_delay_secs(5, 0).unwrap(), 0);
    }

    #[test]
    fn next_delay_caps_at_max_delay() {
        // base=1, max=10, attempt=20 → computed = min(1*2^20, 10) = 10.
        // FixedJitter(50) clamps to upper-1 = 10.
        let p = ExponentialBackoffPolicy::deterministic(1, 10, 72 * 3600, 10, 50);
        let d = p.next_delay_secs(20, 0).unwrap();
        assert!(d <= 10, "delay {d} exceeded cap");
    }

    #[test]
    fn next_delay_returns_none_when_max_age_exceeded() {
        let p = ExponentialBackoffPolicy::deterministic(1, 3600, 100, 10, 0);
        assert!(p.next_delay_secs(0, 101).is_none());
        assert!(p.next_delay_secs(0, 100).is_none());
    }

    #[test]
    fn next_delay_still_returns_when_under_max_age() {
        let p = ExponentialBackoffPolicy::deterministic(1, 3600, 100, 10, 0);
        assert!(p.next_delay_secs(0, 99).is_some());
    }

    #[test]
    fn disable_after_threshold_returned() {
        let p = ExponentialBackoffPolicy::deterministic(1, 3600, 72 * 3600, 7, 0);
        assert_eq!(p.disable_after_consecutive_failures(), 7);
    }

    #[test]
    fn jitter_full_at_zero_attempt_with_zero_rng() {
        // attempt 0 → computed = min(base*1, cap). rng_value=0 → 0.
        assert_eq!(jitter_full(2, 100, 0, 0), 0);
    }

    #[test]
    fn jitter_full_modulates_within_bounds() {
        // attempt 3, base 1, cap 100 → computed = min(8, 100) = 8.
        // rng_value = 5 → 5 % 9 = 5.
        assert_eq!(jitter_full(1, 100, 3, 5), 5);
    }

    #[test]
    fn jitter_full_caps_at_max_delay() {
        // attempt 30, base 1 → 2^30 ~ 1B, capped to 100.
        let d = jitter_full(1, 100, 30, u64::MAX / 2);
        assert!(d <= 100);
    }

    #[test]
    fn system_jitter_produces_varying_output() {
        // Smoke test only — confirm it doesn't lock at zero.
        let mut j = SystemJitter::new();
        let a = j.uniform(1000);
        let b = j.uniform(1000);
        let c = j.uniform(1000);
        // Vanishingly unlikely all three are equal.
        assert!(!(a == b && b == c));
    }

    #[test]
    fn system_jitter_zero_upper_is_zero() {
        let mut j = SystemJitter::new();
        assert_eq!(j.uniform(0), 0);
    }

    #[test]
    fn fixed_jitter_returns_value_clamped() {
        let mut j = FixedJitter(50);
        assert_eq!(j.uniform(100), 50);
        // 50 >= 10 → clamp to 9.
        assert_eq!(j.uniform(10), 9);
        // 50 with upper=0 → 0.
        assert_eq!(j.uniform(0), 0);
    }

    #[test]
    fn new_clamps_zero_inputs_to_one() {
        // Defensive: a misconfigured policy with base=0 would
        // produce infinite immediate retries. The constructor
        // bumps these to 1.
        let p = ExponentialBackoffPolicy::deterministic(0, 0, 72 * 3600, 0, 0);
        // base_secs and max_delay_secs both forced to 1.
        // disable_after forced to 1.
        assert_eq!(p.disable_after_consecutive_failures(), 1);
    }
}
