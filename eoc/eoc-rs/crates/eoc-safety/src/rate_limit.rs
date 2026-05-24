//! Rate-limiting + simple abuse detection.
//!
//! Token-bucket rate limiter keyed by an arbitrary principal string
//! (user-id, IP, API key, ...). All state lives in-memory; deployments
//! that need persistence can wrap the bucket map in their own store.
//! Time is supplied by the caller (`now_secs`) so the limiter is
//! deterministic and `no_std`-friendly enough to run anywhere.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SafetyError};

/// Token-bucket configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BucketConfig {
    /// Bucket capacity (burst size).
    pub capacity: f64,
    /// Refill rate in tokens per second.
    pub refill_per_sec: f64,
    /// Failed-request threshold to flag as abuse over a 60-second window.
    pub abuse_window_fails: u32,
}

impl BucketConfig {
    /// Sensible default: 60 RPM, burst 30, 20 fails/min triggers abuse flag.
    pub fn default_rpm() -> Self {
        Self {
            capacity: 30.0,
            refill_per_sec: 1.0,
            abuse_window_fails: 20,
        }
    }
}

#[derive(Debug, Clone)]
struct Bucket {
    tokens: f64,
    last_refill: f64,
    fail_count: u32,
    fail_window_start: f64,
}

/// Token-bucket rate limiter.
pub struct RateLimiter {
    config: BucketConfig,
    state: Mutex<HashMap<String, Bucket>>,
}

impl RateLimiter {
    /// Build a limiter with the supplied config.
    pub fn new(config: BucketConfig) -> Self {
        Self {
            config,
            state: Mutex::new(HashMap::new()),
        }
    }

    /// Consume one token for `principal` at logical time `now_secs`.
    /// Returns `Ok(remaining)` on success; `Err(RateLimit)` if denied.
    pub fn check(&self, principal: &str, now_secs: f64) -> Result<f64> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| SafetyError::Other("rate limiter mutex poisoned".into()))?;
        let bucket = guard.entry(principal.to_string()).or_insert(Bucket {
            tokens: self.config.capacity,
            last_refill: now_secs,
            fail_count: 0,
            fail_window_start: now_secs,
        });

        // Refill.
        let elapsed = (now_secs - bucket.last_refill).max(0.0);
        bucket.tokens = (bucket.tokens + elapsed * self.config.refill_per_sec)
            .min(self.config.capacity);
        bucket.last_refill = now_secs;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            Ok(bucket.tokens)
        } else {
            // Roll the abuse window.
            if now_secs - bucket.fail_window_start > 60.0 {
                bucket.fail_window_start = now_secs;
                bucket.fail_count = 0;
            }
            bucket.fail_count = bucket.fail_count.saturating_add(1);
            Err(SafetyError::RateLimit {
                principal: principal.to_string(),
            })
        }
    }

    /// Returns `true` if `principal` has exceeded the abuse threshold
    /// within the rolling 60-second window.
    pub fn is_abusive(&self, principal: &str) -> bool {
        let guard = match self.state.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        guard
            .get(principal)
            .map(|b| b.fail_count >= self.config.abuse_window_fails)
            .unwrap_or(false)
    }
}
