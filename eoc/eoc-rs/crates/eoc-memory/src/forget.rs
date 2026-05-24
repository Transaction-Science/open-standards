//! Ebbinghaus forgetting curve scorer.
//!
//! Hermann Ebbinghaus' 1885 retention experiment gave us the
//! exponential curve `R = exp(-t / S)` where `t` is elapsed time
//! since the last review and `S` is the memory's stability /
//! strength. Repeated review *increases* `S`; idle time leaves it
//! constant.
//!
//! We implement a small deterministic approximation: each retrieval
//! multiplies `S` by `(1 + reinforcement)`.

use serde::{Deserialize, Serialize};

use crate::error::{MemoryError, MemoryResult};

/// Configuration for the Ebbinghaus decay function.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForgetConfig {
    /// Initial stability `S_0`, in milliseconds. Larger = slower
    /// decay. Default 24h = 86_400_000.
    pub initial_stability_ms: f64,
    /// Reinforcement factor applied on every successful retrieval.
    /// Must be `>= 0.0`. Default 1.0 (each review doubles stability).
    pub reinforcement: f64,
    /// Retention threshold below which an item is considered
    /// "forgotten". In `[0.0, 1.0]`. Default 0.2.
    pub threshold: f64,
}

impl ForgetConfig {
    /// Construct + validate.
    pub fn new(initial_stability_ms: f64, reinforcement: f64, threshold: f64) -> MemoryResult<Self> {
        if !(initial_stability_ms > 0.0) {
            return Err(MemoryError::Config(
                "initial_stability_ms must be > 0".into(),
            ));
        }
        if reinforcement < 0.0 {
            return Err(MemoryError::Config("reinforcement must be >= 0".into()));
        }
        if !(0.0..=1.0).contains(&threshold) {
            return Err(MemoryError::Config("threshold must be in [0,1]".into()));
        }
        Ok(Self {
            initial_stability_ms,
            reinforcement,
            threshold,
        })
    }
}

impl Default for ForgetConfig {
    fn default() -> Self {
        Self {
            initial_stability_ms: 86_400_000.0,
            reinforcement: 1.0,
            threshold: 0.2,
        }
    }
}

/// Ebbinghaus-curve scorer.
#[derive(Clone, Debug, Default)]
pub struct EbbinghausScorer {
    cfg: ForgetConfig,
}

impl EbbinghausScorer {
    /// Build a scorer from a [`ForgetConfig`].
    #[must_use]
    pub fn new(cfg: ForgetConfig) -> Self {
        Self { cfg }
    }

    /// Compute stability after `access_count` reviews.
    fn stability(&self, access_count: u32) -> f64 {
        let mut s = self.cfg.initial_stability_ms;
        // Each review multiplies stability by `(1 + reinforcement)`.
        let mul = 1.0 + self.cfg.reinforcement;
        for _ in 0..access_count {
            s *= mul;
        }
        s
    }

    /// Retention `R = exp(-dt / S)` clamped into `[0.0, 1.0]`.
    pub fn retention(&self, last_access_ms: u64, now_ms: u64, access_count: u32) -> f64 {
        let dt = (now_ms.saturating_sub(last_access_ms)) as f64;
        let s = self.stability(access_count);
        let r = (-dt / s).exp();
        if r < 0.0 {
            0.0
        } else if r > 1.0 {
            1.0
        } else {
            r
        }
    }

    /// True iff retention is below the configured threshold.
    pub fn is_forgotten(&self, last_access_ms: u64, now_ms: u64, access_count: u32) -> bool {
        self.retention(last_access_ms, now_ms, access_count) < self.cfg.threshold
    }

    /// Borrow the underlying config.
    #[must_use]
    pub fn config(&self) -> &ForgetConfig {
        &self.cfg
    }
}
