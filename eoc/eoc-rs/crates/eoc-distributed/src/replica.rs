//! Replica auto-scaling controller (Ray Serve / KEDA / vLLM-Serve style).
//!
//! The controller observes a fleet-level signal (queue depth, latency,
//! joule-per-token, etc.) and decides whether to add or drop replicas.
//! Two budgets keep the loop sane: a hard floor / ceiling on replica
//! count, and a cooldown that prevents flapping. Both budgets mirror
//! Ray Serve's `Deployment(min_replicas, max_replicas, ...)`.

use std::time::{Duration, Instant};

use crate::error::{DistributedError, Result};

/// Auto-scaling configuration.
#[derive(Debug, Clone, Copy)]
pub struct ScaleConfig {
    /// Lower bound on replica count.
    pub min_replicas: u32,
    /// Upper bound on replica count.
    pub max_replicas: u32,
    /// Per-replica target (e.g. requests in-flight). Above the target,
    /// the controller scales up; below `target * low_band` it scales down.
    pub target_per_replica: f64,
    /// Fraction of `target_per_replica` below which we scale down.
    /// Mirrors Ray Serve's `downscale_smoothing_factor`.
    pub low_band: f64,
    /// Minimum gap between successive scaling decisions.
    pub cooldown: Duration,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            min_replicas: 1,
            max_replicas: 32,
            target_per_replica: 4.0,
            low_band: 0.5,
            cooldown: Duration::from_secs(30),
        }
    }
}

/// A scaling decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleDecision {
    /// No change this tick.
    Hold,
    /// Add `n` replicas.
    Up(u32),
    /// Remove `n` replicas.
    Down(u32),
}

/// Replica controller.
#[derive(Debug)]
pub struct ReplicaController {
    cfg: ScaleConfig,
    current: u32,
    last_decision: Option<Instant>,
}

impl ReplicaController {
    /// Construct, starting at `initial` replicas. Clamped to `[min, max]`.
    pub fn new(cfg: ScaleConfig, initial: u32) -> Result<Self> {
        if cfg.min_replicas > cfg.max_replicas {
            return Err(DistributedError::ScaleRejected(
                "min_replicas > max_replicas".into(),
            ));
        }
        let current = initial.clamp(cfg.min_replicas, cfg.max_replicas);
        Ok(Self {
            cfg,
            current,
            last_decision: None,
        })
    }

    /// Current replica count.
    pub fn current(&self) -> u32 {
        self.current
    }

    /// Reactive decide-and-apply step. `signal` is the fleet-level
    /// utilisation metric (e.g. in-flight requests fleet-wide).
    pub fn observe(&mut self, signal: f64) -> ScaleDecision {
        self.observe_at(signal, Instant::now())
    }

    /// Same as [`observe`](Self::observe) with an explicit clock — for
    /// deterministic tests.
    pub fn observe_at(&mut self, signal: f64, now: Instant) -> ScaleDecision {
        if let Some(last) = self.last_decision {
            if now.duration_since(last) < self.cfg.cooldown {
                return ScaleDecision::Hold;
            }
        }
        let per_replica = signal / (self.current as f64).max(1.0);
        let decision = if per_replica > self.cfg.target_per_replica
            && self.current < self.cfg.max_replicas
        {
            // Scale up enough to bring per_replica back to target.
            let want = (signal / self.cfg.target_per_replica).ceil() as u32;
            let want = want.clamp(self.cfg.min_replicas, self.cfg.max_replicas);
            if want > self.current {
                ScaleDecision::Up(want - self.current)
            } else {
                ScaleDecision::Hold
            }
        } else if per_replica < self.cfg.target_per_replica * self.cfg.low_band
            && self.current > self.cfg.min_replicas
        {
            let want = (signal / self.cfg.target_per_replica).ceil() as u32;
            let want = want.clamp(self.cfg.min_replicas, self.cfg.max_replicas);
            if want < self.current {
                ScaleDecision::Down(self.current - want)
            } else {
                ScaleDecision::Hold
            }
        } else {
            ScaleDecision::Hold
        };

        match decision {
            ScaleDecision::Up(n) => {
                self.current = (self.current + n).min(self.cfg.max_replicas);
                self.last_decision = Some(now);
            }
            ScaleDecision::Down(n) => {
                self.current = self.current.saturating_sub(n).max(self.cfg.min_replicas);
                self.last_decision = Some(now);
            }
            ScaleDecision::Hold => {}
        }

        decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_up_when_hot() {
        let mut rc = ReplicaController::new(ScaleConfig::default(), 1).expect("ok");
        let d = rc.observe(40.0);
        assert!(matches!(d, ScaleDecision::Up(_)));
        assert!(rc.current() > 1);
    }

    #[test]
    fn cooldown_blocks_back_to_back() {
        let cfg = ScaleConfig {
            cooldown: Duration::from_secs(60),
            ..ScaleConfig::default()
        };
        let mut rc = ReplicaController::new(cfg, 1).expect("ok");
        let now = Instant::now();
        let first = rc.observe_at(40.0, now);
        assert!(matches!(first, ScaleDecision::Up(_)));
        let second = rc.observe_at(40.0, now + Duration::from_secs(1));
        assert_eq!(second, ScaleDecision::Hold);
    }

    #[test]
    fn invalid_bounds_rejected() {
        let cfg = ScaleConfig {
            min_replicas: 4,
            max_replicas: 2,
            ..ScaleConfig::default()
        };
        assert!(ReplicaController::new(cfg, 1).is_err());
    }
}
