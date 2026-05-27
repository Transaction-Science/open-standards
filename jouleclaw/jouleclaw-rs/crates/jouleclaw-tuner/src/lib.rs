//! L8 — self-tuner (meta-cognitive control plane).
//!
//! The cascade has knobs: how much joule budget to grant, how high to
//! set the global quality floor, how deep the L6 agent may recurse.
//! Left fixed, those knobs drift out of tune as workload changes. L8
//! runs five **damped control loops** — one per knob — that nudge each
//! knob toward its observed operating point without oscillating.
//!
//! A damped loop is the simplest stable controller: `current += damping
//! * (measurement - current)`. With `damping ∈ (0, 1)` the loop is a
//! first-order low-pass filter — it converges monotonically to a
//! constant input and never overshoots, so the cascade can't be driven
//! into a thrash by a noisy measurement. The control-plane analogue of
//! a thermostat, not a PID with a wind-up failure mode.
//!
//! L8 is meta — it spends client CPU only and never answers a query. It
//! reports [`TuningAdvice`]; the consumer decides whether to apply it to
//! their `Runtime` configuration.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// A single first-order damped control loop.
///
/// `current += damping * (measurement - current)`. Stable for any
/// `damping ∈ (0, 1]`; smaller damping = slower, smoother tracking.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DampedLoop {
    /// The loop's current smoothed estimate.
    pub current: f64,
    /// The set-point the loop is asked to hold (advisory; the loop
    /// tracks `measurement`, and `target` is the value advice is
    /// computed against).
    pub target: f64,
    /// Damping factor in `(0, 1]`. Clamped on construction.
    pub damping: f64,
    /// The error from the most recent step (`measurement - prev`).
    pub last_error: f64,
}

impl DampedLoop {
    /// New loop seeded at `current`, holding `target`, with `damping`
    /// clamped into `[f64::EPSILON, 1.0]`.
    pub fn new(current: f64, target: f64, damping: f64) -> Self {
        Self {
            current,
            target,
            damping: damping.clamp(f64::EPSILON, 1.0),
            last_error: 0.0,
        }
    }

    /// Feed a measurement; update `current` and return the new error
    /// signal `measurement - current_after_step`.
    pub fn step(&mut self, measurement: f64) -> f64 {
        let prev = self.current;
        self.current += self.damping * (measurement - prev);
        self.last_error = measurement - self.current;
        self.last_error
    }

    /// Signed distance from the held set-point: `current - target`.
    /// Positive means the loop is running hot relative to target.
    pub fn deviation(&self) -> f64 {
        self.current - self.target
    }
}

/// The five loops L8 maintains.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tuner {
    /// Mean joules per query — the headline efficiency signal.
    pub joules_per_query: DampedLoop,
    /// Rate at which cheaper tiers get promoted ahead of expensive ones.
    pub tier_promotion_rate: DampedLoop,
    /// Global quality floor (min confidence to accept an answer).
    pub confidence_floor: DampedLoop,
    /// L0 cache hit ratio.
    pub cache_hit_rate: DampedLoop,
    /// Mean L6 agent recursion depth.
    pub agent_recursion_depth: DampedLoop,
}

impl Default for Tuner {
    fn default() -> Self {
        Self {
            // Seeded at neutral set-points; damping chosen so each loop
            // tracks over ~10–20 samples without overshoot.
            joules_per_query: DampedLoop::new(1e-3, 1e-3, 0.15),
            tier_promotion_rate: DampedLoop::new(0.1, 0.2, 0.1),
            confidence_floor: DampedLoop::new(0.7, 0.7, 0.1),
            cache_hit_rate: DampedLoop::new(0.3, 0.5, 0.2),
            agent_recursion_depth: DampedLoop::new(1.0, 2.0, 0.1),
        }
    }
}

/// A measurement frame fed to [`Tuner::observe`] once per query (or per
/// batch). Any field left `None` leaves that loop unchanged this tick.
#[derive(Debug, Clone, Copy, Default)]
pub struct Measurement {
    pub joules_per_query: Option<f64>,
    pub tier_promotion_rate: Option<f64>,
    pub confidence_floor: Option<f64>,
    pub cache_hit_rate: Option<f64>,
    pub agent_recursion_depth: Option<f64>,
}

/// What L8 recommends the consumer do with the cascade config. All
/// fields are deltas/factors to apply, not absolute settings — the
/// consumer owns the live config and clamps as it sees fit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TuningAdvice {
    /// Multiply the joule budget by this. >1 loosens (queries are
    /// running cheap), <1 tightens (running hot).
    pub joule_budget_factor: f64,
    /// Add this to the global quality floor. Positive raises the bar.
    pub quality_floor_delta: f64,
    /// Suggested cap on L6 agent recursion depth.
    pub max_recursion_depth: u32,
    /// Whether the cache is underperforming its target (a hint to
    /// pre-warm or enlarge L0).
    pub cache_underperforming: bool,
}

impl Tuner {
    /// New tuner with default seeding.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one measurement frame, stepping each loop whose field is
    /// present.
    pub fn observe(&mut self, m: &Measurement) {
        if let Some(v) = m.joules_per_query {
            self.joules_per_query.step(v);
        }
        if let Some(v) = m.tier_promotion_rate {
            self.tier_promotion_rate.step(v);
        }
        if let Some(v) = m.confidence_floor {
            self.confidence_floor.step(v);
        }
        if let Some(v) = m.cache_hit_rate {
            self.cache_hit_rate.step(v);
        }
        if let Some(v) = m.agent_recursion_depth {
            self.agent_recursion_depth.step(v);
        }
    }

    /// Compute advice from the current loop states.
    pub fn advise(&self) -> TuningAdvice {
        // If joules/query has drifted above target, tighten the budget
        // proportionally; below target, loosen. Bounded to [0.5, 2.0].
        let jpq = self.joules_per_query.current.max(f64::EPSILON);
        let target = self.joules_per_query.target.max(f64::EPSILON);
        let joule_budget_factor = (target / jpq).clamp(0.5, 2.0);

        // If the cache is below target, raise the quality floor slightly
        // (push more queries to deterministic tiers that cache well).
        let cache_gap = self.cache_hit_rate.target - self.cache_hit_rate.current;
        let quality_floor_delta = (cache_gap * 0.05).clamp(-0.05, 0.05);

        let max_recursion_depth = self.agent_recursion_depth.current.round().max(1.0) as u32;

        TuningAdvice {
            joule_budget_factor,
            quality_floor_delta,
            max_recursion_depth,
            cache_underperforming: cache_gap > 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn damped_loop_converges_to_constant() {
        let mut l = DampedLoop::new(0.0, 0.0, 0.2);
        for _ in 0..200 {
            l.step(10.0);
        }
        assert!((l.current - 10.0).abs() < 1e-6, "current = {}", l.current);
    }

    #[test]
    fn damped_loop_never_overshoots() {
        // First-order low-pass: monotone approach, current never exceeds
        // a constant input from below.
        let mut l = DampedLoop::new(0.0, 0.0, 0.5);
        let mut prev = l.current;
        for _ in 0..50 {
            l.step(5.0);
            assert!(l.current <= 5.0 + 1e-12);
            assert!(l.current >= prev - 1e-12, "non-monotone");
            prev = l.current;
        }
    }

    #[test]
    fn damping_clamped_into_unit_interval() {
        assert_eq!(DampedLoop::new(0.0, 0.0, 5.0).damping, 1.0);
        assert!(DampedLoop::new(0.0, 0.0, -1.0).damping > 0.0);
    }

    #[test]
    fn step_returns_error_signal() {
        let mut l = DampedLoop::new(0.0, 0.0, 0.5);
        let err = l.step(10.0); // current → 5.0, error 10-5 = 5
        assert!((err - 5.0).abs() < 1e-9);
        assert!((l.last_error - 5.0).abs() < 1e-9);
    }

    #[test]
    fn deviation_tracks_target() {
        let mut l = DampedLoop::new(0.0, 3.0, 1.0);
        l.step(5.0); // damping 1.0 → current jumps to 5.0
        assert!((l.deviation() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn tuner_default_has_five_loops() {
        let t = Tuner::new();
        assert_eq!(t.confidence_floor.current, 0.7);
        assert_eq!(t.cache_hit_rate.target, 0.5);
    }

    #[test]
    fn observe_only_steps_present_fields() {
        let mut t = Tuner::new();
        let before = t.cache_hit_rate.current;
        t.observe(&Measurement {
            joules_per_query: Some(2e-3),
            ..Default::default()
        });
        assert_eq!(t.cache_hit_rate.current, before); // untouched
        assert_ne!(t.joules_per_query.current, 1e-3); // moved
    }

    #[test]
    fn advise_tightens_budget_when_hot() {
        let mut t = Tuner::new();
        for _ in 0..50 {
            t.observe(&Measurement {
                joules_per_query: Some(4e-3), // 4x target
                ..Default::default()
            });
        }
        let a = t.advise();
        assert!(a.joule_budget_factor < 1.0, "factor = {}", a.joule_budget_factor);
    }

    #[test]
    fn advise_loosens_budget_when_cheap() {
        let mut t = Tuner::new();
        for _ in 0..50 {
            t.observe(&Measurement {
                joules_per_query: Some(2e-4), // well below target
                ..Default::default()
            });
        }
        let a = t.advise();
        assert!(a.joule_budget_factor > 1.0);
    }

    #[test]
    fn advise_flags_cache_underperformance() {
        let mut t = Tuner::new();
        for _ in 0..50 {
            t.observe(&Measurement {
                cache_hit_rate: Some(0.1), // far below 0.5 target
                ..Default::default()
            });
        }
        let a = t.advise();
        assert!(a.cache_underperforming);
        assert!(a.quality_floor_delta > 0.0);
    }

    #[test]
    fn advise_recursion_depth_at_least_one() {
        let t = Tuner::new();
        assert!(t.advise().max_recursion_depth >= 1);
    }

    #[test]
    fn budget_factor_bounded() {
        let mut t = Tuner::new();
        for _ in 0..100 {
            t.observe(&Measurement {
                joules_per_query: Some(1e9),
                ..Default::default()
            });
        }
        assert!(t.advise().joule_budget_factor >= 0.5);
    }
}
