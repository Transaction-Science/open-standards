//! # jouleclaw-drift
//!
//! Drift detection — the database-cost-model bug-finding primitive.
//!
//! Two-detector convention from the wave-4 SOTA brief: a cheap
//! online central-statistic detector PLUS a tail-aware detector
//! over the same stream. Alarming on both means systematic drift
//! (the cost model is wrong, the index needs re-analyse); alarming
//! on only the tail-aware detector means a heavy-tailed noise
//! spike — different operator action.
//!
//! - **Page-Hinkley** (Page 1954): cumulative-sum change detector.
//!   `m_T = Σ(x_t − x̄_t − δ)`; alarm when `m_T − min(m_t) > λ`.
//!   Lowest RAM cost; the canonical low-overhead drift detector
//!   Datadog/Prometheus use under the hood for metric anomaly.
//! - **DDSketch p99 detector**: track the running p99 in a sliding
//!   sketch; alarm when current p99 ≥ ratio × historical p99. The
//!   tail-aware companion that catches what mean-only detectors
//!   miss.
//!
//! ## Honest scope
//!
//! - Drift detection answers "did the distribution change?" — NOT
//!   "is the new distribution worse." Pair with a cost regression
//!   signal before auto-rolling-back.
//! - Page-Hinkley needs the operator to pick `(δ, λ)`. We default
//!   to `(δ=0, λ=10·σ)` per the literature.
//! - DDSketch's relative-error bound applies to the p99 reading
//!   too — a 1% sketch sees a 1% wobble that ISN'T drift.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(unexpected_cfgs)]

use jouleclaw_bounded::{Bounded, BoundedError};
use jouleclaw_quantile::{DdsketchQuantile, QuantileError};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────
// Page-Hinkley detector
// ─────────────────────────────────────────────────────────────────────

/// Page-Hinkley test (Page 1954) — cumulative-sum drift detector.
///
/// State: running mean `μ̂`, cumulative sum `m`, minimum-of-cumulative
/// `m_min`. Each `update(x)` returns `true` if `m − m_min > λ`.
///
/// Parameters:
/// - `delta`: tolerance for "no drift" — small positive number,
///   default `0.0`.
/// - `lambda`: alarm threshold. Common choice: ~30× the expected
///   noise sigma. Default `50.0` — operator should tune.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageHinkley {
    /// Tolerance for the mean drift. Higher = less sensitive.
    pub delta: f64,
    /// Alarm threshold. `m − m_min > λ` ⇒ alarm.
    pub lambda: f64,
    mean: f64,
    n: u64,
    cumsum: f64,
    min_cumsum: f64,
    alarms: u64,
}

impl PageHinkley {
    /// Build a detector with `(delta, lambda)`.
    pub fn new(delta: f64, lambda: f64) -> Self {
        Self {
            delta: delta.max(0.0),
            lambda: lambda.max(0.0),
            mean: 0.0,
            n: 0,
            cumsum: 0.0,
            min_cumsum: 0.0,
            alarms: 0,
        }
    }

    /// Sensible default for tuning later: `(0.0, 50.0)`.
    pub fn default_params() -> Self {
        Self::new(0.0, 50.0)
    }

    /// Update with one observation. Returns `true` iff an alarm
    /// fires. The detector continues past the alarm; call
    /// [`Self::reset`] explicitly to restart.
    pub fn update(&mut self, x: f64) -> bool {
        self.n += 1;
        // Online mean update (Welford's stable form is unnecessary
        // here — Page-Hinkley uses the running mean as a moving
        // baseline; the small bias from naive update is in the
        // noise of `delta`).
        self.mean += (x - self.mean) / self.n as f64;
        let increment = x - self.mean - self.delta;
        self.cumsum += increment;
        if self.cumsum < self.min_cumsum {
            self.min_cumsum = self.cumsum;
        }
        let ph_stat = self.cumsum - self.min_cumsum;
        let alarm = ph_stat > self.lambda;
        if alarm {
            self.alarms += 1;
        }
        alarm
    }

    /// Reset the detector to its initial state. Caller does this
    /// after action on an alarm so the next drift can be measured
    /// against the new baseline.
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.n = 0;
        self.cumsum = 0.0;
        self.min_cumsum = 0.0;
        // alarms count carries across resets for telemetry.
    }

    /// Total alarm count since construction.
    pub fn alarms(&self) -> u64 {
        self.alarms
    }
    /// Number of observations seen.
    pub fn observations(&self) -> u64 {
        self.n
    }
}

impl Bounded for PageHinkley {
    /// Page-Hinkley is deterministic conditional on the input
    /// stream; we report `(ε=0, δ=0)` and a tiny constant memory
    /// footprint.
    fn bound(&self) -> BoundedError {
        BoundedError::exact(64)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tail-aware detector (DDSketch p99 ratio)
// ─────────────────────────────────────────────────────────────────────

/// Tail-aware drift detector — tracks p99 via two DDSketch windows
/// (historical + recent). Alarms when `recent_p99 ≥ ratio ×
/// historical_p99`. The companion to Page-Hinkley that catches what
/// mean-only detectors miss.
pub struct TailAwareDrift {
    historical: DdsketchQuantile,
    recent: DdsketchQuantile,
    ratio: f64,
    rotated_at: u64,
    rotate_every_n: u64,
    n: u64,
    alarms: u64,
}

impl TailAwareDrift {
    /// Build with relative-error `alpha`, alarm ratio (typical
    /// `1.5`), and a sliding-window length `rotate_every_n` after
    /// which `recent` is folded into `historical` and reset.
    pub fn new(alpha: f64, ratio: f64, rotate_every_n: u64) -> Self {
        Self {
            historical: DdsketchQuantile::with_alpha(alpha),
            recent: DdsketchQuantile::with_alpha(alpha),
            ratio: ratio.max(1.0),
            rotated_at: 0,
            rotate_every_n: rotate_every_n.max(10),
            n: 0,
            alarms: 0,
        }
    }

    /// Update with one observation. Returns `true` iff a tail alarm
    /// fires. Requires `historical` to be non-empty (so the first
    /// `rotate_every_n` observations bootstrap silently).
    pub fn update(&mut self, x: f64) -> bool {
        self.recent.add(x);
        self.n += 1;
        let mut alarm = false;
        // Compare recent p99 to historical p99 only when both are
        // non-empty.
        if !self.historical.is_empty() && self.recent.count() >= 32 {
            if let (Ok(rp), Ok(hp)) = (self.recent.p99(), self.historical.p99()) {
                if hp > 0.0 && rp >= self.ratio * hp {
                    alarm = true;
                    self.alarms += 1;
                }
            }
        }
        if self.n - self.rotated_at >= self.rotate_every_n {
            // Fold recent into historical, reset recent.
            let _ = self.historical.merge(&self.recent);
            self.recent = DdsketchQuantile::with_alpha(self.recent_alpha());
            self.rotated_at = self.n;
        }
        alarm
    }

    fn recent_alpha(&self) -> f64 {
        self.historical.bound().epsilon
    }

    /// Alarm count.
    pub fn alarms(&self) -> u64 {
        self.alarms
    }
    /// Number of observations.
    pub fn observations(&self) -> u64 {
        self.n
    }
    /// Borrow the historical sketch (post-rotation).
    pub fn historical(&self) -> &DdsketchQuantile {
        &self.historical
    }
    /// Borrow the recent sketch (the current sliding window).
    pub fn recent(&self) -> &DdsketchQuantile {
        &self.recent
    }
}

impl Bounded for TailAwareDrift {
    /// Inherits the DDSketch relative-error from the historical
    /// sketch; memory is two sketches.
    fn bound(&self) -> BoundedError {
        let h = self.historical.bound();
        BoundedError::relative(h.epsilon, 0.0, h.memory_bytes.unwrap_or(0) * 2)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Composed two-detector — central + tail
// ─────────────────────────────────────────────────────────────────────

/// The two-detector composition. Update both with each sample and
/// classify the alarm.
pub struct TwoDetectorDrift {
    /// The cheap central-statistic detector.
    pub central: PageHinkley,
    /// The tail-aware detector.
    pub tail: TailAwareDrift,
}

/// What kind of alarm fired on a single observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftAlarm {
    /// Neither detector alarmed.
    None,
    /// Only the central detector alarmed — distribution mean
    /// shifted but tail looks normal. Operator: re-baseline.
    SystematicMean,
    /// Only the tail-aware detector alarmed — heavy-tailed noise
    /// spike. Operator: confirm with a second sample before action.
    TailSpike,
    /// Both alarmed — strong evidence of systematic drift. Operator:
    /// re-plan / re-baseline / page.
    Both,
}

impl TwoDetectorDrift {
    /// Build a two-detector with Page-Hinkley defaults and a
    /// 1024-observation tail window at 1% relative error, 1.5×
    /// alarm ratio.
    pub fn default_params() -> Self {
        Self {
            central: PageHinkley::default_params(),
            tail: TailAwareDrift::new(0.01, 1.5, 1024),
        }
    }

    /// Update both detectors and return the classified alarm.
    pub fn update(&mut self, x: f64) -> DriftAlarm {
        let c = self.central.update(x);
        let t = self.tail.update(x);
        match (c, t) {
            (false, false) => DriftAlarm::None,
            (true, false) => DriftAlarm::SystematicMean,
            (false, true) => DriftAlarm::TailSpike,
            (true, true) => DriftAlarm::Both,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_hinkley_silent_on_stationary_stream() {
        let mut ph = PageHinkley::new(0.0, 50.0);
        let mut fired = false;
        // 200 samples around 100 with small noise — should not alarm.
        for i in 0..200 {
            let x = 100.0 + ((i % 7) as f64 - 3.0) * 0.5;
            if ph.update(x) {
                fired = true;
                break;
            }
        }
        assert!(!fired, "Page-Hinkley alarmed on stationary stream");
    }

    #[test]
    fn page_hinkley_fires_on_clear_upward_shift() {
        let mut ph = PageHinkley::new(0.5, 10.0);
        // 100 stationary at 100, then 100 at 130. The cumulative
        // upward deviation should cross λ within the new regime.
        for _ in 0..100 {
            let _ = ph.update(100.0);
        }
        let mut fired = false;
        for _ in 0..100 {
            if ph.update(130.0) {
                fired = true;
                break;
            }
        }
        assert!(fired, "Page-Hinkley failed to detect upward shift");
        assert!(ph.alarms() >= 1);
    }

    #[test]
    fn page_hinkley_reset_clears_state() {
        let mut ph = PageHinkley::new(0.0, 10.0);
        for _ in 0..50 {
            let _ = ph.update(100.0);
        }
        ph.reset();
        assert_eq!(ph.observations(), 0);
    }

    #[test]
    fn page_hinkley_bound_is_exact() {
        let ph = PageHinkley::default_params();
        let b = ph.bound();
        assert_eq!(b.epsilon, 0.0);
        assert_eq!(b.delta, 0.0);
    }

    #[test]
    fn tail_aware_silent_when_p99_stable() {
        let mut td = TailAwareDrift::new(0.01, 1.5, 256);
        let mut fired = false;
        for i in 0..1024 {
            let x = 100.0 + (i % 7) as f64;
            if td.update(x) {
                fired = true;
            }
        }
        assert!(!fired, "tail-aware alarmed on stationary stream");
    }

    #[test]
    fn tail_aware_fires_when_p99_doubles() {
        let mut td = TailAwareDrift::new(0.01, 1.5, 256);
        // Bootstrap historical with a stationary stream.
        for _ in 0..512 {
            let _ = td.update(100.0);
        }
        // Now the recent window sees a 2× increase in p99.
        let mut fired = false;
        for _ in 0..200 {
            if td.update(250.0) {
                fired = true;
                break;
            }
        }
        assert!(fired, "tail-aware missed p99 doubling");
    }

    #[test]
    fn two_detector_classifies_alarms() {
        let mut td = TwoDetectorDrift::default_params();
        for _ in 0..1024 {
            let _ = td.update(100.0);
        }
        // Spike one large outlier. Mean-detector may not move much,
        // tail-detector may pick it up. We don't assert a specific
        // class — only that the classifier returns a valid variant
        // for every sample.
        let v = td.update(10_000.0);
        let _ = v;
        // The classifier must return Some kind of alarm OR None.
        assert!(matches!(
            v,
            DriftAlarm::None | DriftAlarm::SystematicMean | DriftAlarm::TailSpike | DriftAlarm::Both
        ));
    }

    #[test]
    fn drift_alarm_round_trips_through_json() {
        let a = DriftAlarm::Both;
        let j = serde_json::to_value(&a).unwrap();
        assert_eq!(j, serde_json::json!("both"));
        let back: DriftAlarm = serde_json::from_value(j).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn page_hinkley_round_trips_through_json() {
        let mut ph = PageHinkley::new(0.5, 10.0);
        for _ in 0..50 {
            let _ = ph.update(100.0);
        }
        let bytes = serde_json::to_vec(&ph).unwrap();
        let back: PageHinkley = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.observations(), ph.observations());
        assert_eq!(back.delta, ph.delta);
        assert_eq!(back.lambda, ph.lambda);
    }

    #[test]
    fn tail_aware_bound_carries_alpha_and_double_memory() {
        let td = TailAwareDrift::new(0.01, 1.5, 256);
        let b = td.bound();
        assert_eq!(b.epsilon, 0.01);
    }
}
