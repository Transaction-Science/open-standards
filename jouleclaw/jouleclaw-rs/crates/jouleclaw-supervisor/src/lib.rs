//! L9 — supervisor (meta-cognitive control plane).
//!
//! L9 watches the cascade for *pathologies*: failure modes that no
//! single tier can see because they are properties of the sequence of
//! dispatches, not of any one answer. It consumes a sliding window of
//! [`Heartbeat`]s (one per dispatch) and runs a set of
//! [`PathologyDetector`]s, each emitting [`Pathology`] records the
//! caller acts on (alert, fall back to a safe tier, trip L10's kill
//! switch).
//!
//! Four detectors ship by default:
//! - [`RunawayDetector`] — joules/query above a multiple of the rolling
//!   median for N consecutive samples ⇒ [`Pathology::JouleExplosion`].
//! - [`StarvationDetector`] — a registered tier hasn't been dispatched
//!   in the last M heartbeats ⇒ [`Pathology::Starvation`].
//! - [`OscillationDetector`] — tier choice for one query class flaps
//!   between two tiers ⇒ [`Pathology::Oscillation`].
//! - [`OracleInversionDetector`] — an answer accepted at high confidence
//!   was later marked unsuccessful ⇒ [`Pathology::OracleInversion`].

#![forbid(unsafe_code)]

use jouleclaw_cascade::types::TierId;
use serde::{Deserialize, Serialize};

/// One dispatch's vital signs.
#[derive(Debug, Clone, Copy)]
pub struct Heartbeat {
    /// Query-class fingerprint (caller-supplied).
    pub query_fingerprint: u64,
    /// Tier that handled this dispatch.
    pub tier_used: TierId,
    /// Joules spent.
    pub joules_spent: f64,
    /// Wall-clock latency in milliseconds.
    pub latency_ms: u64,
    /// Whether the answer was accepted.
    pub success: bool,
    /// Confidence in `[0, 1]` at decision time.
    pub confidence: f32,
}

/// A detected failure mode. `Serialize` so it can ride in a receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pathology {
    /// Joules/query ran away above the historical median.
    JouleExplosion {
        observed_joules: f64,
        median_joules: f64,
        consecutive: u32,
    },
    /// A tier hasn't been used in the recent window.
    Starvation {
        #[serde(with = "tier_wire")]
        tier: TierId,
        idle_for: usize,
    },
    /// Tier choice alternates between two tiers for one query class.
    Oscillation {
        query_fingerprint: u64,
        #[serde(with = "tier_wire")]
        tier_a: TierId,
        #[serde(with = "tier_wire")]
        tier_b: TierId,
        flips: u32,
    },
    /// A high-confidence answer was later found wrong.
    OracleInversion {
        query_fingerprint: u64,
        #[serde(with = "tier_wire")]
        tier: TierId,
        confidence: f32,
    },
}

/// A detector consumes the current window and emits zero or more
/// pathologies. Detectors may hold state across `check` calls.
pub trait PathologyDetector: Send {
    fn check(&mut self, window: &[Heartbeat]) -> Vec<Pathology>;
    /// Short name for diagnostics.
    fn name(&self) -> &'static str;
}

/// Joule-explosion detector.
pub struct RunawayDetector {
    /// Multiple of the median that counts as an explosion.
    pub multiple: f64,
    /// Consecutive over-threshold samples required to fire.
    pub consecutive_required: u32,
}

impl Default for RunawayDetector {
    fn default() -> Self {
        Self {
            multiple: 10.0,
            consecutive_required: 3,
        }
    }
}

impl PathologyDetector for RunawayDetector {
    fn check(&mut self, window: &[Heartbeat]) -> Vec<Pathology> {
        if window.len() < 2 {
            return Vec::new();
        }
        let median = median_joules(window);
        if median <= 0.0 {
            return Vec::new();
        }
        let threshold = median * self.multiple;
        // Count the trailing run of over-threshold samples.
        let mut run = 0u32;
        let mut worst = 0.0f64;
        for hb in window.iter().rev() {
            if hb.joules_spent > threshold {
                run += 1;
                worst = worst.max(hb.joules_spent);
            } else {
                break;
            }
        }
        if run >= self.consecutive_required {
            vec![Pathology::JouleExplosion {
                observed_joules: worst,
                median_joules: median,
                consecutive: run,
            }]
        } else {
            Vec::new()
        }
    }

    fn name(&self) -> &'static str {
        "runaway"
    }
}

/// Starvation detector — flags registered tiers absent from the window.
pub struct StarvationDetector {
    /// Tiers that *should* see traffic.
    pub watched: Vec<TierId>,
    /// Window-tail length over which absence counts as starvation.
    pub idle_window: usize,
}

impl StarvationDetector {
    pub fn new(watched: Vec<TierId>, idle_window: usize) -> Self {
        Self {
            watched,
            idle_window: idle_window.max(1),
        }
    }
}

impl PathologyDetector for StarvationDetector {
    fn check(&mut self, window: &[Heartbeat]) -> Vec<Pathology> {
        if window.len() < self.idle_window {
            return Vec::new();
        }
        let tail = &window[window.len() - self.idle_window..];
        let mut out = Vec::new();
        for &tier in &self.watched {
            let seen = tail.iter().any(|hb| hb.tier_used == tier);
            if !seen {
                out.push(Pathology::Starvation {
                    tier,
                    idle_for: self.idle_window,
                });
            }
        }
        out
    }

    fn name(&self) -> &'static str {
        "starvation"
    }
}

/// Oscillation detector — per query class, fires when the tier choice
/// flips between exactly two tiers repeatedly.
pub struct OscillationDetector {
    /// Minimum number of A↔B flips to fire.
    pub min_flips: u32,
}

impl Default for OscillationDetector {
    fn default() -> Self {
        Self { min_flips: 3 }
    }
}

impl PathologyDetector for OscillationDetector {
    fn check(&mut self, window: &[Heartbeat]) -> Vec<Pathology> {
        use std::collections::HashMap;
        // Group heartbeats by query class, preserving order.
        let mut by_class: HashMap<u64, Vec<TierId>> = HashMap::new();
        for hb in window {
            by_class
                .entry(hb.query_fingerprint)
                .or_default()
                .push(hb.tier_used);
        }
        let mut out = Vec::new();
        for (fp, seq) in by_class {
            if seq.len() < 2 {
                continue;
            }
            // Distinct tiers must be exactly two for a clean oscillation.
            let mut distinct: Vec<TierId> = Vec::new();
            for &t in &seq {
                if !distinct.contains(&t) {
                    distinct.push(t);
                }
            }
            if distinct.len() != 2 {
                continue;
            }
            let mut flips = 0u32;
            for w in seq.windows(2) {
                if w[0] != w[1] {
                    flips += 1;
                }
            }
            if flips >= self.min_flips {
                out.push(Pathology::Oscillation {
                    query_fingerprint: fp,
                    tier_a: distinct[0],
                    tier_b: distinct[1],
                    flips,
                });
            }
        }
        // Deterministic ordering by fingerprint.
        out.sort_by(|a, b| match (a, b) {
            (
                Pathology::Oscillation { query_fingerprint: x, .. },
                Pathology::Oscillation { query_fingerprint: y, .. },
            ) => x.cmp(y),
            _ => std::cmp::Ordering::Equal,
        });
        out
    }

    fn name(&self) -> &'static str {
        "oscillation"
    }
}

/// Oracle-inversion detector — a high-confidence accepted answer that
/// later turns up unsuccessful in the same window.
pub struct OracleInversionDetector {
    /// Confidence at or above which an unsuccessful answer is an
    /// inversion (the system was *sure* and *wrong*).
    pub confidence_threshold: f32,
}

impl Default for OracleInversionDetector {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.9,
        }
    }
}

impl PathologyDetector for OracleInversionDetector {
    fn check(&mut self, window: &[Heartbeat]) -> Vec<Pathology> {
        let mut out = Vec::new();
        for hb in window {
            if !hb.success && hb.confidence >= self.confidence_threshold {
                out.push(Pathology::OracleInversion {
                    query_fingerprint: hb.query_fingerprint,
                    tier: hb.tier_used,
                    confidence: hb.confidence,
                });
            }
        }
        out
    }

    fn name(&self) -> &'static str {
        "oracle_inversion"
    }
}

/// The supervisor: a bounded heartbeat window plus a set of detectors.
pub struct Supervisor {
    window: Vec<Heartbeat>,
    capacity: usize,
    detectors: Vec<Box<dyn PathologyDetector>>,
}

impl Supervisor {
    /// New supervisor with the given window capacity and no detectors.
    pub fn new(capacity: usize) -> Self {
        Self {
            window: Vec::new(),
            capacity: capacity.max(1),
            detectors: Vec::new(),
        }
    }

    /// New supervisor with the four default detectors wired up. Pass the
    /// tiers that should be watched for starvation.
    pub fn with_default_detectors(capacity: usize, watched: Vec<TierId>) -> Self {
        let mut s = Self::new(capacity);
        s.add_detector(Box::new(RunawayDetector::default()));
        s.add_detector(Box::new(StarvationDetector::new(watched, capacity.min(16))));
        s.add_detector(Box::new(OscillationDetector::default()));
        s.add_detector(Box::new(OracleInversionDetector::default()));
        s
    }

    pub fn add_detector(&mut self, d: Box<dyn PathologyDetector>) {
        self.detectors.push(d);
    }

    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Append a heartbeat, evicting the oldest if at capacity.
    pub fn record(&mut self, hb: Heartbeat) {
        if self.window.len() >= self.capacity {
            self.window.remove(0);
        }
        self.window.push(hb);
    }

    /// Run every detector against the current window and collect all
    /// pathologies.
    pub fn scan(&mut self) -> Vec<Pathology> {
        let mut out = Vec::new();
        for d in &mut self.detectors {
            out.extend(d.check(&self.window));
        }
        out
    }
}

/// Median of `joules_spent` over a heartbeat window. Returns 0.0 for an
/// empty window. Uses the lower-middle element for even lengths — exact
/// midpoint choice is immaterial to the threshold test.
fn median_joules(window: &[Heartbeat]) -> f64 {
    if window.is_empty() {
        return 0.0;
    }
    let mut vals: Vec<f64> = window.iter().map(|h| h.joules_spent).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals[vals.len() / 2]
}

/// Serde helper: serialize `TierId` as its wire tag.
mod tier_wire {
    use super::TierId;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(t: &TierId, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(t.wire_tag())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<TierId, D::Error> {
        let _ = String::deserialize(d)?;
        // Pathology records are emit-only diagnostics; round-trip maps to
        // the coarse cache tier (the tag string is what carries meaning).
        Ok(TierId::L0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{L1Primitive, L3ModelId};

    fn hb(fp: u64, tier: TierId, joules: f64, success: bool, conf: f32) -> Heartbeat {
        Heartbeat {
            query_fingerprint: fp,
            tier_used: tier,
            joules_spent: joules,
            latency_ms: 10,
            success,
            confidence: conf,
        }
    }

    #[test]
    fn runaway_fires_on_consecutive_spikes() {
        let mut d = RunawayDetector::default();
        let mut w = Vec::new();
        for _ in 0..10 {
            w.push(hb(1, TierId::L0, 1e-6, true, 0.9));
        }
        for _ in 0..3 {
            w.push(hb(1, TierId::L3(L3ModelId(0)), 1.0, true, 0.9));
        }
        let p = d.check(&w);
        assert_eq!(p.len(), 1);
        matches!(p[0], Pathology::JouleExplosion { .. });
    }

    #[test]
    fn runaway_quiet_when_stable() {
        let mut d = RunawayDetector::default();
        let w: Vec<_> = (0..10).map(|_| hb(1, TierId::L0, 1e-6, true, 0.9)).collect();
        assert!(d.check(&w).is_empty());
    }

    #[test]
    fn runaway_needs_consecutive_not_scattered() {
        let mut d = RunawayDetector::default();
        let mut w = Vec::new();
        for _ in 0..10 {
            w.push(hb(1, TierId::L0, 1e-6, true, 0.9));
        }
        // One spike then back to normal — trailing run is 0.
        w.push(hb(1, TierId::L3(L3ModelId(0)), 1.0, true, 0.9));
        w.push(hb(1, TierId::L0, 1e-6, true, 0.9));
        assert!(d.check(&w).is_empty());
    }

    #[test]
    fn starvation_flags_absent_tier() {
        let mut d = StarvationDetector::new(vec![TierId::L0_1FactLut], 5);
        let w: Vec<_> = (0..6).map(|_| hb(1, TierId::L3(L3ModelId(0)), 2.0, true, 0.9)).collect();
        let p = d.check(&w);
        assert_eq!(p.len(), 1);
        assert!(matches!(p[0], Pathology::Starvation { .. }));
    }

    #[test]
    fn starvation_quiet_when_tier_present() {
        let mut d = StarvationDetector::new(vec![TierId::L0], 3);
        let w: Vec<_> = (0..5).map(|_| hb(1, TierId::L0, 1e-6, true, 0.9)).collect();
        assert!(d.check(&w).is_empty());
    }

    #[test]
    fn oscillation_detects_two_tier_flapping() {
        let mut d = OscillationDetector::default();
        let mut w = Vec::new();
        let tiers = [TierId::L0_25FormulaFirst, TierId::L1(L1Primitive::Retrieve)];
        for i in 0..8 {
            w.push(hb(42, tiers[i % 2], 1e-4, true, 0.8));
        }
        let p = d.check(&w);
        assert_eq!(p.len(), 1);
        match &p[0] {
            Pathology::Oscillation { query_fingerprint, flips, .. } => {
                assert_eq!(*query_fingerprint, 42);
                assert!(*flips >= 3);
            }
            _ => panic!("expected oscillation"),
        }
    }

    #[test]
    fn oscillation_ignores_three_tier_churn() {
        let mut d = OscillationDetector::default();
        let tiers = [
            TierId::L0,
            TierId::L0_25FormulaFirst,
            TierId::L1(L1Primitive::Retrieve),
        ];
        let mut w = Vec::new();
        for i in 0..9 {
            w.push(hb(7, tiers[i % 3], 1e-4, true, 0.8));
        }
        // Three distinct tiers → not a clean A↔B oscillation.
        assert!(d.check(&w).is_empty());
    }

    #[test]
    fn oracle_inversion_on_confident_failure() {
        let mut d = OracleInversionDetector::default();
        let w = vec![hb(3, TierId::L4_5Proof, 60e-6, false, 0.97)];
        let p = d.check(&w);
        assert_eq!(p.len(), 1);
        assert!(matches!(p[0], Pathology::OracleInversion { .. }));
    }

    #[test]
    fn oracle_inversion_ignores_low_confidence_failure() {
        let mut d = OracleInversionDetector::default();
        let w = vec![hb(3, TierId::L3(L3ModelId(0)), 2.0, false, 0.3)];
        assert!(d.check(&w).is_empty());
    }

    #[test]
    fn supervisor_scan_runs_all_detectors() {
        let mut s = Supervisor::with_default_detectors(32, vec![TierId::L0_1FactLut]);
        // Build a window that trips runaway + oracle inversion + starvation.
        for _ in 0..10 {
            s.record(hb(1, TierId::L0, 1e-6, true, 0.9));
        }
        for _ in 0..3 {
            s.record(hb(1, TierId::L3(L3ModelId(0)), 1.0, true, 0.9));
        }
        s.record(hb(1, TierId::L3(L3ModelId(0)), 1.0, false, 0.95)); // inversion
        let pathologies = s.scan();
        assert!(!pathologies.is_empty());
    }

    #[test]
    fn supervisor_window_bounded() {
        let mut s = Supervisor::new(3);
        for _ in 0..5 {
            s.record(hb(1, TierId::L0, 1e-6, true, 0.9));
        }
        assert_eq!(s.window_len(), 3);
    }

    #[test]
    fn pathology_serializes_with_wire_tag() {
        let p = Pathology::Starvation {
            tier: TierId::L0_1FactLut,
            idle_for: 5,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("L0.1"), "json = {json}");
    }

    #[test]
    fn detector_names() {
        assert_eq!(RunawayDetector::default().name(), "runaway");
        assert_eq!(OscillationDetector::default().name(), "oscillation");
    }
}
