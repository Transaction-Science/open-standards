//! # jouleclaw-bounded
//!
//! Doctrine traits the wave-4 SOTA brief identified as load-bearing
//! across every verification + error-tracking primitive in
//! JouleClaw. Three shapes the field has converged on, lifted out
//! of the individual primitives so they compose:
//!
//! ## Pattern A — `(ε, δ)` parameterisation
//!
//! Every approximating primitive carries a **bounded error**:
//! relative tolerance `ε` (the result is within `ε · true` with
//! probability `1 − δ`), at memory cost `M`. DDSketch's `α`, CMS's
//! `(ε, δ)`, HLL's standard error `1.04/√m`, ADWIN's Hoeffding
//! `ε_cut`, Page-Hinkley's `λ` — all are the same shape. The
//! [`BoundedError`] type is the uniform introspection contract.
//!
//! ## Pattern B — Fast-path scalar + tamper-evident chain
//!
//! Every primitive splits into a **cheap online tier** (per-call,
//! lossy, fast) and a **strong tier** (per-batch, proof-grade).
//! CRC32C + blake3 is the canonical pair; cheap-detector +
//! tail-aware-detector for drift, libtest + Kani for verification,
//! Tukey fences + Isolation Forest for outliers. The
//! [`FastStrong`] trait makes both reachable from the same caller.
//!
//! ## Pattern C — Score + explanation, never Boolean
//!
//! Detectors without ground truth (cognitive complexity, drift,
//! outliers) return a *rank* and a *reason*, not a label. The
//! [`Scored`] trait makes discarding the explanation
//! syntactically awkward — you have to destructure it.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(unexpected_cfgs)]

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────
// Pattern A — BoundedError
// ─────────────────────────────────────────────────────────────────────

/// `(ε, δ)`-bounded approximation parameter set. The result is
/// within `epsilon · true_value` (or `± epsilon` when absolute) of
/// the true value with probability at least `1 − delta`, at memory
/// cost `memory_bytes`.
///
/// The triple `(ε, δ, M)` is the only honest way to compare
/// probabilistic sketches; quoting one without the other two is
/// the canonical advertising failure mode the SOTA brief flagged.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundedError {
    /// Relative or absolute tolerance — see `kind`.
    pub epsilon: f64,
    /// Failure probability. `0.0` for deterministic primitives;
    /// `[0.0, 1.0]` for probabilistic.
    pub delta: f64,
    /// Memory cost of the primitive's working state, bytes.
    /// `None` if the primitive does not have a meaningful upper
    /// bound (e.g. unbounded histograms).
    pub memory_bytes: Option<u64>,
    /// Whether `epsilon` is a relative or absolute bound.
    pub kind: BoundKind,
}

/// Whether the bound is relative or absolute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundKind {
    /// `epsilon` is multiplicative — result within `ε · |true|`.
    /// DDSketch, HLL, CMS use this.
    Relative,
    /// `epsilon` is additive — result within `±epsilon`.
    /// HDR (within its bounded range), some custom estimators.
    Absolute,
}

impl BoundedError {
    /// Build a relative bound `(ε, δ)` at `memory_bytes` cost.
    pub fn relative(epsilon: f64, delta: f64, memory_bytes: u64) -> Self {
        Self {
            epsilon,
            delta,
            memory_bytes: Some(memory_bytes),
            kind: BoundKind::Relative,
        }
    }

    /// Build an absolute bound `±ε` at `memory_bytes` cost.
    pub fn absolute(epsilon: f64, delta: f64, memory_bytes: u64) -> Self {
        Self {
            epsilon,
            delta,
            memory_bytes: Some(memory_bytes),
            kind: BoundKind::Absolute,
        }
    }

    /// Build a deterministic bound: `epsilon = 0.0`, `delta = 0.0`
    /// (no probabilistic failure mode).
    pub fn exact(memory_bytes: u64) -> Self {
        Self {
            epsilon: 0.0,
            delta: 0.0,
            memory_bytes: Some(memory_bytes),
            kind: BoundKind::Absolute,
        }
    }
}

/// Anything that approximates with a bound carries this. One method,
/// pure read.
pub trait Bounded {
    /// The current bound. May be parameter-dependent (e.g. CMS
    /// constructed with `width=w, depth=d` reports `(ε=e/w, δ=2^-d)`).
    fn bound(&self) -> BoundedError;
}

// ─────────────────────────────────────────────────────────────────────
// Pattern B — FastStrong (two-tier evidence)
// ─────────────────────────────────────────────────────────────────────

/// Two-tier evidence: a cheap `fast` reading per call, a stronger
/// `strong` reading per batch / checkpoint. The two methods MUST
/// return evidence the consumer can compare — e.g. a checksum
/// (fast) and a hash (strong) over the same bytes.
///
/// The contract: `fast(x)` is monotone with respect to `strong(x)`
/// — if `fast` says "ok," `strong` should not say "broken." A
/// `strong(x)` divergence from `fast(x)` is an escalation signal
/// (re-verify, re-checksum, refuse). Implementations that violate
/// monotonicity are liars and the trait makes the constraint
/// explicit.
pub trait FastStrong {
    /// The evidence type emitted by both tiers. Must implement
    /// `PartialEq` so the consumer can detect divergence.
    type Evidence: PartialEq + Clone;
    /// The input the primitive ingests.
    type Input: ?Sized;

    /// Cheap online evidence per call. SHOULD complete in O(input
    /// size) or smaller; SHOULD be safe to call in a hot loop.
    fn fast(&self, input: &Self::Input) -> Self::Evidence;

    /// Strong evidence per batch / per checkpoint. May be more
    /// expensive; the consumer chooses when to escalate.
    fn strong(&self, input: &Self::Input) -> Self::Evidence;

    /// Return `true` iff the two tiers agree on `input`. Useful as
    /// an asserts-in-prod escape hatch when both are cheap to
    /// compute on a sample.
    fn agree(&self, input: &Self::Input) -> bool {
        self.fast(input) == self.strong(input)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Pattern C — Scored (no-ground-truth detectors)
// ─────────────────────────────────────────────────────────────────────

/// A score with an attached explanation. Detectors without ground
/// truth (cognitive complexity, drift, outlier) return this rather
/// than a bool — converting it to a binary requires destructuring,
/// which is the point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Score<E: Clone + PartialEq> {
    /// Numerical rank. Higher = more notable. Range is detector-
    /// defined; consumers SHOULD compare ranks within a single
    /// detector, NOT across detectors.
    pub value: f64,
    /// Structured reason — what features deviated, what neighbours
    /// were considered, what threshold was crossed. Detector-defined
    /// type so the explanation can be precise.
    pub explanation: E,
    /// The detector that produced this score. Stable string id for
    /// cross-detector audit.
    pub detector: String,
}

impl<E: Clone + PartialEq> Score<E> {
    /// Convenience constructor.
    pub fn new(value: f64, explanation: E, detector: impl Into<String>) -> Self {
        Self {
            value,
            explanation,
            detector: detector.into(),
        }
    }

    /// Convert to a binary label using an explicit threshold. This
    /// is intentionally a method on `Score` (not a `From` impl) —
    /// you have to call it deliberately, with a threshold in scope.
    pub fn binarise(&self, threshold: f64) -> bool {
        self.value > threshold
    }
}

/// Anything that scores carries this. The associated type pins the
/// explanation shape.
pub trait Scored {
    /// The input being scored.
    type Input: ?Sized;
    /// The detector-specific explanation type.
    type Explanation: Clone + PartialEq;

    /// Score one input. Returns a [`Score`] the consumer
    /// destructures.
    fn score(&self, input: &Self::Input) -> Score<Self::Explanation>;
}

// ─────────────────────────────────────────────────────────────────────
// Kani proof harness
// ─────────────────────────────────────────────────────────────────────

/// `binarise` is monotone in `threshold` — lowering the threshold
/// can only flip the label from `false` to `true`, never the other
/// way.
#[cfg(kani)]
#[kani::proof]
fn kani_binarise_monotone_in_threshold() {
    let v: f64 = kani::any();
    let t_lo: f64 = kani::any();
    let t_hi: f64 = kani::any();
    kani::assume(t_lo <= t_hi);
    let s = Score::new(v, (), "test");
    let lo = s.binarise(t_lo);
    let hi = s.binarise(t_hi);
    // hi=true ⇒ lo=true (lowering threshold can only add positives)
    kani::assert(!hi || lo, "binarise monotone in threshold");
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyDdsketch {
        alpha: f64,
        memory: u64,
    }

    impl Bounded for DummyDdsketch {
        fn bound(&self) -> BoundedError {
            BoundedError::relative(self.alpha, 0.0, self.memory)
        }
    }

    #[test]
    fn bounded_relative_carries_epsilon_and_memory() {
        let s = DummyDdsketch {
            alpha: 0.01,
            memory: 3200 * 16,
        };
        let b = s.bound();
        assert_eq!(b.epsilon, 0.01);
        assert_eq!(b.kind, BoundKind::Relative);
        assert_eq!(b.memory_bytes, Some(3200 * 16));
    }

    #[test]
    fn bounded_exact_is_zero_zero() {
        let b = BoundedError::exact(64);
        assert_eq!(b.epsilon, 0.0);
        assert_eq!(b.delta, 0.0);
        assert_eq!(b.kind, BoundKind::Absolute);
    }

    #[test]
    fn bounded_error_round_trips_through_json() {
        let b = BoundedError::relative(0.01, 1e-6, 1024);
        let j = serde_json::to_value(&b).unwrap();
        assert_eq!(j["kind"], "relative");
        let back: BoundedError = serde_json::from_value(j).unwrap();
        assert_eq!(back, b);
    }

    struct FastEqStrongHasher;
    impl FastStrong for FastEqStrongHasher {
        type Evidence = String;
        type Input = [u8];
        fn fast(&self, input: &[u8]) -> String {
            format!("len={}-first={:02x}", input.len(), input.first().unwrap_or(&0))
        }
        fn strong(&self, input: &[u8]) -> String {
            format!("len={}-first={:02x}", input.len(), input.first().unwrap_or(&0))
        }
    }

    #[test]
    fn fast_strong_agree_returns_true_when_evidence_matches() {
        let p = FastEqStrongHasher;
        assert!(p.agree(b"hello"));
        assert_eq!(p.fast(b"hello"), p.strong(b"hello"));
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct OutlierExplanation {
        nearest_neighbour_distance: f64,
        features_deviated: Vec<String>,
    }

    struct DummyOutlierDetector;
    impl Scored for DummyOutlierDetector {
        type Input = [f64];
        type Explanation = OutlierExplanation;
        fn score(&self, input: &[f64]) -> Score<OutlierExplanation> {
            Score::new(
                input.iter().sum::<f64>().abs(),
                OutlierExplanation {
                    nearest_neighbour_distance: 3.14,
                    features_deviated: vec!["joules_uj".into()],
                },
                "dummy-outlier",
            )
        }
    }

    #[test]
    fn scored_returns_score_with_structured_explanation() {
        let d = DummyOutlierDetector;
        let s = d.score(&[1.0, 2.0, 3.0]);
        assert_eq!(s.value, 6.0);
        assert_eq!(s.detector, "dummy-outlier");
        assert_eq!(s.explanation.features_deviated, vec!["joules_uj"]);
    }

    #[test]
    fn binarise_requires_explicit_threshold() {
        let d = DummyOutlierDetector;
        let s = d.score(&[1.0, 2.0, 3.0]);
        assert!(s.binarise(5.0));
        assert!(!s.binarise(7.0));
    }

    #[test]
    fn binarise_monotone_in_threshold() {
        let s = Score::new(10.0, (), "t");
        let labels: Vec<bool> = [5.0, 9.0, 10.0, 15.0]
            .iter()
            .map(|t| s.binarise(*t))
            .collect();
        for w in labels.windows(2) {
            assert!(w[1] as u8 <= w[0] as u8);
        }
    }

    #[test]
    fn score_round_trips_through_json() {
        let s = Score::new(
            1.23,
            OutlierExplanation {
                nearest_neighbour_distance: 0.5,
                features_deviated: vec!["a".into(), "b".into()],
            },
            "test",
        );
        let bytes = serde_json::to_vec(&s).unwrap();
        let back: Score<OutlierExplanation> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, s);
    }
}
