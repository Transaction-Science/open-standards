//! # jouleclaw-outlier
//!
//! Geometric outlier detection — the database-anomaly-finding
//! primitive.
//!
//! Two detectors per wave-4 SOTA brief, picked from "low-D + clean
//! → Tukey/IQR; high-D + structured → Isolation Forest":
//!
//! - **Tukey fences** (Tukey 1977). Quartile-based, distribution-
//!   free. Outlier if `x < Q1 − k·IQR` or `x > Q3 + k·IQR` with
//!   `k = 1.5` (outer fence) / `k = 3.0` (extreme). One-dimensional.
//!   The cheap default that gets 90% of low-D use cases right.
//! - **Isolation Forest** (Liu/Ting/Zhou 2008). Tree-ensemble that
//!   builds random splits over the data; outliers are isolated in
//!   *shallow* paths. Anomaly score `s(x, n) = 2^{−E(h(x))/c(n)}`
//!   where `E(h(x))` is the mean path length and `c(n)` is the
//!   normalization (average path length of an unsuccessful BST
//!   search). Robust to high dimensions and concept drift.
//!
//! Both return [`jouleclaw_bounded::Score`] with structured
//! explanations, never bare booleans (Pattern C — no ground truth
//! ⇒ no labels).
//!
//! ## Honest scope
//!
//! - **LOF dropped** per the SOTA brief — degrades to `O(n²)` in
//!   high dimensions (curse of dimensionality).
//! - **Isolation Forest's `contamination` is NOT a false-positive
//!   rate.** It's a prior on the expected anomaly fraction. We
//!   default to score-based ranking, not contamination-based
//!   binary labels — caller calls `Score::binarise(threshold)`
//!   with a threshold they own.
//! - **`rstar`/`kiddo`-style spatial indexes** for k-NN are NOT
//!   wired here in v1 — Isolation Forest is robust enough to
//!   high-D that a k-NN structure is unnecessary for the
//!   primary use case (jouleclaw observations are typically
//!   `≤ 8` dimensions).

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(unexpected_cfgs)]

use jouleclaw_bounded::{Bounded, BoundedError, Score, Scored};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────
// Shared explanation type
// ─────────────────────────────────────────────────────────────────────

/// Structured explanation for an outlier score. Every detector
/// emits this so consumers can render a uniform UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutlierExplanation {
    /// The features (dimension indices or names) that deviated.
    pub features_deviated: Vec<String>,
    /// Distance / depth / fence multiplier — detector-specific
    /// numeric. Documented per-detector.
    pub deviation_metric: f64,
    /// The threshold used by the detector (e.g. Tukey k=1.5; IF
    /// score cut). Surfaced so the explanation is reproducible.
    pub threshold: f64,
    /// Short, human-readable summary the consumer renders in a
    /// dashboard.
    pub summary: String,
}

// ─────────────────────────────────────────────────────────────────────
// Tukey fences
// ─────────────────────────────────────────────────────────────────────

/// 1-D Tukey-fence outlier detector. Pre-fit with a sample
/// distribution to derive Q1/Q3/IQR; then `score(x)` reports how
/// far outside the fence `x` lies.
#[derive(Debug, Clone)]
pub struct TukeyFence {
    q1: f64,
    q3: f64,
    iqr: f64,
    /// Multiplier on IQR for the fence — 1.5 outer, 3.0 extreme.
    pub k: f64,
}

impl TukeyFence {
    /// Build from a sample slice. Sorts internally — input is not
    /// mutated. Picks the conventional Q1=p25, Q3=p75; for empty
    /// samples both quartiles are 0 (fence is trivially around 0).
    pub fn from_samples(samples: &[f64], k: f64) -> Self {
        let mut s = samples.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q1 = quantile(&s, 0.25);
        let q3 = quantile(&s, 0.75);
        Self {
            q1,
            q3,
            iqr: q3 - q1,
            k: k.max(0.0),
        }
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

impl Scored for TukeyFence {
    type Input = f64;
    type Explanation = OutlierExplanation;
    fn score(&self, x: &f64) -> Score<OutlierExplanation> {
        let upper = self.q3 + self.k * self.iqr;
        let lower = self.q1 - self.k * self.iqr;
        let deviation = if *x > upper {
            (*x - upper).abs()
        } else if *x < lower {
            (lower - *x).abs()
        } else {
            0.0
        };
        // Score = deviation / IQR (unitless, scale-aware). Higher =
        // more anomalous; 0 = within fence.
        let score = if self.iqr > 0.0 {
            deviation / self.iqr
        } else {
            deviation
        };
        let summary = if score > 0.0 {
            format!(
                "x={x:.3} outside Tukey fence [{lower:.3}, {upper:.3}] (k={k})",
                x = x, lower = lower, upper = upper, k = self.k
            )
        } else {
            format!("x={x:.3} within fence", x = x)
        };
        Score::new(
            score,
            OutlierExplanation {
                features_deviated: if score > 0.0 { vec!["x".into()] } else { vec![] },
                deviation_metric: deviation,
                threshold: self.k,
                summary,
            },
            "tukey-fence",
        )
    }
}

impl Bounded for TukeyFence {
    /// Tukey fences are deterministic given the fit sample — no
    /// probabilistic error. We report `(ε=0, δ=0)` and a small
    /// fixed memory footprint.
    fn bound(&self) -> BoundedError {
        BoundedError::exact(64)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Isolation Forest (own impl, ~150 LOC)
// ─────────────────────────────────────────────────────────────────────

/// Isolation Forest (Liu 2008). Anomaly detection via random
/// trees built over `n_dims`-dimensional samples; the score is
/// the expected isolation depth — outliers isolate fast in
/// shallow paths.
pub struct IsolationForest {
    trees: Vec<IsolationTree>,
    sample_size: usize,
}

struct IsolationTree {
    root: IsoNode,
}

enum IsoNode {
    External {
        size: usize,
    },
    Internal {
        feature: usize,
        split: f64,
        left: Box<IsoNode>,
        right: Box<IsoNode>,
    },
}

impl IsolationForest {
    /// Build an isolation forest from `samples` (each sample is
    /// `n_dims` long). `n_trees` controls ensemble size; default
    /// 100 per the paper. `sample_size = ψ` is the sub-sample size
    /// per tree; default 256 (the paper's recommendation).
    /// `seed` makes construction deterministic.
    pub fn fit(samples: &[Vec<f64>], n_trees: usize, sample_size: usize, seed: u64) -> Self {
        let n_trees = n_trees.max(1);
        let sample_size = sample_size.max(2).min(samples.len().max(2));
        let height_limit = ((sample_size as f64).log2().ceil()) as usize;
        let mut rng = LcgRng::new(seed);
        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            let mut sub: Vec<&Vec<f64>> = Vec::with_capacity(sample_size);
            for _ in 0..sample_size {
                let i = (rng.next_u64() as usize) % samples.len().max(1);
                sub.push(&samples[i.min(samples.len().saturating_sub(1))]);
            }
            let root = Self::build_tree(&sub, 0, height_limit, &mut rng);
            trees.push(IsolationTree { root });
        }
        Self { trees, sample_size }
    }

    fn build_tree(
        samples: &[&Vec<f64>],
        depth: usize,
        height_limit: usize,
        rng: &mut LcgRng,
    ) -> IsoNode {
        if depth >= height_limit || samples.len() <= 1 {
            return IsoNode::External { size: samples.len() };
        }
        let n_dims = samples[0].len();
        if n_dims == 0 {
            return IsoNode::External { size: samples.len() };
        }
        let feature = (rng.next_u64() as usize) % n_dims;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for s in samples {
            let v = s[feature];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        if !lo.is_finite() || hi <= lo {
            return IsoNode::External { size: samples.len() };
        }
        let split = lo + (rng.next_f64() * (hi - lo));
        let mut left = Vec::new();
        let mut right = Vec::new();
        for s in samples {
            if s[feature] < split {
                left.push(*s);
            } else {
                right.push(*s);
            }
        }
        if left.is_empty() || right.is_empty() {
            return IsoNode::External { size: samples.len() };
        }
        IsoNode::Internal {
            feature,
            split,
            left: Box::new(Self::build_tree(&left, depth + 1, height_limit, rng)),
            right: Box::new(Self::build_tree(&right, depth + 1, height_limit, rng)),
        }
    }

    fn path_length(node: &IsoNode, x: &[f64], depth: usize) -> f64 {
        match node {
            IsoNode::External { size } => depth as f64 + c_factor(*size),
            IsoNode::Internal {
                feature,
                split,
                left,
                right,
            } => {
                if x[*feature] < *split {
                    Self::path_length(left, x, depth + 1)
                } else {
                    Self::path_length(right, x, depth + 1)
                }
            }
        }
    }

    /// Raw anomaly score `s(x, ψ) = 2^{−E(h(x))/c(ψ)}` in `[0, 1]`.
    /// `s ≈ 0.5` ⇒ ambiguous; `s → 1` ⇒ likely anomaly; `s ≪ 0.5`
    /// ⇒ likely normal.
    pub fn anomaly_score(&self, x: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.5;
        }
        let mean_h: f64 = self
            .trees
            .iter()
            .map(|t| Self::path_length(&t.root, x, 0))
            .sum::<f64>()
            / self.trees.len() as f64;
        let c = c_factor(self.sample_size);
        if c <= 0.0 {
            return 0.5;
        }
        2f64.powf(-mean_h / c)
    }
}

/// Average path length of an unsuccessful BST search over `n`
/// points (the normalization c(n) from the IF paper).
fn c_factor(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let n_f = n as f64;
    2.0 * ((n_f - 1.0).ln() + 0.5772156649) - (2.0 * (n_f - 1.0) / n_f)
}

impl Scored for IsolationForest {
    type Input = [f64];
    type Explanation = OutlierExplanation;
    fn score(&self, x: &[f64]) -> Score<OutlierExplanation> {
        let s = self.anomaly_score(x);
        let summary = format!(
            "isolation-forest score={s:.3} (≥0.6 typically flagged; 0.5 ambiguous)",
            s = s
        );
        Score::new(
            s,
            OutlierExplanation {
                features_deviated: (0..x.len()).map(|i| format!("dim_{i}")).collect(),
                deviation_metric: s,
                threshold: 0.6,
                summary,
            },
            "isolation-forest",
        )
    }
}

impl Bounded for IsolationForest {
    /// Isolation Forest has no closed-form `(ε, δ)` guarantee. We
    /// report a coarse approximation: `ε ≈ 1/√n_trees` (standard
    /// error of the score from ensemble averaging). Memory ≈ tree
    /// count × node-count.
    fn bound(&self) -> BoundedError {
        let eps = if self.trees.is_empty() {
            1.0
        } else {
            1.0 / (self.trees.len() as f64).sqrt()
        };
        let mem = (self.trees.len() * self.sample_size * 32) as u64;
        BoundedError::relative(eps, 0.0, mem)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Deterministic small RNG — no rand dep
// ─────────────────────────────────────────────────────────────────────

/// Linear-congruential PRNG. NOT cryptographic; used for
/// deterministic tree construction with a caller-supplied seed.
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        // Avoid the all-zeros fixed point.
        let s = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        Self { state: s }
    }
    fn next_u64(&mut self) -> u64 {
        // Knuth's MMIX LCG constants.
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tukey fence ──────────────────────────────────────────────

    #[test]
    fn tukey_fence_returns_zero_score_inside_fence() {
        let samples: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let t = TukeyFence::from_samples(&samples, 1.5);
        let s = t.score(&50.0);
        assert_eq!(s.value, 0.0);
        assert!(s.explanation.features_deviated.is_empty());
        assert_eq!(s.detector, "tukey-fence");
    }

    #[test]
    fn tukey_fence_returns_positive_score_outside_fence() {
        let samples: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let t = TukeyFence::from_samples(&samples, 1.5);
        let s = t.score(&500.0);
        assert!(s.value > 0.0);
        assert!(!s.explanation.features_deviated.is_empty());
    }

    #[test]
    fn tukey_extreme_k_widens_fence() {
        let samples: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let outer = TukeyFence::from_samples(&samples, 1.5);
        let extreme = TukeyFence::from_samples(&samples, 3.0);
        // Same data; k=3 is more permissive than k=1.5 — a point
        // borderline at k=1.5 may be inside at k=3.
        let s_outer = outer.score(&180.0);
        let s_extreme = extreme.score(&180.0);
        assert!(s_outer.value >= s_extreme.value);
    }

    // ── Isolation Forest ─────────────────────────────────────────

    fn cluster_data() -> Vec<Vec<f64>> {
        // Tight cluster around (0,0) with 200 points + 3 obvious
        // outliers.
        let mut v = Vec::with_capacity(203);
        for i in 0..200 {
            let x = ((i % 20) as f64 - 10.0) / 100.0;
            let y = ((i / 20) as f64 - 5.0) / 100.0;
            v.push(vec![x, y]);
        }
        v
    }

    #[test]
    fn isolation_forest_scores_inliers_low_and_outliers_high() {
        let data = cluster_data();
        let forest = IsolationForest::fit(&data, 100, 256, 42);
        let inlier_score = forest.anomaly_score(&[0.0, 0.0]);
        let outlier_score = forest.anomaly_score(&[100.0, 100.0]);
        assert!(outlier_score > inlier_score, "outlier={outlier_score}, inlier={inlier_score}");
    }

    #[test]
    fn isolation_forest_score_in_unit_interval() {
        let data = cluster_data();
        let forest = IsolationForest::fit(&data, 50, 64, 7);
        for p in [vec![0.0, 0.0], vec![1.0, 1.0], vec![10.0, 10.0]] {
            let s = forest.anomaly_score(&p);
            assert!(s >= 0.0 && s <= 1.0, "score {s} out of [0,1]");
        }
    }

    #[test]
    fn isolation_forest_scored_emits_structured_explanation() {
        let data = cluster_data();
        let forest = IsolationForest::fit(&data, 50, 64, 7);
        let s = forest.score(&[100.0, 100.0]);
        assert_eq!(s.detector, "isolation-forest");
        assert_eq!(s.explanation.features_deviated.len(), 2);
        assert!(!s.explanation.summary.is_empty());
    }

    #[test]
    fn isolation_forest_is_deterministic_under_fixed_seed() {
        let data = cluster_data();
        let a = IsolationForest::fit(&data, 50, 64, 99);
        let b = IsolationForest::fit(&data, 50, 64, 99);
        let sa = a.anomaly_score(&[50.0, 50.0]);
        let sb = b.anomaly_score(&[50.0, 50.0]);
        assert_eq!(sa, sb);
    }

    #[test]
    fn isolation_forest_bound_reports_ensemble_sqrt_n_error() {
        let data = cluster_data();
        let f = IsolationForest::fit(&data, 100, 256, 1);
        let b = f.bound();
        // 1/sqrt(100) = 0.1
        assert!((b.epsilon - 0.1).abs() < 1e-6);
    }

    #[test]
    fn tukey_bound_is_exact() {
        let samples: Vec<f64> = vec![1.0, 2.0, 3.0];
        let t = TukeyFence::from_samples(&samples, 1.5);
        let b = t.bound();
        assert_eq!(b.epsilon, 0.0);
        assert_eq!(b.delta, 0.0);
    }

    #[test]
    fn outlier_explanation_round_trips_through_json() {
        let e = OutlierExplanation {
            features_deviated: vec!["joules".into()],
            deviation_metric: 3.14,
            threshold: 1.5,
            summary: "x outside fence".into(),
        };
        let j = serde_json::to_value(&e).unwrap();
        let back: OutlierExplanation = serde_json::from_value(j).unwrap();
        assert_eq!(back, e);
    }

    // ── c_factor sanity ──────────────────────────────────────────

    #[test]
    fn c_factor_zero_for_singleton() {
        assert_eq!(c_factor(0), 0.0);
        assert_eq!(c_factor(1), 0.0);
    }

    #[test]
    fn c_factor_monotone_in_n() {
        let a = c_factor(10);
        let b = c_factor(100);
        let c = c_factor(1000);
        assert!(a < b && b < c);
    }
}
