//! # jouleclaw-sketch
//!
//! Probabilistic sketches with proven `(ε, δ)` error bounds.
//!
//! The 2026 SOTA trio (wave-4 brief):
//!
//! - **Count-Min Sketch** (Cormode/Muthukrishnan 2005). Frequency
//!   estimator. `width = ⌈e/ε⌉`, `depth = ⌈ln(1/δ)⌉`. The estimate is
//!   within `ε · ||a||₁` with probability `≥ 1 − δ`. Always
//!   over-estimates for non-negative counts.
//! - **HyperLogLog++** (Heule 2013). Cardinality estimator with
//!   `~1.04/√m` standard error, bias-corrected sparse representation
//!   for small cardinalities, 64-bit hashing for `> 2³² uniques`.
//! - **Binary Fuse 8** (Graf/Lemire 2022). Static-set membership
//!   filter. FPR ≈ `2⁻⁸ ≈ 0.39 %`; ~13 % above the
//!   information-theoretic lower bound on storage; smaller and
//!   faster to construct than Xor or Cuckoo. **Replaces Bloom** for
//!   static use cases — the wave-4 brief's explicit cut.
//!
//! All three implement [`jouleclaw_bounded::Bounded`] so the
//! consumer can read the `(ε, δ, M)` triple uniformly.
//!
//! ## Honest scope
//!
//! - **Bloom dropped.** Use [`BinaryFuse8Filter`] for static membership;
//!   Bloom is only competitive for online-insert workloads and we
//!   defer that to a future `legacy::Bloom` if anyone asks.
//! - **CMS over-estimates by construction.** Never use it where
//!   false-negatives are tolerable but over-counts are not (e.g.
//!   billing).
//! - **BinaryFuse construction FAILS on duplicate keys.** This is
//!   the #1 production-crash cause when migrating from Bloom — the
//!   builder returns an error you MUST handle.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(unexpected_cfgs)]

use jouleclaw_bounded::{Bounded, BoundedError};
use std::collections::HashSet;
use std::hash::Hash;
use xorf::{BinaryFuse8, Filter};

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Errors a sketch construction can surface.
#[derive(Debug, thiserror::Error)]
pub enum SketchError {
    /// Binary Fuse construction needs distinct keys; duplicates
    /// abort. Caller MUST dedupe upstream.
    #[error("binary fuse construction failed (often duplicate keys): {0}")]
    BinaryFuseConstruction(String),
    /// CMS parameters yielded a degenerate sketch (0 width or depth).
    #[error("count-min parameters degenerate: width={width}, depth={depth}")]
    CountMinParams { width: usize, depth: usize },
}

// ─────────────────────────────────────────────────────────────────────
// Count-Min Sketch
// ─────────────────────────────────────────────────────────────────────

/// Count-Min Sketch — frequency estimator with `(ε, δ)` bound.
/// Always over-estimates for non-negative counts; use conservative
/// update (min-of-row) to tighten estimates further.
pub struct CountMin {
    /// `depth` rows × `width` columns of `u64` counters.
    rows: Vec<Vec<u64>>,
    /// Per-row seeds for the H(x) → column mapping.
    seeds: Vec<u64>,
    /// Total inserts (for `bound()` reporting).
    inserts: u64,
    epsilon: f64,
    delta: f64,
}

impl CountMin {
    /// Build a CMS sized for `(epsilon, delta)`:
    /// `width = ⌈e/ε⌉, depth = ⌈ln(1/δ)⌉`.
    pub fn with_params(epsilon: f64, delta: f64) -> Result<Self, SketchError> {
        let eps = epsilon.max(1e-6).min(1.0);
        let del = delta.max(1e-12).min(0.5);
        let width = (std::f64::consts::E / eps).ceil() as usize;
        let depth = ((1.0 / del).ln().ceil()) as usize;
        if width == 0 || depth == 0 {
            return Err(SketchError::CountMinParams { width, depth });
        }
        let seeds: Vec<u64> = (0..depth).map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
        let rows = vec![vec![0u64; width]; depth];
        Ok(Self {
            rows,
            seeds,
            inserts: 0,
            epsilon: eps,
            delta: del,
        })
    }

    fn col_of<T: Hash>(&self, item: &T, row: usize) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let mut h = DefaultHasher::new();
        h.write_u64(self.seeds[row]);
        item.hash(&mut h);
        let w = self.rows[row].len() as u64;
        (h.finish() % w) as usize
    }

    /// Increment the count for `item` by 1.
    pub fn add<T: Hash>(&mut self, item: &T) {
        self.add_n(item, 1);
    }

    /// Increment by `n`.
    pub fn add_n<T: Hash>(&mut self, item: &T, n: u64) {
        for r in 0..self.rows.len() {
            let c = self.col_of(item, r);
            self.rows[r][c] = self.rows[r][c].saturating_add(n);
        }
        self.inserts = self.inserts.saturating_add(n);
    }

    /// Estimate the count for `item`. Always `≥` the true count
    /// (CMS over-estimates).
    pub fn estimate<T: Hash>(&self, item: &T) -> u64 {
        (0..self.rows.len())
            .map(|r| self.rows[r][self.col_of(item, r)])
            .min()
            .unwrap_or(0)
    }

    /// Total inserts across the sketch.
    pub fn inserts(&self) -> u64 {
        self.inserts
    }
}

impl Bounded for CountMin {
    /// Reports `(ε, δ)` set at construction. Memory ≈
    /// `width × depth × 8`.
    fn bound(&self) -> BoundedError {
        let mem = (self.rows.len() * self.rows[0].len() * 8) as u64;
        BoundedError::relative(self.epsilon, self.delta, mem)
    }
}

// ─────────────────────────────────────────────────────────────────────
// HyperLogLog++ wrapper
// ─────────────────────────────────────────────────────────────────────

/// HyperLogLog++ cardinality estimator. Wraps
/// `cardinality_estimator::CardinalityEstimator`.
///
/// Standard error `≈ 1.04 / √m` where `m = 2ᵖ` registers.
pub struct CardinalitySketch {
    inner: cardinality_estimator::CardinalityEstimator<u64>,
    precision: u8,
}

impl CardinalitySketch {
    /// Build with default precision (12 → m=4096, σ ≈ 1.63%, ~3 KB
    /// memory). The `cardinality_estimator` crate uses a hybrid
    /// representation (exact for small, HLL++ for large) — the
    /// precision controls the HLL phase.
    pub fn new() -> Self {
        Self {
            inner: cardinality_estimator::CardinalityEstimator::<u64>::new(),
            precision: 12,
        }
    }

    /// Insert one item.
    pub fn insert<T: Hash>(&mut self, item: &T) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let mut h = DefaultHasher::new();
        item.hash(&mut h);
        let v = h.finish();
        self.inner.insert(&v);
    }

    /// Insert many items.
    pub fn extend<T: Hash, I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(&item);
        }
    }

    /// Estimated cardinality (number of distinct items).
    pub fn estimate(&self) -> usize {
        self.inner.estimate()
    }
}

impl Default for CardinalitySketch {
    fn default() -> Self {
        Self::new()
    }
}

impl Bounded for CardinalitySketch {
    /// Standard error `1.04 / √m` at the configured precision;
    /// memory ≈ `m × 6 bits ≈ m × 1` byte.
    fn bound(&self) -> BoundedError {
        let m = 1u64 << self.precision;
        let eps = 1.04 / (m as f64).sqrt();
        BoundedError::relative(eps, 0.0, m)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Binary Fuse 8 wrapper
// ─────────────────────────────────────────────────────────────────────

/// Binary Fuse 8 static-membership filter. **Builds from a known
/// key set** — no online insertion.
///
/// FPR ≈ 2⁻⁸ ≈ 0.39 %. Memory ≈ `1.13 × n` bytes for `n` keys —
/// ~13 % above the information-theoretic floor.
pub struct BinaryFuse8Filter {
    inner: BinaryFuse8,
    n_keys: u64,
}

impl BinaryFuse8Filter {
    /// Build from a set of `u64` keys. Caller is responsible for
    /// deduplicating; we additionally dedupe via `HashSet` for
    /// safety (the construction otherwise fails on duplicate keys —
    /// the #1 production migration footgun).
    pub fn build(keys: &[u64]) -> Result<Self, SketchError> {
        let dedup: Vec<u64> = keys.iter().copied().collect::<HashSet<_>>().into_iter().collect();
        let inner = BinaryFuse8::try_from(&dedup)
            .map_err(|e| SketchError::BinaryFuseConstruction(e.to_string()))?;
        Ok(Self {
            inner,
            n_keys: dedup.len() as u64,
        })
    }

    /// Query: is `key` in the filter? May produce false positives
    /// with rate ≈ 2⁻⁸.
    pub fn contains(&self, key: u64) -> bool {
        self.inner.contains(&key)
    }

    /// Number of keys the filter was built from (post-dedup).
    pub fn len(&self) -> u64 {
        self.n_keys
    }

    /// `true` iff the filter was built from an empty set.
    pub fn is_empty(&self) -> bool {
        self.n_keys == 0
    }
}

impl Bounded for BinaryFuse8Filter {
    /// FPR ≈ 2⁻⁸. Memory ≈ 1.13 × n_keys bytes.
    fn bound(&self) -> BoundedError {
        let mem = ((self.n_keys as f64) * 1.13).ceil() as u64;
        BoundedError::relative(2f64.powi(-8), 0.0, mem)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CountMin ─────────────────────────────────────────────────

    #[test]
    fn count_min_estimate_is_at_least_true_count() {
        let mut cms = CountMin::with_params(0.001, 0.001).unwrap();
        for _ in 0..100 {
            cms.add(&"a");
        }
        for _ in 0..50 {
            cms.add(&"b");
        }
        // CMS over-estimates by construction.
        assert!(cms.estimate(&"a") >= 100);
        assert!(cms.estimate(&"b") >= 50);
    }

    #[test]
    fn count_min_bound_reports_epsilon_delta_memory() {
        let cms = CountMin::with_params(0.01, 0.001).unwrap();
        let b = cms.bound();
        assert_eq!(b.epsilon, 0.01);
        assert!((b.delta - 0.001).abs() < 1e-9);
        assert!(b.memory_bytes.unwrap() > 0);
    }

    #[test]
    fn count_min_inserts_track_total() {
        let mut cms = CountMin::with_params(0.01, 0.01).unwrap();
        cms.add_n(&"x", 100);
        cms.add(&"y");
        assert_eq!(cms.inserts(), 101);
    }

    // ── CardinalitySketch ────────────────────────────────────────

    #[test]
    fn cardinality_estimate_within_expected_error_for_small_sets() {
        let mut hll = CardinalitySketch::new();
        for i in 0..100u64 {
            hll.insert(&i);
        }
        let est = hll.estimate();
        // For small sets (cardinality_estimator uses exact phase
        // until precision threshold), should be exact or very close.
        assert!(est >= 95 && est <= 110, "got {est}");
    }

    #[test]
    fn cardinality_distinguishes_duplicates() {
        let mut hll = CardinalitySketch::new();
        for _ in 0..1000 {
            hll.insert(&42u64);
        }
        let est = hll.estimate();
        assert!(est <= 2, "got {est}");
    }

    #[test]
    fn cardinality_bound_reports_standard_error_and_memory() {
        let hll = CardinalitySketch::new();
        let b = hll.bound();
        // 1.04 / sqrt(2^12) ≈ 0.01625
        assert!((b.epsilon - 0.01625).abs() < 0.001, "epsilon={}", b.epsilon);
        assert_eq!(b.memory_bytes, Some(4096));
    }

    // ── BinaryFuse8Filter ────────────────────────────────────────

    #[test]
    fn binary_fuse_contains_all_inserted_keys() {
        let keys: Vec<u64> = (1..=1_000).collect();
        let f = BinaryFuse8Filter::build(&keys).unwrap();
        for k in &keys {
            assert!(f.contains(*k), "missing key {k}");
        }
    }

    #[test]
    fn binary_fuse_false_positive_rate_around_one_in_256() {
        let keys: Vec<u64> = (0..1_000).collect();
        let f = BinaryFuse8Filter::build(&keys).unwrap();
        // Query 100k keys NOT in the set; FPR ≈ 1/256 ≈ 0.39%.
        let mut hits = 0;
        for k in 1_000_000u64..1_100_000 {
            if f.contains(k) {
                hits += 1;
            }
        }
        let fpr = hits as f64 / 100_000.0;
        // Expected ~0.39%; allow 0.0% to 1.0% (CI noise).
        assert!(fpr < 0.012, "fpr={fpr}");
    }

    #[test]
    fn binary_fuse_handles_duplicate_keys_via_internal_dedup() {
        let keys: Vec<u64> = vec![1, 2, 3, 2, 1, 4];
        let f = BinaryFuse8Filter::build(&keys).unwrap();
        // After dedupe → 4 keys.
        assert_eq!(f.len(), 4);
        for k in [1u64, 2, 3, 4] {
            assert!(f.contains(k));
        }
    }

    #[test]
    fn binary_fuse_bound_reports_fpr_and_memory_estimate() {
        let keys: Vec<u64> = (0..1000).collect();
        let f = BinaryFuse8Filter::build(&keys).unwrap();
        let b = f.bound();
        // 2^-8 = 0.00390625
        assert!((b.epsilon - 0.00390625).abs() < 1e-6);
        // ~1.13 * 1000 = 1130 bytes (we ceil it).
        assert!(b.memory_bytes.unwrap() >= 1000);
    }

    #[test]
    fn binary_fuse_empty_filter_constructs() {
        let f = BinaryFuse8Filter::build(&[]).unwrap();
        assert!(f.is_empty());
    }
}
