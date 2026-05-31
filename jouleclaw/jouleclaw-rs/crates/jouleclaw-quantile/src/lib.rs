//! # jouleclaw-quantile
//!
//! Streaming quantile sketches with bounded *relative* error.
//!
//! DDSketch (Masson/Rim/Lee 2019, `arxiv:1908.10693`) is the
//! default: returns `q(x)` within `α · x` of the true quantile,
//! fully mergeable across shards, bucket count `≈ log(max/min) / α`.
//! Wave-4 SOTA brief: DDSketch has overtaken t-digest for new
//! observability infrastructure (Datadog, OpenTelemetry's
//! exponential-bucket histogram is structurally DDSketch); t-digest
//! is one-way mergeable and shows large p99.9 error on heavy-tailed
//! data (Cormode/Karnin `arxiv:2102.09299`).
//!
//! HDR (Gil Tene) is opt-in behind the `hdr` feature flag for
//! JVM-interop. Its bounded range is a footgun (off-range samples
//! silently truncate to the highest bucket — "the histogram lies,
//! then continues to lie"); the type's docs surface that warning
//! explicitly.
//!
//! ## Energy as the natural use case
//!
//! Per-call-site joule distribution is the canonical jouleclaw use:
//! one `DdsketchQuantile` per node id / per skill id, populated
//! from `RunEvent::NodeRecorded`. Outlier = sample > p99 + k·IQR
//! using only the sketch's quantile readouts, no full sample
//! retained.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(unexpected_cfgs)]

use jouleclaw_bounded::{Bounded, BoundedError};
use sketches_ddsketch::{Config, DDSketch};

/// Errors a quantile sketch can return.
#[derive(Debug, thiserror::Error)]
pub enum QuantileError {
    /// The sketch has zero observations.
    #[error("quantile sketch is empty")]
    Empty,
    /// Quantile must be in the open interval `(0, 1)`.
    #[error("invalid quantile {0}: must be in (0, 1)")]
    InvalidQuantile(f64),
    /// Backend (DDSketch) error.
    #[error("ddsketch: {0}")]
    Backend(String),
}

// ─────────────────────────────────────────────────────────────────────
// DDSketch wrapper — the default
// ─────────────────────────────────────────────────────────────────────

/// DDSketch — the default streaming quantile sketch with bounded
/// relative error.
pub struct DdsketchQuantile {
    sketch: DDSketch,
    alpha: f64,
    count: u64,
}

impl DdsketchQuantile {
    /// Build a sketch with relative-error parameter `alpha`. Clamped
    /// into `[1e-4, 0.5]` — both extremes are honest about what
    /// they cost.
    pub fn with_alpha(alpha: f64) -> Self {
        let alpha = alpha.max(1e-4).min(0.5);
        let cfg = Config::new(alpha, 2048, 1e-9);
        Self {
            sketch: DDSketch::new(cfg),
            alpha,
            count: 0,
        }
    }

    /// Sensible default — `alpha = 0.01` (1% relative error).
    pub fn default_alpha() -> Self {
        Self::with_alpha(0.01)
    }

    /// Add one sample.
    pub fn add(&mut self, value: f64) {
        self.sketch.add(value);
        self.count += 1;
    }

    /// Convenience: add many samples.
    pub fn extend<I: IntoIterator<Item = f64>>(&mut self, iter: I) {
        for v in iter {
            self.add(v);
        }
    }

    /// Number of observations added.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// `true` iff no observations have been added.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Estimated quantile at `q ∈ (0, 1)`. Returns `Err(Empty)`
    /// when no samples; the result is within `α · true_value`.
    pub fn quantile(&self, q: f64) -> Result<f64, QuantileError> {
        if self.is_empty() {
            return Err(QuantileError::Empty);
        }
        if !(q > 0.0 && q < 1.0) {
            return Err(QuantileError::InvalidQuantile(q));
        }
        match self.sketch.quantile(q) {
            Ok(Some(v)) => Ok(v),
            Ok(None) => Err(QuantileError::Empty),
            Err(e) => Err(QuantileError::Backend(format!("{e:?}"))),
        }
    }

    /// p50 convenience.
    pub fn p50(&self) -> Result<f64, QuantileError> {
        self.quantile(0.5)
    }
    /// p95 convenience.
    pub fn p95(&self) -> Result<f64, QuantileError> {
        self.quantile(0.95)
    }
    /// p99 convenience.
    pub fn p99(&self) -> Result<f64, QuantileError> {
        self.quantile(0.99)
    }
    /// p99.9 convenience.
    pub fn p999(&self) -> Result<f64, QuantileError> {
        self.quantile(0.999)
    }

    /// Tukey-fence based outlier threshold over the IQR window.
    /// Default `k = 1.5` (Tukey 1977 outer fence). `k = 3.0` for
    /// "extreme outlier."
    pub fn tukey_upper_fence(&self, k: f64) -> Result<f64, QuantileError> {
        let q1 = self.quantile(0.25)?;
        let q3 = self.quantile(0.75)?;
        let iqr = q3 - q1;
        Ok(q3 + k * iqr)
    }

    /// Merge another sketch into this one.
    pub fn merge(&mut self, other: &Self) -> Result<(), QuantileError> {
        self.sketch
            .merge(&other.sketch)
            .map_err(|e| QuantileError::Backend(format!("{e:?}")))?;
        self.count = self.count.saturating_add(other.count);
        Ok(())
    }
}

impl Bounded for DdsketchQuantile {
    /// `(ε, δ) = (α, 0)`. DDSketch is a deterministic sketch (no
    /// probabilistic failure); the bound is purely relative error.
    /// Memory ≈ bucket-count × 8 B; we report the upper bound at
    /// the configured 2048-bucket cap.
    fn bound(&self) -> BoundedError {
        BoundedError::relative(self.alpha, 0.0, 2048 * 8)
    }
}

// ─────────────────────────────────────────────────────────────────────
// HDR wrapper — opt-in for JVM interop
// ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "hdr")]
mod hdr_impl {
    use super::*;
    use hdrhistogram::Histogram;

    /// HDR-shaped sketch (Gil Tene). Opt-in via the `hdr` feature.
    ///
    /// **Footgun warning** (from the SOTA brief): HDR's bounded
    /// range silently truncates off-range samples to the highest
    /// bucket. DDSketch handles this; HDR does not. Use HDR only
    /// for JVM-interop where the peer also uses HDR.
    pub struct HdrQuantile {
        h: Histogram<u64>,
        sig_digits: u8,
    }

    impl HdrQuantile {
        /// Build with significant digits in `[1, 5]` — relative
        /// error ≤ `10^(-n)`.
        pub fn with_sig_digits(sig_digits: u8) -> Self {
            let n = sig_digits.clamp(1, 5);
            let h = Histogram::<u64>::new(n).expect("valid sig digits");
            Self { h, sig_digits: n }
        }

        /// Record one sample. Non-positive samples clamp to 1 (HDR
        /// is positive-only).
        pub fn add(&mut self, value: u64) {
            let v = value.max(1);
            let _ = self.h.record(v);
        }

        /// Estimated quantile.
        pub fn quantile(&self, q: f64) -> Result<u64, QuantileError> {
            if self.h.is_empty() {
                return Err(QuantileError::Empty);
            }
            if !(q > 0.0 && q < 1.0) {
                return Err(QuantileError::InvalidQuantile(q));
            }
            Ok(self.h.value_at_quantile(q))
        }
    }

    impl Bounded for HdrQuantile {
        fn bound(&self) -> BoundedError {
            let eps = 10f64.powi(-(self.sig_digits as i32));
            BoundedError::relative(eps, 0.0, 32 * 1024)
        }
    }
}

#[cfg(feature = "hdr")]
pub use hdr_impl::HdrQuantile;

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_sketch_errors_on_quantile() {
        let s = DdsketchQuantile::default_alpha();
        assert!(matches!(s.p50().unwrap_err(), QuantileError::Empty));
    }

    #[test]
    fn invalid_quantile_errors() {
        let mut s = DdsketchQuantile::default_alpha();
        s.add(1.0);
        assert!(matches!(s.quantile(0.0).unwrap_err(), QuantileError::InvalidQuantile(_)));
        assert!(matches!(s.quantile(1.0).unwrap_err(), QuantileError::InvalidQuantile(_)));
        assert!(matches!(s.quantile(-0.1).unwrap_err(), QuantileError::InvalidQuantile(_)));
    }

    #[test]
    fn quantiles_within_relative_error_for_uniform_distribution() {
        let mut s = DdsketchQuantile::with_alpha(0.01);
        for i in 1..=1000 {
            s.add(i as f64);
        }
        let p50 = s.p50().unwrap();
        assert!((p50 - 500.0).abs() <= 500.0 * 0.02 + 2.0, "p50={p50}");
        let p99 = s.p99().unwrap();
        assert!((p99 - 990.0).abs() <= 990.0 * 0.02 + 2.0, "p99={p99}");
    }

    #[test]
    fn count_tracks_observations() {
        let mut s = DdsketchQuantile::default_alpha();
        assert_eq!(s.count(), 0);
        s.extend([1.0, 2.0, 3.0]);
        assert_eq!(s.count(), 3);
    }

    #[test]
    fn tukey_upper_fence_above_q3() {
        let mut s = DdsketchQuantile::with_alpha(0.01);
        for i in 1..=1000 {
            s.add(i as f64);
        }
        let fence = s.tukey_upper_fence(1.5).unwrap();
        let q3 = s.quantile(0.75).unwrap();
        assert!(fence > q3, "fence={fence}, q3={q3}");
    }

    #[test]
    fn merge_combines_two_sketches() {
        let mut a = DdsketchQuantile::with_alpha(0.01);
        let mut b = DdsketchQuantile::with_alpha(0.01);
        for i in 1..=500 {
            a.add(i as f64);
            b.add(i as f64 + 500.0);
        }
        a.merge(&b).unwrap();
        assert_eq!(a.count(), 1000);
        let p50 = a.p50().unwrap();
        assert!((p50 - 500.0).abs() <= 500.0 * 0.02 + 2.0, "p50={p50}");
    }

    #[test]
    fn alpha_clamped_into_safe_range() {
        let tiny = DdsketchQuantile::with_alpha(0.0).alpha;
        assert!(tiny >= 1e-4);
        let huge = DdsketchQuantile::with_alpha(99.0).alpha;
        assert!(huge <= 0.5);
    }

    #[test]
    fn bound_reports_alpha_and_memory() {
        let s = DdsketchQuantile::with_alpha(0.01);
        let b = s.bound();
        assert_eq!(b.epsilon, 0.01);
        assert!(b.memory_bytes.unwrap() >= 1024);
    }

    #[test]
    fn convenience_percentile_methods_align_with_quantile() {
        let mut s = DdsketchQuantile::with_alpha(0.01);
        for i in 1..=1000 {
            s.add(i as f64);
        }
        assert_eq!(s.p50().unwrap(), s.quantile(0.5).unwrap());
        assert_eq!(s.p95().unwrap(), s.quantile(0.95).unwrap());
        assert_eq!(s.p99().unwrap(), s.quantile(0.99).unwrap());
        assert_eq!(s.p999().unwrap(), s.quantile(0.999).unwrap());
    }
}
