//! `MatryoshkaEmbedder` — wraps an `Embedder` with a dim ladder and a
//! quality model so the cascade can pick a truncation point that meets
//! the query's quality floor under its retrieval budget.
//!
//! The MRL property the wrapper assumes: for an MRL-trained model, any
//! prefix of length `d ≤ full_dim` is itself a usable embedding for
//! downstream retrieval / classification, with a *monotonic*, gentle
//! quality drop as `d` decreases. This wrapper does not enforce that
//! property — it relies on the underlying model having been trained
//! with the MRL objective. For a non-MRL model, the wrapper still works
//! but the quality at smaller dims will degrade much faster than the
//! quality model predicts.

use crate::embedder::{Embedder, EmbedderError};

/// Maps a truncation dim to a predicted quality score in [0, 1].
///
/// The default model is a log-shaped curve calibrated against the
/// Kusupati et al. published numbers on ImageNet: at full dim quality =
/// 1.0; at full_dim / 32 quality ≈ 0.95; at full_dim / 256 quality ≈ 0.85.
/// Production should plug in a measured curve.
#[derive(Debug, Clone)]
pub enum QualityModel {
    /// `quality(d) = 1 - α * ln(full_dim / d)` clamped to [0, 1].
    LogDecay { alpha: f32, full_dim: usize },
    /// Direct lookup: each `(dim, quality)` pair gives a measured point;
    /// queries at other dims are linearly interpolated. Must be sorted
    /// by `dim` ascending.
    Measured(Vec<(usize, f32)>),
}

impl QualityModel {
    /// Default calibration: α=0.022 gives ~0.95 quality at 1/32 of full,
    /// matching the rough shape of the MRL ImageNet results.
    pub fn default_for(full_dim: usize) -> Self {
        Self::LogDecay { alpha: 0.022, full_dim }
    }

    pub fn at(&self, dim: usize) -> f32 {
        match self {
            Self::LogDecay { alpha, full_dim } => {
                if dim == 0 {
                    return 0.0;
                }
                if dim >= *full_dim {
                    return 1.0;
                }
                let ratio = (*full_dim as f32) / (dim as f32);
                let q = 1.0 - alpha * ratio.ln();
                q.clamp(0.0, 1.0)
            }
            Self::Measured(points) => {
                if points.is_empty() {
                    return 0.0;
                }
                // Clamp to endpoints.
                if dim <= points[0].0 {
                    return points[0].1;
                }
                if dim >= points[points.len() - 1].0 {
                    return points[points.len() - 1].1;
                }
                // Linear interpolation between bracketing points.
                for w in points.windows(2) {
                    let (d0, q0) = w[0];
                    let (d1, q1) = w[1];
                    if dim >= d0 && dim <= d1 {
                        let t = (dim - d0) as f32 / (d1 - d0) as f32;
                        return q0 + t * (q1 - q0);
                    }
                }
                points[points.len() - 1].1
            }
        }
    }
}

/// Wrapper around any `Embedder` with MRL-compatible truncation.
pub struct MatryoshkaEmbedder<E: Embedder> {
    embedder: E,
    /// Sorted ascending. Must be a subset of [1, embedder.full_dim()].
    dims: Vec<usize>,
    /// Quality predictor.
    pub quality: QualityModel,
    /// Energy cost of one inner-product against a single doc at dim 1.
    /// Total retrieval cost = `doc_count * dim * retrieval_pj_per_op * 1e-12`.
    /// Default 1 pJ per multiply-add — calibrated against a CPU baseline.
    pub retrieval_pj_per_op: f64,
}

impl<E: Embedder> MatryoshkaEmbedder<E> {
    /// Build with the default power-of-2 dim ladder: {1, 2, 4, 8, …, full_dim}.
    pub fn with_powers_of_two(embedder: E) -> Self {
        let full = embedder.full_dim();
        let mut dims = Vec::new();
        let mut d = 1usize;
        while d <= full {
            dims.push(d);
            d *= 2;
        }
        if *dims.last().unwrap_or(&0) != full {
            dims.push(full);
        }
        let quality = QualityModel::default_for(full);
        Self { embedder, dims, quality, retrieval_pj_per_op: 1.0 }
    }

    /// Build with a custom dim ladder. Sorts ascending; deduplicates.
    /// Any dim > full_dim() is clamped to full_dim(); 0 is removed.
    pub fn with_dims(embedder: E, mut dims: Vec<usize>) -> Self {
        let full = embedder.full_dim();
        dims.retain(|&d| d != 0);
        for d in dims.iter_mut() {
            if *d > full {
                *d = full;
            }
        }
        dims.sort_unstable();
        dims.dedup();
        let quality = QualityModel::default_for(full);
        Self { embedder, dims, quality, retrieval_pj_per_op: 1.0 }
    }

    pub fn full_dim(&self) -> usize {
        self.embedder.full_dim()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Embed at full dim. Pass-through to the underlying embedder.
    pub fn embed(&self, input: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        self.embedder.embed(input)
    }

    /// Embed at dim `d` — computes the full embedding then truncates the
    /// prefix. `d` must be one of `self.dims()`.
    pub fn embed_at_dim(&self, input: &[f32], d: usize) -> Result<Vec<f32>, EmbedderError> {
        if !self.dims.contains(&d) {
            return Err(EmbedderError::Other(format!(
                "dim {} not in ladder {:?}",
                d, self.dims
            )));
        }
        let mut v = self.embedder.embed(input)?;
        v.truncate(d);
        Ok(v)
    }

    /// Cost of one embedding forward pass. The truncation is free; cost
    /// is whatever the underlying embedder reports for the full pass.
    pub fn embed_joules(&self) -> f64 {
        self.embedder.embed_joules()
    }

    /// Cost of one inner-product against a single doc at dimension `d`.
    pub fn retrieval_joules_per_doc(&self, d: usize) -> f64 {
        (d as f64) * self.retrieval_pj_per_op * 1e-12
    }

    /// Total cost of nearest-neighbor search at dim `d` over `n` docs.
    pub fn retrieval_joules(&self, d: usize, n_docs: usize) -> f64 {
        self.retrieval_joules_per_doc(d) * (n_docs as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::IdentityEmbedder;

    #[test]
    fn default_dim_ladder_is_powers_of_two_through_full() {
        let m = MatryoshkaEmbedder::with_powers_of_two(IdentityEmbedder::new(16));
        assert_eq!(m.dims(), &[1, 2, 4, 8, 16]);
    }

    #[test]
    fn embed_at_dim_yields_prefix_of_full_embed() {
        let m = MatryoshkaEmbedder::with_powers_of_two(IdentityEmbedder::new(8));
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let full = m.embed(&input).unwrap();
        let trunc4 = m.embed_at_dim(&input, 4).unwrap();
        assert_eq!(trunc4.len(), 4);
        assert_eq!(&full[..4], &trunc4[..]);
    }

    #[test]
    fn embed_at_dim_rejects_off_ladder() {
        let m = MatryoshkaEmbedder::with_powers_of_two(IdentityEmbedder::new(8));
        // 3 is not in {1, 2, 4, 8}.
        assert!(matches!(
            m.embed_at_dim(&vec![0.0; 8], 3),
            Err(EmbedderError::Other(_))
        ));
    }

    #[test]
    fn custom_dim_ladder_is_sorted_and_clamped() {
        let m = MatryoshkaEmbedder::with_dims(
            IdentityEmbedder::new(16),
            vec![32, 8, 0, 4, 8, 100],
        );
        assert_eq!(m.dims(), &[4, 8, 16]);
    }

    #[test]
    fn quality_model_is_monotone_non_decreasing() {
        let q = QualityModel::default_for(2048);
        let dims = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];
        let mut prev = -1.0_f32;
        for d in dims {
            let v = q.at(d);
            assert!(v >= prev, "quality({}) = {} not ≥ {} (prev)", d, v, prev);
            assert!(v >= 0.0 && v <= 1.0);
            prev = v;
        }
        assert!(q.at(2048) >= 0.999);
    }

    #[test]
    fn quality_model_lookup_interpolation_matches_measured_points() {
        let q = QualityModel::Measured(vec![(8, 0.5), (64, 0.8), (512, 1.0)]);
        assert_eq!(q.at(8), 0.5);
        assert_eq!(q.at(64), 0.8);
        assert_eq!(q.at(512), 1.0);
        // Linear interpolation between 8 and 64: at d=36, t ≈ (36-8)/(64-8) = 0.5
        let interp = q.at(36);
        assert!((interp - 0.65).abs() < 0.02);
    }

    #[test]
    fn retrieval_cost_scales_linearly_with_dim() {
        let m = MatryoshkaEmbedder::with_powers_of_two(IdentityEmbedder::new(512));
        let n_docs = 1_000_000;
        let c_512 = m.retrieval_joules(512, n_docs);
        let c_64 = m.retrieval_joules(64, n_docs);
        let ratio = c_512 / c_64;
        assert!((ratio - 8.0).abs() < 1e-6, "expected 8× ratio, got {}", ratio);
    }
}
