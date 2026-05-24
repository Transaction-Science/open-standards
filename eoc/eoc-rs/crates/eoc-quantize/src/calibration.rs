//! Calibration helpers — per-channel and per-tensor scale derivation.
//!
//! Post-training quantization needs to know the dynamic range of each
//! tensor (or each channel) under realistic input. This module
//! provides:
//!
//! * `CalibrationDataset` — a tiny container of fp32 activation
//!   batches with `push_batch` for streaming collection.
//! * `per_tensor_absmax` / `per_channel_absmax` — the workhorse
//!   summary statistics.
//! * `percentile_clip` — drop the top-`p` magnitudes before computing
//!   the scale (classic outlier suppression).

use crate::error::QuantError;

/// Streaming calibration dataset. Batches are stored densely to keep
/// the implementation focused; production code should sketch instead.
#[derive(Debug, Default, Clone)]
pub struct CalibrationDataset {
    /// Number of feature channels per row (set on first push).
    pub channels: usize,
    /// Flat row-major storage: `rows * channels` floats.
    pub data: Vec<f32>,
}

impl CalibrationDataset {
    /// Empty dataset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a batch of shape `(rows, channels)` in row-major order.
    pub fn push_batch(&mut self, rows: usize, channels: usize, batch: &[f32]) -> Result<(), QuantError> {
        if batch.len() != rows * channels {
            return Err(QuantError::BadGroupSize {
                len: batch.len(),
                group: rows * channels,
            });
        }
        if self.channels == 0 {
            self.channels = channels;
        } else if self.channels != channels {
            return Err(QuantError::InvalidFormat("calibration channel mismatch"));
        }
        self.data.extend_from_slice(batch);
        Ok(())
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.data.len() / self.channels
        }
    }
}

/// Per-tensor absmax (single scalar across all values).
pub fn per_tensor_absmax(x: &[f32]) -> f32 {
    x.iter().copied().fold(0.0_f32, |a, v| a.max(v.abs()))
}

/// Per-channel absmax across a row-major `(rows, channels)` tensor.
pub fn per_channel_absmax(x: &[f32], channels: usize) -> Vec<f32> {
    if channels == 0 || x.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0_f32; channels];
    for row in x.chunks_exact(channels) {
        for (c, v) in row.iter().enumerate() {
            let av = v.abs();
            if av > out[c] {
                out[c] = av;
            }
        }
    }
    out
}

/// Per-channel absmax clipped at the `percentile`-th value (0..1).
///
/// `percentile = 0.99` keeps the 99th percentile and drops the top 1%
/// as outliers — common practice for activation calibration.
pub fn per_channel_percentile(x: &[f32], channels: usize, percentile: f32) -> Vec<f32> {
    if channels == 0 || x.is_empty() {
        return Vec::new();
    }
    let p = percentile.clamp(0.0, 1.0);
    let rows = x.len() / channels;
    let mut out = Vec::with_capacity(channels);
    let mut col = Vec::with_capacity(rows);
    for c in 0..channels {
        col.clear();
        for r in 0..rows {
            col.push(x[r * channels + c].abs());
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let idx = ((rows as f32) * p).floor() as usize;
        let clamped = idx.min(rows.saturating_sub(1));
        out.push(col.get(clamped).copied().unwrap_or(0.0));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_accumulates_rows() {
        let mut d = CalibrationDataset::new();
        d.push_batch(2, 4, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("push");
        d.push_batch(1, 4, &[9.0, 10.0, 11.0, 12.0]).expect("push2");
        assert_eq!(d.rows(), 3);
        assert_eq!(d.channels, 4);
    }

    #[test]
    fn per_channel_absmax_picks_largest_per_column() {
        let x = vec![1.0_f32, -5.0, 2.0, 0.5, 3.0, -2.0];
        let m = per_channel_absmax(&x, 3);
        assert_eq!(m, vec![1.0, 5.0, 2.0]);
    }

    #[test]
    fn percentile_clip_trims_outliers() {
        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let m = per_channel_percentile(&x, 1, 0.95);
        assert!(m[0] < 100.0);
        assert!(m[0] >= 90.0);
    }
}
