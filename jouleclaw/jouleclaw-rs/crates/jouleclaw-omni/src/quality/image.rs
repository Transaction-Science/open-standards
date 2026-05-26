//! Image quality metrics for diffusion / video / 3D render outputs.
//!
//! Operates on CHW `&[f32]` data (planar: [R..., G..., B...]).
//! Expected value range: [-0.5, 0.5] (pre-clamp) or [0.0, 1.0] (post-normalize).

use super::stats::{self, TensorStats};

/// Quality report for a generated image.
#[derive(Debug, Clone)]
pub struct ImageReport {
    /// Per-channel (R, G, B) statistics
    pub channel_stats: [TensorStats; 3],
    /// Percentage of pixels at saturation bounds (0.0 or 1.0 after normalization)
    pub saturation_pct: f32,
    /// Shannon entropy across spatial pixel intensities (bits)
    pub spatial_entropy: f32,
    /// Laplacian sharpness: mean absolute second derivative
    pub laplacian_sharpness: f32,
    /// True if all pixels are approximately the same color
    pub is_uniform: bool,
    /// True if data is finite and within reasonable range
    pub is_valid: bool,
    /// Width of the image
    pub width: usize,
    /// Height of the image
    pub height: usize,
}

impl ImageReport {
    /// Generate human-readable warnings.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if !self.is_valid {
            w.push("Image contains NaN or Inf values".into());
        }
        if self.is_uniform {
            w.push("Image is uniform (all pixels same color)".into());
        }
        if self.saturation_pct > 10.0 {
            w.push(format!("High saturation: {:.1}% of pixels at bounds", self.saturation_pct));
        }
        if self.spatial_entropy < 1.0 {
            w.push(format!("Very low entropy: {:.2} bits (possible noise or flat image)", self.spatial_entropy));
        }
        if self.laplacian_sharpness < 0.001 && !self.is_uniform {
            w.push("Very low sharpness (blurry or flat gradient)".into());
        }
        w
    }

    /// One-line summary.
    pub fn summary(&self) -> String {
        let status = if self.is_valid && !self.is_uniform { "PASS" } else { "WARN" };
        format!(
            "[{status}] {}x{} entropy={:.1}b sharp={:.3} sat={:.1}%",
            self.width, self.height, self.spatial_entropy, self.laplacian_sharpness, self.saturation_pct
        )
    }
}

/// Compute image quality metrics from CHW planar float data.
///
/// `data` layout: `[R_0, R_1, ..., R_{h*w-1}, G_0, ..., G_{h*w-1}, B_0, ..., B_{h*w-1}]`
///
/// Values are assumed in range [-0.5, 0.5] or [0.0, 1.0]. The function normalizes
/// to [0, 1] for metric computation.
pub fn image_report(data: &[f32], c: usize, h: usize, w: usize) -> ImageReport {
    let area = h * w;
    let expected = c * area;

    // Fallback stats for missing/empty data
    let empty_stats = || TensorStats {
        mean: 0.0, std: 0.0, min: 0.0, max: 0.0,
        nan_count: 0, inf_count: 0, numel: 0,
    };

    if data.len() < expected || c < 3 || area == 0 {
        return ImageReport {
            channel_stats: [empty_stats(), empty_stats(), empty_stats()],
            saturation_pct: 0.0,
            spatial_entropy: 0.0,
            laplacian_sharpness: 0.0,
            is_uniform: true,
            is_valid: false,
            width: w,
            height: h,
        };
    }

    // Extract per-channel data
    let ch_r = &data[0..area];
    let ch_g = &data[area..2 * area];
    let ch_b = &data[2 * area..3 * area];

    let stats_r = stats::stats(ch_r).unwrap_or_else(empty_stats);
    let stats_g = stats::stats(ch_g).unwrap_or_else(empty_stats);
    let stats_b = stats::stats(ch_b).unwrap_or_else(empty_stats);

    // Check for NaN/Inf in any channel
    let total_bad = stats_r.nan_count + stats_r.inf_count
        + stats_g.nan_count + stats_g.inf_count
        + stats_b.nan_count + stats_b.inf_count;
    let is_valid = total_bad == 0;

    // Detect value range: if min < -0.1, values are in [-0.5, 0.5] range
    let global_min = stats_r.min.min(stats_g.min).min(stats_b.min);
    let offset = if global_min < -0.1 { 0.5 } else { 0.0 };

    // Saturation: count pixels near 0 or 1 (after offset normalization)
    let sat_threshold = 0.01;
    let mut saturated = 0usize;
    for &v in data.iter().take(expected) {
        if !v.is_finite() { continue; }
        let normalized = v + offset;
        if normalized < sat_threshold || normalized > (1.0 - sat_threshold) {
            saturated += 1;
        }
    }
    let saturation_pct = (saturated as f32 / expected as f32) * 100.0;

    // Spatial entropy: compute luminance histogram
    // Always push a value per pixel to maintain h×w grid for Laplacian
    let mut luma = Vec::with_capacity(area);
    for i in 0..area {
        let r = ch_r.get(i).copied().unwrap_or(0.0);
        let g = ch_g.get(i).copied().unwrap_or(0.0);
        let b = ch_b.get(i).copied().unwrap_or(0.0);
        if r.is_finite() && g.is_finite() && b.is_finite() {
            luma.push((r + offset) * 0.299 + (g + offset) * 0.587 + (b + offset) * 0.114);
        } else {
            luma.push(f32::NAN);
        }
    }
    let spatial_entropy = stats::entropy(&luma, 256);

    // Laplacian sharpness: 3x3 discrete Laplacian on luminance
    let laplacian_sharpness = if h >= 3 && w >= 3 {
        laplacian_mean_abs(&luma, h, w)
    } else {
        0.0
    };

    // Uniform check: all channels have very low std
    let uniform_threshold = 0.01;
    let is_uniform = stats_r.std < uniform_threshold
        && stats_g.std < uniform_threshold
        && stats_b.std < uniform_threshold;

    ImageReport {
        channel_stats: [stats_r, stats_g, stats_b],
        saturation_pct,
        spatial_entropy,
        laplacian_sharpness,
        is_uniform,
        is_valid,
        width: w,
        height: h,
    }
}

/// Mean absolute Laplacian over a 2D grayscale image.
///
/// Kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
fn laplacian_mean_abs(data: &[f32], h: usize, w: usize) -> f32 {
    if h < 3 || w < 3 {
        return 0.0;
    }

    let mut sum = 0.0_f64;
    let mut count = 0usize;

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let center = data[y * w + x];
            let top = data[(y - 1) * w + x];
            let bottom = data[(y + 1) * w + x];
            let left = data[y * w + (x - 1)];
            let right = data[y * w + (x + 1)];

            if center.is_finite() && top.is_finite() && bottom.is_finite()
                && left.is_finite() && right.is_finite()
            {
                let lap = top + bottom + left + right - 4.0 * center;
                sum += lap.abs() as f64;
                count += 1;
            }
        }
    }

    if count > 0 { (sum / count as f64) as f32 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform(h: usize, w: usize, val: f32) -> Vec<f32> {
        vec![val; 3 * h * w]
    }

    fn make_gradient(h: usize, w: usize) -> Vec<f32> {
        let area = h * w;
        let mut data = vec![0.0f32; 3 * area];
        for y in 0..h {
            for x in 0..w {
                let t = (x as f32) / (w as f32);
                data[y * w + x] = t;             // R = horizontal gradient
                data[area + y * w + x] = 0.5;    // G = mid gray
                data[2 * area + y * w + x] = 1.0 - t; // B = inverse gradient
            }
        }
        data
    }

    #[test]
    fn all_black_detected() {
        let data = make_uniform(64, 64, 0.0);
        let report = image_report(&data, 3, 64, 64);
        assert!(report.is_uniform, "all-black should be uniform");
        assert!(report.spatial_entropy < 0.1, "all-black should have ~0 entropy");
    }

    #[test]
    fn all_white_detected() {
        let data = make_uniform(64, 64, 1.0);
        let report = image_report(&data, 3, 64, 64);
        assert!(report.is_uniform, "all-white should be uniform");
    }

    #[test]
    fn gradient_is_valid() {
        let data = make_gradient(64, 64);
        let report = image_report(&data, 3, 64, 64);
        assert!(!report.is_uniform, "gradient should not be uniform");
        assert!(report.is_valid, "gradient should be valid");
        assert!(report.spatial_entropy > 3.0, "gradient should have decent entropy, got {}", report.spatial_entropy);
        assert!(report.laplacian_sharpness < 0.01, "smooth gradient should have low laplacian, got {}", report.laplacian_sharpness);
    }

    #[test]
    fn nan_image_detected() {
        let mut data = make_gradient(16, 16);
        data[0] = f32::NAN;
        data[100] = f32::NAN;
        let report = image_report(&data, 3, 16, 16);
        assert!(!report.is_valid, "NaN image should not be valid");
    }

    #[test]
    fn high_saturation_detected() {
        let area = 64 * 64;
        let mut data = vec![0.0f32; 3 * area];
        // Set 50% of R channel to 1.0, rest to 0.0 — high saturation
        for i in 0..area {
            data[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
            data[area + i] = 0.5;
            data[2 * area + i] = 0.5;
        }
        let report = image_report(&data, 3, 64, 64);
        // R channel has 50% at bounds, G and B have 0% → overall ~1/3 * 50% ≈ 17%
        assert!(report.saturation_pct > 15.0, "should detect high saturation, got {}", report.saturation_pct);
    }
}
