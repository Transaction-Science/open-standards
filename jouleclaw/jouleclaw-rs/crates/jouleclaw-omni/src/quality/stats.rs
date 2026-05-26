//! Core statistical functions for output quality validation.
//!
//! All functions operate on `&[f32]` slices — no Tensor dependency.
//! Pure Rust, zero external dependencies.

/// Basic statistics for a numeric dataset.
#[derive(Debug, Clone)]
pub struct TensorStats {
    /// Arithmetic mean
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Minimum value (excluding NaN)
    pub min: f32,
    /// Maximum value (excluding NaN)
    pub max: f32,
    /// Count of NaN values
    pub nan_count: usize,
    /// Count of Inf values
    pub inf_count: usize,
    /// Total number of elements
    pub numel: usize,
}

impl TensorStats {
    /// Whether the data contains no NaN or Inf values.
    pub fn is_finite(&self) -> bool {
        self.nan_count == 0 && self.inf_count == 0
    }

    /// Dynamic range (max - min).
    pub fn range(&self) -> f32 {
        self.max - self.min
    }
}

/// Compute basic statistics for a float slice.
///
/// Returns `None` if data is empty.
pub fn stats(data: &[f32]) -> Option<TensorStats> {
    if data.is_empty() {
        return None;
    }

    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut finite_count = 0usize;

    for &v in data {
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        if v.is_infinite() {
            inf_count += 1;
            continue;
        }
        sum += v as f64;
        sum_sq += (v as f64) * (v as f64);
        if v < min { min = v; }
        if v > max { max = v; }
        finite_count += 1;
    }

    let mean = if finite_count > 0 {
        (sum / finite_count as f64) as f32
    } else {
        0.0
    };

    let variance = if finite_count > 1 {
        let n = finite_count as f64;
        ((sum_sq - sum * sum / n) / (n - 1.0)).max(0.0)
    } else {
        0.0
    };

    let std = (variance as f32).sqrt();

    // If no finite values, set min/max to 0
    if finite_count == 0 {
        min = 0.0;
        max = 0.0;
    }

    Some(TensorStats {
        mean,
        std,
        min,
        max,
        nan_count,
        inf_count,
        numel: data.len(),
    })
}

/// Compute a uniform-width histogram.
///
/// Returns `bins` counts. Values outside [min, max] are clamped to edge bins.
/// Returns empty vec if data is empty or all values are NaN/Inf.
pub fn histogram(data: &[f32], bins: usize) -> Vec<u32> {
    if bins == 0 || data.is_empty() {
        return vec![0; bins];
    }

    let s = match stats(data) {
        Some(s) if s.range() > 0.0 => s,
        Some(s) => {
            // All values equal — everything in first bin
            let mut h = vec![0u32; bins];
            h[0] = (s.numel - s.nan_count - s.inf_count) as u32;
            return h;
        }
        None => return vec![0; bins],
    };

    let mut h = vec![0u32; bins];
    let range = s.range();

    for &v in data {
        if !v.is_finite() {
            continue;
        }
        let normalized = (v - s.min) / range;
        let bin = ((normalized * bins as f32) as usize).min(bins - 1);
        h[bin] += 1;
    }

    h
}

/// Compute Shannon entropy from a histogram (in bits).
///
/// Higher entropy = more uniform distribution.
/// Lower entropy = concentrated in few bins.
pub fn entropy_from_histogram(hist: &[u32]) -> f32 {
    let total: u32 = hist.iter().sum();
    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;
    let mut entropy = 0.0_f64;

    for &count in hist {
        if count == 0 {
            continue;
        }
        let p = count as f64 / total_f;
        entropy -= p * p.log2();
    }

    entropy as f32
}

/// Compute Shannon entropy of data using a given number of bins.
pub fn entropy(data: &[f32], bins: usize) -> f32 {
    let hist = histogram(data, bins);
    entropy_from_histogram(&hist)
}

/// Compute an approximate percentile (0.0 to 1.0).
///
/// Uses linear interpolation on a sorted copy.
/// Returns 0.0 if data is empty or all non-finite.
pub fn percentile(data: &[f32], p: f32) -> f32 {
    let p = p.clamp(0.0, 1.0);

    let mut finite: Vec<f32> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return 0.0;
    }

    finite.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = p * (finite.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(finite.len() - 1);
    let frac = idx - lo as f32;

    finite[lo] * (1.0 - frac) + finite[hi] * frac
}

/// Fraction of distinct values in the dataset.
///
/// Uses a tolerance for floating-point comparison.
/// Returns 0.0 if data is empty.
pub fn unique_ratio(data: &[f32], tolerance: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let mut finite: Vec<f32> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return 0.0;
    }

    finite.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut unique = 1usize;
    for i in 1..finite.len() {
        if (finite[i] - finite[i - 1]).abs() > tolerance {
            unique += 1;
        }
    }

    unique as f32 / finite.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let s = stats(&data).unwrap();
        assert!((s.mean - 3.0).abs() < 1e-5);
        assert!((s.min - 1.0).abs() < 1e-5);
        assert!((s.max - 5.0).abs() < 1e-5);
        assert_eq!(s.nan_count, 0);
        assert_eq!(s.inf_count, 0);
        assert_eq!(s.numel, 5);
    }

    #[test]
    fn stats_with_nan() {
        let data = [1.0, f32::NAN, 3.0];
        let s = stats(&data).unwrap();
        assert_eq!(s.nan_count, 1);
        assert!((s.mean - 2.0).abs() < 1e-5);
    }

    #[test]
    fn entropy_uniform() {
        // 256 distinct values → max entropy for 256 bins
        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let e = entropy(&data, 256);
        assert!(e > 7.9, "uniform distribution should have ~8 bits entropy, got {e}");
    }

    #[test]
    fn entropy_constant() {
        let data = vec![42.0; 1000];
        let e = entropy(&data, 256);
        assert!(e < 0.01, "constant data should have ~0 entropy, got {e}");
    }

    #[test]
    fn percentile_median() {
        let data: Vec<f32> = (0..101).map(|i| i as f32).collect();
        let p50 = percentile(&data, 0.5);
        assert!((p50 - 50.0).abs() < 0.5, "median should be ~50, got {p50}");
    }

    #[test]
    fn unique_ratio_all_same() {
        let data = vec![1.0; 100];
        let r = unique_ratio(&data, 1e-6);
        assert!((r - 0.01).abs() < 0.02, "all same should have ~1/100 ratio, got {r}");
    }

    #[test]
    fn unique_ratio_all_different() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let r = unique_ratio(&data, 1e-6);
        assert!((r - 1.0).abs() < 1e-5, "all different should have ratio 1.0, got {r}");
    }
}
