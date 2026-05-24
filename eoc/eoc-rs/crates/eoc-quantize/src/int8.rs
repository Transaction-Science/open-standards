//! Int8 post-training quantization — symmetric and asymmetric.
//!
//! Symmetric:
//!     q = round(x / scale)               , scale = max(|x|) / 127
//!     x' = q * scale
//!
//! Asymmetric:
//!     scale = (max - min) / 255
//!     zp    = round(-min / scale)
//!     q     = clamp(round(x/scale) + zp, 0, 255)
//!     x'    = (q - zp) * scale
//!
//! References: Jacob et al. 2018, Krishnamoorthi 2018, LLM.int8()
//! (Dettmers et al. 2022) for the outlier-aware mixed-precision case.

use crate::error::QuantError;
use crate::scheme::{Numeric, QuantizationScheme};

/// Symmetric per-tensor int8 quantizer (signed: [-127, 127]).
#[derive(Debug, Clone)]
pub struct Int8Symmetric;

/// Encoded symmetric tensor.
#[derive(Debug, Clone)]
pub struct Int8SymEncoded {
    /// Per-tensor scale: x ≈ q * scale.
    pub scale: f32,
    /// Quantized values in [-127, 127].
    pub q: Vec<i8>,
}

impl Int8Symmetric {
    /// Construct.
    pub fn new() -> Self {
        Self
    }

    /// Quantize, returning an error only for an empty input.
    pub fn try_quantize(&self, x: &[f32]) -> Result<Int8SymEncoded, QuantError> {
        if x.is_empty() {
            return Err(QuantError::EmptyInput);
        }
        let amax = x
            .iter()
            .copied()
            .fold(0.0_f32, |acc, v| acc.max(v.abs()))
            .max(f32::EPSILON);
        let scale = amax / 127.0;
        let q = x
            .iter()
            .map(|&v| {
                let r = (v / scale).round();
                r.clamp(-127.0, 127.0) as i8
            })
            .collect();
        Ok(Int8SymEncoded { scale, q })
    }
}

impl Default for Int8Symmetric {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizationScheme for Int8Symmetric {
    type Encoded = Int8SymEncoded;

    fn numeric(&self) -> Numeric {
        Numeric::Int8
    }

    fn quantize(&self, weights: &[f32]) -> Self::Encoded {
        self.try_quantize(weights).unwrap_or(Int8SymEncoded {
            scale: 1.0,
            q: Vec::new(),
        })
    }

    fn dequantize(&self, encoded: &Self::Encoded) -> Vec<f32> {
        encoded.q.iter().map(|&q| q as f32 * encoded.scale).collect()
    }
}

/// Asymmetric per-tensor int8 quantizer (unsigned domain, stored u8).
#[derive(Debug, Clone)]
pub struct Int8Asymmetric;

/// Encoded asymmetric tensor.
#[derive(Debug, Clone)]
pub struct Int8AsymEncoded {
    /// Quantization scale.
    pub scale: f32,
    /// Zero point (u8 storage domain).
    pub zero_point: u8,
    /// Quantized values [0, 255].
    pub q: Vec<u8>,
}

impl Int8Asymmetric {
    /// Construct.
    pub fn new() -> Self {
        Self
    }

    /// Quantize, returning an error only for an empty input.
    ///
    /// Uses the "include zero" affine convention: storage domain is
    /// `[0, 255]`, scale spans the (possibly extended) `[min, max]`
    /// range, and the zero point is the integer that maps to fp `0.0`.
    /// We extend the range to include zero so the affine map is
    /// representable end-to-end.
    pub fn try_quantize(&self, x: &[f32]) -> Result<Int8AsymEncoded, QuantError> {
        if x.is_empty() {
            return Err(QuantError::EmptyInput);
        }
        let (mn, mx) = x
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
                (a.min(v), b.max(v))
            });
        // Extend the observed range to include zero so the affine
        // q = round((x - mn)/scale) map covers fp 0 exactly.
        let mn_ext = mn.min(0.0);
        let mx_ext = mx.max(0.0);
        let span = (mx_ext - mn_ext).max(f32::EPSILON);
        let scale = span / 255.0;
        let zp_f = (-mn_ext / scale).round().clamp(0.0, 255.0);
        let zero_point = zp_f as u8;
        let q = x
            .iter()
            .map(|&v| {
                let r = ((v - mn_ext) / scale).round();
                r.clamp(0.0, 255.0) as u8
            })
            .collect();
        Ok(Int8AsymEncoded {
            scale,
            zero_point,
            q,
        })
    }
}

impl Default for Int8Asymmetric {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizationScheme for Int8Asymmetric {
    type Encoded = Int8AsymEncoded;

    fn numeric(&self) -> Numeric {
        Numeric::Int8
    }

    fn quantize(&self, weights: &[f32]) -> Self::Encoded {
        self.try_quantize(weights).unwrap_or(Int8AsymEncoded {
            scale: 1.0,
            zero_point: 0,
            q: Vec::new(),
        })
    }

    fn dequantize(&self, encoded: &Self::Encoded) -> Vec<f32> {
        let zp = encoded.zero_point as f32;
        encoded
            .q
            .iter()
            .map(|&q| (q as f32 - zp) * encoded.scale)
            .collect()
    }
}

/// LLM.int8()-style outlier-aware mixed precision. Weights with
/// absolute magnitude above `outlier_threshold` are retained in fp16;
/// the rest go to int8 symmetric.
///
/// Returns the int8 encoding plus the fp16-retained outlier indices.
pub fn mixed_precision(
    x: &[f32],
    outlier_threshold: f32,
) -> Result<(Int8SymEncoded, Vec<(usize, f32)>), QuantError> {
    if x.is_empty() {
        return Err(QuantError::EmptyInput);
    }
    let mut outliers = Vec::new();
    let mut inliers = Vec::with_capacity(x.len());
    for (i, &v) in x.iter().enumerate() {
        if v.abs() >= outlier_threshold {
            outliers.push((i, v));
            inliers.push(0.0); // hole filled with zero, dequant will be replaced
        } else {
            inliers.push(v);
        }
    }
    let enc = Int8Symmetric::new().try_quantize(&inliers)?;
    Ok((enc, outliers))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_roundtrip_small() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let q = Int8Symmetric::new();
        let enc = q.quantize(&x);
        let xr = q.dequantize(&enc);
        assert_eq!(xr.len(), x.len());
        for (a, b) in x.iter().zip(xr.iter()) {
            assert!((a - b).abs() < enc.scale + 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn asymmetric_roundtrip_offset() {
        let x: Vec<f32> = (0..64).map(|i| 5.0 + i as f32 * 0.05).collect();
        let q = Int8Asymmetric::new();
        let enc = q.quantize(&x);
        let xr = q.dequantize(&enc);
        for (a, b) in x.iter().zip(xr.iter()) {
            assert!((a - b).abs() < enc.scale + 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn mixed_precision_collects_outliers() {
        let mut x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.01).collect();
        x[7] = 5.0;
        x[19] = -6.0;
        let (_enc, outliers) = mixed_precision(&x, 1.0).expect("mixed");
        assert_eq!(outliers.len(), 2);
    }
}
