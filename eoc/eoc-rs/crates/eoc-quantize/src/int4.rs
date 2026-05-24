//! Int4 group quantization in the GPTQ / AWQ family.
//!
//! Both GPTQ (Frantar et al., 2022) and AWQ (Lin et al., 2023) share a
//! storage layout: weights are split into groups of `G` consecutive
//! elements; each group carries its own `scale` (and optionally a
//! `zero_point`). The 4-bit codes are then packed two per byte.
//!
//! The differences live in *how* the scale + zero point are chosen:
//!
//! * **GPTQ** — picks scales via OBS-style error compensation that
//!   sequentially adjusts surviving weights to absorb the rounding
//!   error of already-quantized ones. We approximate this here with a
//!   greedy single-pass propagation of residual error within each
//!   group. The on-disk format is identical.
//! * **AWQ** — uses an *activation-aware* per-channel salience score to
//!   scale the weight matrix before quantization (so high-importance
//!   channels eat less rounding noise). We expose `AwqGroupQuantizer`
//!   that accepts a per-channel scale and applies it before grouping.
//!
//! Both produce `Int4GroupEncoded`.

use crate::error::QuantError;
use crate::scheme::{Numeric, QuantizationScheme};

/// Encoded form: one (scale, zp) per group plus packed nibbles.
#[derive(Debug, Clone)]
pub struct Int4GroupEncoded {
    /// Group size used during encoding.
    pub group_size: usize,
    /// Per-group scales (`weights.len() / group_size` entries).
    pub scales: Vec<f32>,
    /// Per-group zero points in [0, 15].
    pub zero_points: Vec<u8>,
    /// Packed 4-bit codes — two values per byte, low nibble first.
    pub packed: Vec<u8>,
    /// Original element count (for round-trip).
    pub elem_count: usize,
}

/// GPTQ-style int4 group quantizer with greedy error compensation.
#[derive(Debug, Clone)]
pub struct GptqGroupQuantizer {
    /// Group size (typically 32, 64, or 128).
    pub group_size: usize,
}

impl GptqGroupQuantizer {
    /// Construct.
    pub fn new(group_size: usize) -> Self {
        Self { group_size }
    }

    fn quantize_group(group: &[f32]) -> (f32, u8, Vec<u8>) {
        let (mn, mx) = group.iter().copied().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(a, b), v| (a.min(v), b.max(v)),
        );
        // Extend the range to include zero so the affine map covers
        // fp 0 exactly (matches the int8 asymmetric convention).
        let mn_ext = mn.min(0.0);
        let mx_ext = mx.max(0.0);
        let span = (mx_ext - mn_ext).max(f32::EPSILON);
        let scale = span / 15.0;
        let zp_f = (-mn_ext / scale).round().clamp(0.0, 15.0);
        let zp = zp_f as u8;

        // GPTQ-style: walk forward, push residual error onto the next
        // weight in the group before quantizing it. We cap the absorbed
        // residual at half a step to avoid the well-known instability
        // of naive forward propagation without a Hessian damping term.
        let half_step = 0.5 * scale;
        let mut codes = Vec::with_capacity(group.len());
        let mut residual = 0.0_f32;
        for &v in group {
            let absorb = residual.clamp(-half_step, half_step);
            let target = v + absorb;
            let q = (((target - mn_ext) / scale).round()).clamp(0.0, 15.0);
            let recon = (q - zp_f) * scale;
            residual = v - recon;
            codes.push(q as u8);
        }
        (scale, zp, codes)
    }

    /// Quantize with explicit error reporting.
    pub fn try_quantize(&self, x: &[f32]) -> Result<Int4GroupEncoded, QuantError> {
        if x.is_empty() {
            return Err(QuantError::EmptyInput);
        }
        if !x.len().is_multiple_of(self.group_size) {
            return Err(QuantError::BadGroupSize {
                len: x.len(),
                group: self.group_size,
            });
        }
        let n_groups = x.len() / self.group_size;
        let mut scales = Vec::with_capacity(n_groups);
        let mut zps = Vec::with_capacity(n_groups);
        let mut codes = Vec::with_capacity(x.len());
        for g in 0..n_groups {
            let start = g * self.group_size;
            let end = start + self.group_size;
            let (s, z, c) = Self::quantize_group(&x[start..end]);
            scales.push(s);
            zps.push(z);
            codes.extend(c);
        }
        Ok(Int4GroupEncoded {
            group_size: self.group_size,
            scales,
            zero_points: zps,
            packed: pack_nibbles(&codes),
            elem_count: x.len(),
        })
    }
}

impl QuantizationScheme for GptqGroupQuantizer {
    type Encoded = Int4GroupEncoded;

    fn numeric(&self) -> Numeric {
        Numeric::Int4
    }

    fn quantize(&self, weights: &[f32]) -> Self::Encoded {
        self.try_quantize(weights).unwrap_or(Int4GroupEncoded {
            group_size: self.group_size,
            scales: Vec::new(),
            zero_points: Vec::new(),
            packed: Vec::new(),
            elem_count: 0,
        })
    }

    fn dequantize(&self, encoded: &Self::Encoded) -> Vec<f32> {
        dequant_int4_group(encoded)
    }
}

/// AWQ-style quantizer: applies a per-channel salience scale before
/// grouping. `channel_scales.len()` must equal `weights.len()`.
#[derive(Debug, Clone)]
pub struct AwqGroupQuantizer {
    /// Group size.
    pub group_size: usize,
}

impl AwqGroupQuantizer {
    /// Construct.
    pub fn new(group_size: usize) -> Self {
        Self { group_size }
    }

    /// Apply per-element salience scaling then group-quantize. The
    /// returned encoding bakes the salience scale into `scales` so
    /// `dequantize` returns weights in the original space.
    pub fn try_quantize(
        &self,
        x: &[f32],
        channel_scales: &[f32],
    ) -> Result<Int4GroupEncoded, QuantError> {
        if x.is_empty() {
            return Err(QuantError::EmptyInput);
        }
        if x.len() != channel_scales.len() {
            return Err(QuantError::BadGroupSize {
                len: channel_scales.len(),
                group: x.len(),
            });
        }
        let scaled: Vec<f32> = x
            .iter()
            .zip(channel_scales.iter())
            .map(|(&v, &s)| v * s)
            .collect();
        let mut enc = GptqGroupQuantizer::new(self.group_size).try_quantize(&scaled)?;
        // Bake the inverse salience back into the per-group scale to
        // average out the channel adjustment within each group.
        for (g_idx, scale) in enc.scales.iter_mut().enumerate() {
            let start = g_idx * enc.group_size;
            let end = start + enc.group_size;
            let mean_inv: f32 = channel_scales[start..end]
                .iter()
                .map(|s| 1.0 / s.max(f32::EPSILON))
                .sum::<f32>()
                / enc.group_size as f32;
            *scale *= mean_inv;
        }
        Ok(enc)
    }
}

/// Pack a slice of 4-bit codes (low nibble first).
pub fn pack_nibbles(codes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(codes.len().div_ceil(2));
    let mut i = 0;
    while i < codes.len() {
        let lo = codes[i] & 0x0f;
        let hi = if i + 1 < codes.len() {
            codes[i + 1] & 0x0f
        } else {
            0
        };
        out.push(lo | (hi << 4));
        i += 2;
    }
    out
}

/// Unpack a 4-bit code stream of length `elem_count`.
pub fn unpack_nibbles(packed: &[u8], elem_count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(elem_count);
    for b in packed {
        out.push(b & 0x0f);
        if out.len() < elem_count {
            out.push((b >> 4) & 0x0f);
        }
    }
    out.truncate(elem_count);
    out
}

/// Dequantize an int4 group encoding back to fp32.
pub fn dequant_int4_group(enc: &Int4GroupEncoded) -> Vec<f32> {
    let codes = unpack_nibbles(&enc.packed, enc.elem_count);
    let mut out = Vec::with_capacity(enc.elem_count);
    for (i, &c) in codes.iter().enumerate() {
        let g = i / enc.group_size;
        let scale = enc.scales.get(g).copied().unwrap_or(0.0);
        let zp = enc.zero_points.get(g).copied().unwrap_or(0) as f32;
        out.push((c as f32 - zp) * scale);
    }
    out
}

/// SmoothQuant (Xiao et al., 2022) — migrate per-channel activation
/// magnitude into the weights so both are easier to quantize.
///
/// Given per-channel activation absmax `act_absmax` and per-channel
/// weight absmax `w_absmax`, this returns the migration factor `s_i`
/// such that `act_i / s_i` and `w_i * s_i` are jointly more friendly
/// to quantization. The standard choice is `s_i = a_i^α / w_i^(1-α)`
/// with α in [0, 1] (default 0.5).
pub fn smoothquant_scales(act_absmax: &[f32], w_absmax: &[f32], alpha: f32) -> Vec<f32> {
    let n = act_absmax.len().min(w_absmax.len());
    (0..n)
        .map(|i| {
            let a = act_absmax[i].max(f32::EPSILON);
            let w = w_absmax[i].max(f32::EPSILON);
            a.powf(alpha) / w.powf(1.0 - alpha)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gptq_roundtrip_within_tolerance() {
        let x: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.02).collect();
        let q = GptqGroupQuantizer::new(32);
        let enc = q.quantize(&x);
        assert_eq!(enc.scales.len(), 4);
        let xr = q.dequantize(&enc);
        let max_err = x
            .iter()
            .zip(xr.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // int4 with group=32 over a small range should be well-bounded.
        assert!(max_err < 0.1, "max int4 err {max_err}");
    }

    #[test]
    fn nibble_pack_unpack_roundtrip() {
        let codes: Vec<u8> = (0..7).map(|i| i as u8 & 0x0f).collect();
        let packed = pack_nibbles(&codes);
        let unpacked = unpack_nibbles(&packed, codes.len());
        assert_eq!(unpacked, codes);
    }

    #[test]
    fn awq_accepts_salience_scales() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let scales: Vec<f32> = (0..64).map(|_| 1.0).collect();
        let enc = AwqGroupQuantizer::new(32)
            .try_quantize(&x, &scales)
            .expect("awq quantize");
        let xr = dequant_int4_group(&enc);
        assert_eq!(xr.len(), x.len());
    }
}
