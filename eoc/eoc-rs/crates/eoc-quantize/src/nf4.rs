//! NormalFloat4 (NF4) — QLoRA (Dettmers et al., 2023).
//!
//! NF4 is a 4-bit code with 16 levels chosen so that each level is
//! approximately equally probable under a standard normal weight
//! distribution. The published levels are:
//!
//! ```text
//!   [-1.0,            -0.6961928,  -0.5250730, -0.3949175,
//!    -0.2844689595,   -0.1848996,  -0.0916872,  0.0,
//!     0.07958029,      0.16093843,  0.24611230, 0.33791524,
//!     0.44070983,      0.5626170,   0.7229568,  1.0]
//! ```
//!
//! Quantization picks the nearest level, then we store the index in 4
//! bits along with a per-block scale (block size typically 64).

use crate::error::QuantError;
use crate::scheme::{Numeric, QuantizationScheme};

/// The 16 NF4 levels (from the QLoRA paper).
pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073,
    -0.394_917_5,
    -0.284_468_96,
    -0.184_899_6,
    -0.091_687_2,
    0.0,
    0.079_580_29,
    0.160_938_43,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_8,
    1.0,
];

/// Encoded NF4 tensor.
#[derive(Debug, Clone)]
pub struct Nf4Encoded {
    /// Block size used.
    pub block_size: usize,
    /// Per-block absmax used as the scale.
    pub scales: Vec<f32>,
    /// Packed 4-bit indices into `NF4_LEVELS`.
    pub packed: Vec<u8>,
    /// Original length.
    pub elem_count: usize,
}

/// NF4 quantizer.
#[derive(Debug, Clone)]
pub struct Nf4Quantizer {
    /// Block size (QLoRA default = 64).
    pub block_size: usize,
}

impl Nf4Quantizer {
    /// Construct.
    pub fn new(block_size: usize) -> Self {
        Self { block_size }
    }

    fn nearest_level(v: f32) -> u8 {
        let mut best = 0u8;
        let mut best_err = f32::INFINITY;
        for (i, &l) in NF4_LEVELS.iter().enumerate() {
            let e = (v - l).abs();
            if e < best_err {
                best_err = e;
                best = i as u8;
            }
        }
        best
    }

    /// Quantize with explicit errors.
    pub fn try_quantize(&self, x: &[f32]) -> Result<Nf4Encoded, QuantError> {
        if x.is_empty() {
            return Err(QuantError::EmptyInput);
        }
        if !x.len().is_multiple_of(self.block_size) {
            return Err(QuantError::BadGroupSize {
                len: x.len(),
                group: self.block_size,
            });
        }
        let n_blocks = x.len() / self.block_size;
        let mut scales = Vec::with_capacity(n_blocks);
        let mut codes = Vec::with_capacity(x.len());
        for b in 0..n_blocks {
            let start = b * self.block_size;
            let end = start + self.block_size;
            let absmax = x[start..end]
                .iter()
                .copied()
                .fold(0.0_f32, |a, v| a.max(v.abs()))
                .max(f32::EPSILON);
            scales.push(absmax);
            for &v in &x[start..end] {
                let n = (v / absmax).clamp(-1.0, 1.0);
                codes.push(Self::nearest_level(n));
            }
        }
        Ok(Nf4Encoded {
            block_size: self.block_size,
            scales,
            packed: crate::int4::pack_nibbles(&codes),
            elem_count: x.len(),
        })
    }
}

impl QuantizationScheme for Nf4Quantizer {
    type Encoded = Nf4Encoded;

    fn numeric(&self) -> Numeric {
        Numeric::Nf4
    }

    fn quantize(&self, weights: &[f32]) -> Self::Encoded {
        self.try_quantize(weights).unwrap_or(Nf4Encoded {
            block_size: self.block_size,
            scales: Vec::new(),
            packed: Vec::new(),
            elem_count: 0,
        })
    }

    fn dequantize(&self, encoded: &Self::Encoded) -> Vec<f32> {
        let codes = crate::int4::unpack_nibbles(&encoded.packed, encoded.elem_count);
        let mut out = Vec::with_capacity(encoded.elem_count);
        for (i, &c) in codes.iter().enumerate() {
            let b = i / encoded.block_size;
            let scale = encoded.scales.get(b).copied().unwrap_or(0.0);
            let level = NF4_LEVELS.get(c as usize).copied().unwrap_or(0.0);
            out.push(level * scale);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nf4_levels_are_sorted_and_unique() {
        let mut prev = f32::NEG_INFINITY;
        for &l in &NF4_LEVELS {
            assert!(l > prev, "NF4 levels must be strictly increasing");
            prev = l;
        }
    }

    #[test]
    fn nf4_roundtrip_within_block_tolerance() {
        let x: Vec<f32> = (0..64)
            .map(|i| ((i as f32 - 32.0) / 32.0).clamp(-1.0, 1.0))
            .collect();
        let q = Nf4Quantizer::new(64);
        let enc = q.quantize(&x);
        let xr = q.dequantize(&enc);
        let max_err = x
            .iter()
            .zip(xr.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_err < 0.2, "nf4 max err {max_err}");
    }
}
