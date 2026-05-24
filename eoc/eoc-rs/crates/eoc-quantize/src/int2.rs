//! Int2 weights, BitNet-style.
//!
//! BitNet (Wang et al., 2023) and the follow-up BitNet b1.58 use
//! ternary weights {-1, 0, +1}. We pack two ternary codes per byte
//! (4 codes per byte if we are aggressive, but 2 keeps the API
//! consistent with int4).
//!
//! Codes:
//!   0b00 → -1
//!   0b01 →  0
//!   0b10 → +1
//!   0b11 → reserved (treated as 0)

use crate::error::QuantError;
use crate::scheme::{Numeric, QuantizationScheme};

/// Encoded ternary tensor.
#[derive(Debug, Clone)]
pub struct Int2Encoded {
    /// Single per-tensor scale.
    pub scale: f32,
    /// Packed codes, four per byte (low pair first).
    pub packed: Vec<u8>,
    /// Original length.
    pub elem_count: usize,
}

/// BitNet-style ternary quantizer.
#[derive(Debug, Clone, Default)]
pub struct BitNetQuantizer;

impl BitNetQuantizer {
    /// Construct.
    pub fn new() -> Self {
        Self
    }

    /// Quantize with explicit errors.
    pub fn try_quantize(&self, x: &[f32]) -> Result<Int2Encoded, QuantError> {
        if x.is_empty() {
            return Err(QuantError::EmptyInput);
        }
        // BitNet b1.58 uses the per-tensor absmean as the scale.
        let absmean = x.iter().map(|v| v.abs()).sum::<f32>() / x.len() as f32;
        let scale = absmean.max(f32::EPSILON);
        let threshold = 0.5 * scale;
        let codes: Vec<u8> = x
            .iter()
            .map(|&v| {
                if v > threshold {
                    0b10
                } else if v < -threshold {
                    0b00
                } else {
                    0b01
                }
            })
            .collect();
        Ok(Int2Encoded {
            scale,
            packed: pack_pairs(&codes),
            elem_count: x.len(),
        })
    }
}

/// Pack 2-bit codes four per byte (LSBs first).
pub fn pack_pairs(codes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(codes.len().div_ceil(4));
    let mut i = 0;
    while i < codes.len() {
        let mut byte = 0u8;
        for (slot, code) in codes[i..(i + 4).min(codes.len())].iter().enumerate() {
            byte |= (code & 0b11) << (2 * slot);
        }
        out.push(byte);
        i += 4;
    }
    out
}

/// Unpack 2-bit codes for `elem_count` elements.
pub fn unpack_pairs(packed: &[u8], elem_count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(elem_count);
    for &b in packed {
        for slot in 0..4 {
            if out.len() == elem_count {
                return out;
            }
            out.push((b >> (2 * slot)) & 0b11);
        }
    }
    out
}

impl QuantizationScheme for BitNetQuantizer {
    type Encoded = Int2Encoded;

    fn numeric(&self) -> Numeric {
        Numeric::Int2
    }

    fn quantize(&self, weights: &[f32]) -> Self::Encoded {
        self.try_quantize(weights).unwrap_or(Int2Encoded {
            scale: 1.0,
            packed: Vec::new(),
            elem_count: 0,
        })
    }

    fn dequantize(&self, encoded: &Self::Encoded) -> Vec<f32> {
        let codes = unpack_pairs(&encoded.packed, encoded.elem_count);
        codes
            .iter()
            .map(|&c| match c {
                0b00 => -encoded.scale,
                0b10 => encoded.scale,
                _ => 0.0,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_roundtrip_preserves_sign() {
        let x = vec![-1.0_f32, -0.05, 0.0, 0.05, 1.0, -0.7, 0.7];
        let q = BitNetQuantizer::new();
        let enc = q.quantize(&x);
        let xr = q.dequantize(&enc);
        for (a, b) in x.iter().zip(xr.iter()) {
            assert_eq!(a.signum().abs() > 0.5 && a.abs() > 0.1, b.abs() > 0.0);
            if *a > 0.5 {
                assert!(*b > 0.0);
            }
            if *a < -0.5 {
                assert!(*b < 0.0);
            }
        }
    }

    #[test]
    fn pair_pack_unpack_roundtrip() {
        let codes = vec![0b00, 0b01, 0b10, 0b11, 0b00, 0b10, 0b01];
        let p = pack_pairs(&codes);
        let u = unpack_pairs(&p, codes.len());
        assert_eq!(u, codes);
    }
}
