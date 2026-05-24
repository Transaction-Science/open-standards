//! FP8 — both OFP8 variants used in modern accelerators.
//!
//! * **E4M3** — 1 sign bit, 4 exponent bits, 3 mantissa bits, bias 7.
//!   Range roughly ±448. Used for inference activations on Hopper.
//! * **E5M2** — 1 sign bit, 5 exponent bits, 2 mantissa bits, bias 15.
//!   Range roughly ±57344. Used for gradients during training.
//!
//! These are byte-level encoders implemented with f32 arithmetic only.
//! They are deliberately simple and treat NaN/Inf conservatively (any
//! non-finite input becomes the largest finite representable value
//! with the original sign).

use crate::error::QuantError;

/// E4M3 encoding (no infinities; one NaN encoding 0xFF / 0x7F).
pub fn encode_e4m3(x: f32) -> u8 {
    encode_fpx(x, 4, 3, 7, 448.0)
}

/// E4M3 decoding.
pub fn decode_e4m3(b: u8) -> f32 {
    decode_fpx(b, 4, 3, 7)
}

/// E5M2 encoding.
pub fn encode_e5m2(x: f32) -> u8 {
    encode_fpx(x, 5, 2, 15, 57344.0)
}

/// E5M2 decoding.
pub fn decode_e5m2(b: u8) -> f32 {
    decode_fpx(b, 5, 2, 15)
}

fn encode_fpx(x: f32, exp_bits: u32, man_bits: u32, bias: i32, max_finite: f32) -> u8 {
    let sign = if x.is_sign_negative() { 1u8 } else { 0u8 };
    let mag = x.abs();
    if !mag.is_finite() {
        return (sign << 7) | (((1 << exp_bits) - 2) as u8) << man_bits | ((1 << man_bits) - 1) as u8;
    }
    let clamped = mag.min(max_finite);
    if clamped == 0.0 {
        return sign << 7;
    }
    let exp_unbiased = clamped.log2().floor() as i32;
    let exp_field = (exp_unbiased + bias).clamp(0, (1 << exp_bits) - 1);
    // Subnormal handling: if exp_field is 0, encode magnitude as
    // mantissa relative to 2^(1 - bias).
    if exp_field == 0 {
        let denorm_scale = (1u32 << man_bits) as f32 / 2f32.powi(1 - bias);
        let mant = (clamped * denorm_scale).round() as u32;
        let mant = mant.min((1 << man_bits) - 1);
        return (sign << 7) | (mant as u8 & ((1 << man_bits) - 1));
    }
    let pow = 2f32.powi(exp_unbiased);
    let frac = clamped / pow - 1.0;
    let mant = (frac * (1u32 << man_bits) as f32).round() as u32;
    let mant = mant.min((1 << man_bits) - 1);
    let exp_u = exp_field as u32;
    (sign << 7) | ((exp_u as u8) << man_bits) | (mant as u8 & ((1 << man_bits) - 1))
}

fn decode_fpx(b: u8, exp_bits: u32, man_bits: u32, bias: i32) -> f32 {
    let sign = if (b >> 7) & 1 == 1 { -1.0_f32 } else { 1.0 };
    let exp_mask = (1u8 << exp_bits) - 1;
    let man_mask = (1u8 << man_bits) - 1;
    let exp_field = ((b >> man_bits) & exp_mask) as i32;
    let mant_field = (b & man_mask) as u32;
    if exp_field == 0 {
        let denorm = mant_field as f32 / (1u32 << man_bits) as f32;
        return sign * denorm * 2f32.powi(1 - bias);
    }
    let frac = 1.0 + mant_field as f32 / (1u32 << man_bits) as f32;
    sign * frac * 2f32.powi(exp_field - bias)
}

/// Encode a slice as E4M3 bytes.
pub fn encode_slice_e4m3(x: &[f32]) -> Result<Vec<u8>, QuantError> {
    if x.is_empty() {
        return Err(QuantError::EmptyInput);
    }
    Ok(x.iter().map(|&v| encode_e4m3(v)).collect())
}

/// Encode a slice as E5M2 bytes.
pub fn encode_slice_e5m2(x: &[f32]) -> Result<Vec<u8>, QuantError> {
    if x.is_empty() {
        return Err(QuantError::EmptyInput);
    }
    Ok(x.iter().map(|&v| encode_e5m2(v)).collect())
}

/// Decode a slice of E4M3 bytes.
pub fn decode_slice_e4m3(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| decode_e4m3(b)).collect()
}

/// Decode a slice of E5M2 bytes.
pub fn decode_slice_e5m2(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| decode_e5m2(b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_roundtrip_small_values() {
        for &v in &[0.5_f32, 1.0, 2.0, -2.0, 16.0, -64.0] {
            let b = encode_e4m3(v);
            let r = decode_e4m3(b);
            let rel = ((r - v) / v).abs();
            assert!(rel < 0.25, "{v} -> {r}");
        }
    }

    #[test]
    fn e5m2_roundtrip_large_range() {
        for &v in &[1.0_f32, 100.0, 1000.0, -10000.0] {
            let b = encode_e5m2(v);
            let r = decode_e5m2(b);
            let rel = ((r - v) / v).abs();
            assert!(rel < 0.5, "{v} -> {r}");
        }
    }

    #[test]
    fn fp8_zero_roundtrips() {
        assert_eq!(decode_e4m3(encode_e4m3(0.0)), 0.0);
        assert_eq!(decode_e5m2(encode_e5m2(0.0)), 0.0);
    }
}
