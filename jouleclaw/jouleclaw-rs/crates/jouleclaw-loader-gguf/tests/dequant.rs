//! Tests for quantized dequantization. Each format is verified with
//! hand-constructed blocks where the expected output is computable from
//! the formula.

use jouleclaw_loader_gguf::dequant::{dequantize_q4_k, dequantize_q5_k, dequantize_q8_0};

/// Encode an f32 as IEEE 754 binary16 little-endian. Just inverse of the
/// loader's f16->f32 we already have, used here only for test-block
/// construction. We pick exactly representable values so rounding doesn't
/// confuse anything.
fn f32_to_f16_bytes(v: f32) -> [u8; 2] {
    // Implementation only needs to round-trip the values we use in tests:
    // 0.0, 1.0, 2.0, 0.5, 0.25.
    let bits = v.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    let h_bits: u16 = if v == 0.0 {
        (sign as u16) << 15
    } else {
        let h_exp = exp - 127 + 15;
        if !(1..=30).contains(&h_exp) {
            panic!("test helper does not support {} (out of normal F16 range)", v);
        }
        let h_mant = (mant >> 13) as u16;
        ((sign as u16) << 15) | ((h_exp as u16) << 10) | h_mant
    };
    h_bits.to_le_bytes()
}

// =====================================================================
// Q8_0
// =====================================================================

#[test]
fn q8_0_zero_scale_yields_all_zeros() {
    // d = 0, all q = anything → all output should be 0.
    let mut block = vec![0u8; 34];
    block[0..2].copy_from_slice(&f32_to_f16_bytes(0.0));
    for i in 0..32 { block[2 + i] = (i as i8 + 1) as u8; }

    let out = dequantize_q8_0(&block, 32).unwrap();
    assert_eq!(out, vec![0.0; 32]);
}

#[test]
fn q8_0_unit_scale_returns_quants_as_floats() {
    // d = 1.0, q = [-5, -4, ..., 26]. Output should equal q.
    let mut block = vec![0u8; 34];
    block[0..2].copy_from_slice(&f32_to_f16_bytes(1.0));
    for i in 0..32 {
        let q = (i as i32 - 5) as i8;
        block[2 + i] = q as u8;
    }
    let out = dequantize_q8_0(&block, 32).unwrap();
    let expected: Vec<f32> = (0..32).map(|i| (i as i32 - 5) as f32).collect();
    assert_eq!(out, expected);
}

#[test]
fn q8_0_scale_two_doubles_quants() {
    // d = 2.0, q = 1..=32. Output = 2,4,6,...,64.
    let mut block = vec![0u8; 34];
    block[0..2].copy_from_slice(&f32_to_f16_bytes(2.0));
    for i in 0..32 { block[2 + i] = (i as i8 + 1) as u8; }
    let out = dequantize_q8_0(&block, 32).unwrap();
    for i in 0..32 {
        assert_eq!(out[i], 2.0 * (i as f32 + 1.0));
    }
}

#[test]
fn q8_0_multiple_blocks() {
    // Two blocks: first with d=1.0 and q=1..32, second with d=2.0 and q=1..32.
    let mut block = vec![0u8; 68];
    block[0..2].copy_from_slice(&f32_to_f16_bytes(1.0));
    for i in 0..32 { block[2 + i] = (i as i8 + 1) as u8; }
    block[34..36].copy_from_slice(&f32_to_f16_bytes(2.0));
    for i in 0..32 { block[36 + i] = (i as i8 + 1) as u8; }

    let out = dequantize_q8_0(&block, 64).unwrap();
    for i in 0..32 {
        assert_eq!(out[i], (i + 1) as f32);
        assert_eq!(out[32 + i], 2.0 * (i + 1) as f32);
    }
}

// =====================================================================
// Q4_K
// =====================================================================

/// Build a Q4_K block with given d, dmin, all sub-block scales = `sc`,
/// all sub-block mins = `mn`, all 4-bit quants = `q_value` (0..=15).
fn build_q4_k_block(d: f32, dmin: f32, sc: u8, mn: u8, q_value: u8) -> Vec<u8> {
    let mut block = vec![0u8; 144];
    block[0..2].copy_from_slice(&f32_to_f16_bytes(d));
    block[2..4].copy_from_slice(&f32_to_f16_bytes(dmin));

    // Pack 12-byte scales array so all scale_j = sc and all min_j = mn for j in 0..8.
    // Layout (verified against ggml's get_scale_min_k4):
    //   For j in 0..4:
    //     scales[j]      bits 0-5 = scale_j, bits 6-7 = (scale_{j+4} >> 4) & 0x3
    //     scales[j + 4]  bits 0-5 = min_j,   bits 6-7 = (min_{j+4} >> 4)   & 0x3
    //   For j in 4..8:
    //     scales[j + 4]  bits 0-3 = scale_j & 0xF, bits 4-7 = min_j & 0xF
    assert!(sc < 64 && mn < 64, "6-bit values only");

    let scales_high_2 = (sc >> 4) & 0x3;
    let mins_high_2 = (mn >> 4) & 0x3;

    for j in 0..4 {
        block[4 + j] = (sc & 0x3F) | (scales_high_2 << 6);
        block[4 + j + 4] = (mn & 0x3F) | (mins_high_2 << 6);
    }
    for j in 4..8 {
        // bytes 8..11 = scales[j+4]; we want (sc & 0xF) in low nibble, (mn & 0xF) in high nibble
        block[4 + j + 4] = (sc & 0x0F) | ((mn & 0x0F) << 4);
    }

    // Quantized data: 128 bytes; both nibbles = q_value.
    let packed = (q_value & 0x0F) | ((q_value & 0x0F) << 4);
    for i in 0..128 { block[16 + i] = packed; }

    block
}

#[test]
fn q4_k_zero_scales_yield_zero_minus_zero() {
    // d=0, dmin=0 → all output = 0 regardless of q.
    let block = build_q4_k_block(0.0, 0.0, 5, 3, 12);
    let out = dequantize_q4_k(&block, 256).unwrap();
    assert_eq!(out, vec![0.0; 256]);
}

#[test]
fn q4_k_unit_d_with_zero_min_returns_sc_times_q() {
    // d=1, dmin=0, all sc=2, all q=3 → output = 1*2*3 - 0*m = 6.
    let block = build_q4_k_block(1.0, 0.0, 2, 7, 3);
    let out = dequantize_q4_k(&block, 256).unwrap();
    for v in &out { assert!((v - 6.0).abs() < 1e-6, "expected 6, got {}", v); }
}

#[test]
fn q4_k_min_subtraction_works() {
    // d=0, dmin=1, all sc=anything, all m=4 → output = 0 - 1*4 = -4.
    let block = build_q4_k_block(0.0, 1.0, 7, 4, 9);
    let out = dequantize_q4_k(&block, 256).unwrap();
    for v in &out { assert!((v + 4.0).abs() < 1e-6, "expected -4, got {}", v); }
}

#[test]
fn q4_k_general_formula() {
    // d=2.0, dmin=0.5, all sc=3, all m=2, all q=10.
    // Output = 2.0 * 3 * 10 - 0.5 * 2 = 60 - 1 = 59.
    let block = build_q4_k_block(2.0, 0.5, 3, 2, 10);
    let out = dequantize_q4_k(&block, 256).unwrap();
    for v in &out { assert!((v - 59.0).abs() < 1e-4, "expected 59, got {}", v); }
}

#[test]
fn q4_k_with_high_6bit_values_packs_correctly() {
    // sc=63, mn=63 (max 6-bit values). Tests that the high-2-bit packing
    // for sub-blocks 4..7 works correctly.
    let block = build_q4_k_block(1.0, 0.0, 63, 0, 1);
    let out = dequantize_q4_k(&block, 256).unwrap();
    // Output = 1.0 * 63 * 1 - 0 = 63 for every element across all 8 sub-blocks.
    for (i, v) in out.iter().enumerate() {
        assert!((v - 63.0).abs() < 1e-4,
            "elem {}: expected 63, got {}", i, v);
    }
}

// =====================================================================
// Q5_K
// =====================================================================

/// Build a Q5_K block with given d, dmin, sc, mn, and a single low-4-bit
/// q_lo value plus a single high-bit value (0 or 1).
fn build_q5_k_block(d: f32, dmin: f32, sc: u8, mn: u8, q_lo: u8, q_hi: u8) -> Vec<u8> {
    let mut block = vec![0u8; 176];
    block[0..2].copy_from_slice(&f32_to_f16_bytes(d));
    block[2..4].copy_from_slice(&f32_to_f16_bytes(dmin));

    let scales_high_2 = (sc >> 4) & 0x3;
    let mins_high_2 = (mn >> 4) & 0x3;
    for j in 0..4 {
        block[4 + j] = (sc & 0x3F) | (scales_high_2 << 6);
        block[4 + j + 4] = (mn & 0x3F) | (mins_high_2 << 6);
    }
    for j in 4..8 {
        block[4 + j + 4] = (sc & 0x0F) | ((mn & 0x0F) << 4);
    }

    // qh: 32 bytes. We want every output element's high bit = q_hi.
    // qh[k % 32] >> (k / 32) & 1 = q_hi for all k in 0..256.
    // If q_hi = 0, all qh bytes = 0. If q_hi = 1, all qh bytes = 0xFF.
    let qh_byte = if q_hi & 1 != 0 { 0xFF } else { 0x00 };
    for i in 0..32 { block[16 + i] = qh_byte; }

    // qs: 128 bytes. Both nibbles = q_lo.
    let packed = (q_lo & 0x0F) | ((q_lo & 0x0F) << 4);
    for i in 0..128 { block[48 + i] = packed; }

    block
}

#[test]
fn q5_k_low_bits_only_match_q4_k_formula() {
    // q_hi=0 → effective q = q_lo (0..=15). With d=1, dmin=0, sc=1, mn=0, q_lo=7:
    // output = 1 * 1 * 7 - 0*0 = 7.
    let block = build_q5_k_block(1.0, 0.0, 1, 0, 7, 0);
    let out = dequantize_q5_k(&block, 256).unwrap();
    for v in &out { assert!((v - 7.0).abs() < 1e-6, "got {}", v); }
}

#[test]
fn q5_k_high_bit_extends_quant_to_5_bits() {
    // q_hi=1, q_lo=3 → effective q = (1<<4) | 3 = 19. With d=1, sc=1, dmin=0:
    // output = 1*1*19 - 0 = 19.
    let block = build_q5_k_block(1.0, 0.0, 1, 0, 3, 1);
    let out = dequantize_q5_k(&block, 256).unwrap();
    for v in &out { assert!((v - 19.0).abs() < 1e-6, "expected 19, got {}", v); }
}

#[test]
fn q5_k_max_quant_value() {
    // q_lo=15, q_hi=1 → effective q = 31 (max for 5-bit unsigned).
    // d=2, sc=1, dmin=0 → output = 2*1*31 = 62.
    let block = build_q5_k_block(2.0, 0.0, 1, 0, 15, 1);
    let out = dequantize_q5_k(&block, 256).unwrap();
    for v in &out { assert!((v - 62.0).abs() < 1e-4, "got {}", v); }
}

// =====================================================================
// Error handling
// =====================================================================

#[test]
fn rejects_misaligned_element_count() {
    let block = vec![0u8; 144];
    assert!(dequantize_q4_k(&block, 100).is_err(),
        "100 is not a multiple of 256");
    assert!(dequantize_q5_k(&block, 100).is_err());
    assert!(dequantize_q8_0(&block, 31).is_err(),
        "31 is not a multiple of 32");
}

#[test]
fn rejects_buffer_too_small() {
    let block = vec![0u8; 50];  // way too small for a Q4_K block
    assert!(dequantize_q4_k(&block, 256).is_err());
}
