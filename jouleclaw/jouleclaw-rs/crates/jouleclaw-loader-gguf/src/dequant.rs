//! Dequantization of GGML quantized tensor formats.
//!
//! Supported in Phase 1.6:
//! - **Q8_0** — symmetric 8-bit, block of 32 elements, 1 FP16 scale per block.
//!   34 bytes / 32 elements = 8.5 bits/weight.
//! - **Q4_K** — K-quant 4-bit, block of 256 elements with 8 sub-blocks.
//!   144 bytes / 256 elements = 4.5 bits/weight. The dominant format in
//!   distributed Llama-family models (Q4_K_M).
//! - **Q5_K** — K-quant 5-bit, block of 256 elements with 8 sub-blocks.
//!   176 bytes / 256 elements = 5.5 bits/weight.
//!
//! All implementations follow the `dequantize_row_q*_K` reference functions
//! in ggml/llama.cpp. Outputs are fully F32; deterministic across platforms.
//!
//! References: ggml/src/ggml-quants.c

use crate::f16_to_f32;

#[derive(Debug)]
pub enum DequantError {
    NotMultipleOfBlockSize { total: usize, block: usize },
    BufferTooSmall { need: usize, have: usize },
}

impl std::fmt::Display for DequantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotMultipleOfBlockSize { total, block } =>
                write!(f, "tensor element count {} not a multiple of block size {}", total, block),
            Self::BufferTooSmall { need, have } =>
                write!(f, "tensor data buffer too small: need {} bytes, have {}", need, have),
        }
    }
}

impl std::error::Error for DequantError {}

// =====================================================================
// Q8_0
// =====================================================================
//
// Per-block layout (34 bytes for 32 elements):
//   0..2   : FP16 scale `d`
//   2..34  : 32 signed int8 quantized values
// Dequant: y[i] = d * q[i]

const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_ELEMS: usize = 32;

pub fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, DequantError> {
    if n_elements % Q8_0_ELEMS != 0 {
        return Err(DequantError::NotMultipleOfBlockSize {
            total: n_elements, block: Q8_0_ELEMS,
        });
    }
    let n_blocks = n_elements / Q8_0_ELEMS;
    let need = n_blocks * Q8_0_BLOCK_BYTES;
    if data.len() < need {
        return Err(DequantError::BufferTooSmall { need, have: data.len() });
    }
    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let off = b * Q8_0_BLOCK_BYTES;
        let d = f16_from_le(&data[off..off + 2]);
        for i in 0..Q8_0_ELEMS {
            let q = data[off + 2 + i] as i8;
            out.push(d * (q as f32));
        }
    }
    Ok(out)
}

// =====================================================================
// Q4_K
// =====================================================================
//
// Per-block layout (144 bytes for 256 elements):
//   0..2     : FP16 super-scale `d`
//   2..4     : FP16 super-min `dmin`
//   4..16    : 12 bytes packed 6-bit scales (8) and 6-bit mins (8)
//   16..144  : 128 bytes of 4-bit quantized values (256 nibbles)
//
// 8 sub-blocks of 32 elements each.
// Dequant: y[k] = d * scale[s] * q[k] - dmin * min[s]
//   where s is the sub-block index, q is the 4-bit unsigned quant value.
//
// Scale/min decoding from the 12-byte `scales` array:
//   for j in 0..4:  scale_j = scales[j]   & 0x3F
//                   min_j   = scales[j+4] & 0x3F
//   for j in 4..8:  scale_j = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
//                   min_j   = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)

const Q_K_SUPER: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;

fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, mn)
    }
}

pub fn dequantize_q4_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, DequantError> {
    if n_elements % Q_K_SUPER != 0 {
        return Err(DequantError::NotMultipleOfBlockSize {
            total: n_elements, block: Q_K_SUPER,
        });
    }
    let n_blocks = n_elements / Q_K_SUPER;
    let need = n_blocks * Q4_K_BLOCK_BYTES;
    if data.len() < need {
        return Err(DequantError::BufferTooSmall { need, have: data.len() });
    }

    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let off = b * Q4_K_BLOCK_BYTES;
        let d = f16_from_le(&data[off..off + 2]);
        let dmin = f16_from_le(&data[off + 2..off + 4]);
        let scales = &data[off + 4..off + 16];
        let qs = &data[off + 16..off + 144];

        // 4 sub-block pairs, each pair processes 64 output elements
        // (32 from low nibbles + 32 from high nibbles), advancing qs by 32 bytes.
        let mut is = 0usize;
        let mut q_off = 0usize;
        for _pair in 0..4 {
            let (sc1, m1_q) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1_q as f32;
            let (sc2, m2_q) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2_q as f32;

            // First sub-block: low nibbles of qs[q_off .. q_off+32]
            for l in 0..32 {
                let q = (qs[q_off + l] & 0x0F) as f32;
                out.push(d1 * q - m1);
            }
            // Second sub-block: high nibbles of qs[q_off .. q_off+32]
            for l in 0..32 {
                let q = (qs[q_off + l] >> 4) as f32;
                out.push(d2 * q - m2);
            }

            is += 2;
            q_off += 32;
        }
    }
    Ok(out)
}

// =====================================================================
// Q5_K
// =====================================================================
//
// Per-block layout (176 bytes for 256 elements):
//   0..2     : FP16 super-scale `d`
//   2..4     : FP16 super-min `dmin`
//   4..16    : 12 bytes packed 6-bit scales/mins (same layout as Q4_K)
//   16..48   : 32 bytes high-bit storage `qh` (1 bit per element)
//   48..176  : 128 bytes of low-4-bit quants
//
// Each 5-bit quant: low 4 bits from qs nibble, high 1 bit from qh.
// Dequant identical to Q4_K with a 5-bit q (0..31).

const Q5_K_BLOCK_BYTES: usize = 176;

pub fn dequantize_q5_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, DequantError> {
    if n_elements % Q_K_SUPER != 0 {
        return Err(DequantError::NotMultipleOfBlockSize {
            total: n_elements, block: Q_K_SUPER,
        });
    }
    let n_blocks = n_elements / Q_K_SUPER;
    let need = n_blocks * Q5_K_BLOCK_BYTES;
    if data.len() < need {
        return Err(DequantError::BufferTooSmall { need, have: data.len() });
    }

    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let off = b * Q5_K_BLOCK_BYTES;
        let d = f16_from_le(&data[off..off + 2]);
        let dmin = f16_from_le(&data[off + 2..off + 4]);
        let scales = &data[off + 4..off + 16];
        let qh = &data[off + 16..off + 48];
        let qs = &data[off + 48..off + 176];

        let mut is = 0usize;
        let mut q_off = 0usize;
        // qh has 32 bytes; we use one bit per output element. The bit layout
        // mirrors Q5_K reference: for output element k (within block of 256),
        // the high bit comes from qh[k % 32] >> (k / 32).
        // Track per-element output index `k` so we can read qh correctly.
        let mut k = 0usize;
        for _pair in 0..4 {
            let (sc1, m1_q) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1_q as f32;
            let (sc2, m2_q) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2_q as f32;

            for l in 0..32 {
                let lo = qs[q_off + l] & 0x0F;
                let hi = (qh[k % 32] >> (k / 32)) & 1;
                let q = (lo | (hi << 4)) as f32;
                out.push(d1 * q - m1);
                k += 1;
            }
            for l in 0..32 {
                let lo = qs[q_off + l] >> 4;
                let hi = (qh[k % 32] >> (k / 32)) & 1;
                let q = (lo | (hi << 4)) as f32;
                out.push(d2 * q - m2);
                k += 1;
            }

            is += 2;
            q_off += 32;
        }
    }
    Ok(out)
}

// ============================================================
// Q6_K — 6.5625 bpw "super-block" k-quant (standard llama.cpp format)
// ============================================================
//
// Ported from the upstream `dequantize_row_q6_K` (ggml-org/llama.cpp,
// `ggml-quants.c`). Super-block of QK_K=256 weights, 210 bytes:
//
//   ql[128]      — low 4 bits per weight (2 weights/byte)
//   qh[64]       — high 2 bits per weight (4 weights/byte)
//   scales[16]   — i8 scales (one per 16-elem sub-block)
//   d            — f16 super-block scale
//
// The 256-element block is processed in two halves of 128. Within each
// half: 32 iterations of `l`, each emitting 4 weights at positions
// l, l+32, l+64, l+96. The full 6-bit value is `(ql<low/high nibble>)
// | ((qh<2 bits>) << 4)` minus 32 (signed offset), then scaled by
// `d * scales[is + p*2]` where `is = l/16` and `p` is the sub-block.
//
// Tencent ships `token_embd.weight` as Q6_K alongside the STQ1_0
// projections — keeping the embedding at higher precision while the
// computational weights stay 1.3125 bpw is the engineering trade-off
// that lets a 1.8B model fit in 440 MB without tanking the embedding
// quality.

const Q6_K_BLOCK_ELEMS: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210; // 128 + 64 + 16 + 2

pub fn dequantize_q6_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, DequantError> {
    if n_elements % Q6_K_BLOCK_ELEMS != 0 {
        return Err(DequantError::NotMultipleOfBlockSize {
            total: n_elements,
            block: Q6_K_BLOCK_ELEMS,
        });
    }
    let n_blocks = n_elements / Q6_K_BLOCK_ELEMS;
    let need = n_blocks * Q6_K_BLOCK_BYTES;
    if data.len() < need {
        return Err(DequantError::BufferTooSmall { need, have: data.len() });
    }
    let mut out = vec![0f32; n_elements];
    for b in 0..n_blocks {
        let base = b * Q6_K_BLOCK_BYTES;
        let ql = &data[base..base + 128];
        let qh = &data[base + 128..base + 128 + 64];
        let scales = &data[base + 192..base + 192 + 16]; // i8 as u8 bytes
        let d = f16_from_le(&data[base + 208..base + 210]);
        let mut y_off = b * Q6_K_BLOCK_ELEMS;
        let mut ql_off = 0;
        let mut qh_off = 0;
        let mut sc_off = 0;
        for _ in 0..2 {
            // Each half: 128 weights via 32 iterations × 4 sub-blocks.
            for l in 0..32 {
                let is = l / 16;
                let qh_byte = qh[qh_off + l];
                let q1 = ((ql[ql_off + l]      & 0xF) as i16
                    | (((qh_byte      ) & 3) as i16) << 4) - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) as i16
                    | (((qh_byte >> 2) & 3) as i16) << 4) - 32;
                let q3 = ((ql[ql_off + l]      >> 4) as i16
                    | (((qh_byte >> 4) & 3) as i16) << 4) - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) as i16
                    | (((qh_byte >> 6) & 3) as i16) << 4) - 32;
                let s0 = scales[sc_off + is]     as i8 as f32;
                let s1 = scales[sc_off + is + 2] as i8 as f32;
                let s2 = scales[sc_off + is + 4] as i8 as f32;
                let s3 = scales[sc_off + is + 6] as i8 as f32;
                out[y_off + l]      = d * s0 * q1 as f32;
                out[y_off + l + 32] = d * s1 * q2 as f32;
                out[y_off + l + 64] = d * s2 * q3 as f32;
                out[y_off + l + 96] = d * s3 * q4 as f32;
            }
            y_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod q6_k_tests {
    use super::*;

    #[test]
    fn q6_k_zero_weights_decode_to_zero() {
        // ql=0, qh=0 → 6-bit value 0 minus 32 = -32. scales = 0 → result 0.
        let mut buf = vec![0u8; Q6_K_BLOCK_BYTES];
        // d = 1.0 (f16 0x3C00)
        buf[208] = 0x00; buf[209] = 0x3C;
        let v = dequantize_q6_k(&buf, Q6_K_BLOCK_ELEMS).unwrap();
        // All scales 0 → all outputs 0.
        assert!(v.iter().all(|&x| x == 0.0), "all-zero scales → all-zero output");
    }

    #[test]
    fn q6_k_unit_scale_decode_centered_at_minus_32() {
        // ql=0, qh=0 → 6-bit value 0 - 32 = -32. scales = 1 → -32 * d.
        let mut buf = vec![0u8; Q6_K_BLOCK_BYTES];
        for i in 192..208 { buf[i] = 1; } // every i8 scale = +1
        buf[208] = 0x00; buf[209] = 0x3C; // d = 1.0
        let v = dequantize_q6_k(&buf, Q6_K_BLOCK_ELEMS).unwrap();
        assert!(v.iter().all(|&x| x == -32.0),
            "ql=qh=0 with unit scales → -32.0 everywhere");
    }

    #[test]
    fn q6_k_max_weight_decodes_to_plus_31() {
        // ql[0] = 0xFF → both nibbles = 15. qh[0] = 0xFF → both 2-bit fields = 3.
        // 6-bit values: low=15|(3<<4)=63, high=15|(3<<4)=63. Centered: 63-32 = 31.
        let mut buf = vec![0u8; Q6_K_BLOCK_BYTES];
        buf[0] = 0xFF;        // ql[0]: positions 0 and 64 both = 63 (6-bit max)
        buf[128] = 0x33;      // qh[0]: bits 0-1 = 3 → position 0; bits 4-5 = 3 → position 64
        for i in 192..208 { buf[i] = 1; }
        buf[208] = 0x00; buf[209] = 0x3C; // d = 1.0
        let v = dequantize_q6_k(&buf, Q6_K_BLOCK_ELEMS).unwrap();
        assert_eq!(v[0],  31.0, "low nibble + qh bits 0..1 → 31 at position 0");
        assert_eq!(v[64], 31.0, "high nibble + qh bits 4..5 → 31 at position 64");
    }

    #[test]
    fn q6_k_rejects_bad_shape_and_short_buffer() {
        assert!(matches!(
            dequantize_q6_k(&[0u8; 210], 100),
            Err(DequantError::NotMultipleOfBlockSize { block: 256, .. })
        ));
        assert!(matches!(
            dequantize_q6_k(&[0u8; 32], 256),
            Err(DequantError::BufferTooSmall { .. })
        ));
    }
}

// =====================================================================
// helpers
// =====================================================================

fn f16_from_le(bytes: &[u8]) -> f32 {
    let mut b = [0u8; 2];
    b.copy_from_slice(&bytes[..2]);
    f16_to_f32(u16::from_le_bytes(b))
}

// ============================================================
// I2_S — Microsoft bitnet.cpp ternary packing (ggml type 36)
// ============================================================
//
// Layout was derived *empirically* from the byte offsets of
// `bitnet-b1.58-2B-4T-gguf` (not assumed): for a tensor of `n`
// elements the data is `ceil(n/4)` bytes of 2-bit codes followed by
// a single f32 scale, the whole tensor padded to GGUF alignment.
//
//   byte  = [w3:2][w2:2][w1:2][w0:2]   (4 weights/byte, low pair first)
//   code  → ternary value:  0 → 0,  1 → +1,  2 → −1   (bitnet.cpp table)
//   weight = ternary * scale            (scale = trailing f32)
//
// The 2-bit→ternary table and scale placement are the bitnet.cpp
// convention; their *correctness* is gated by the end-to-end
// coherence oracle (a real BitNet prompt must produce coherent
// text), exactly as the TinyLlama path was gated by "Paris".
// Structure (element count, ternary domain, scale application) is
// unit-tested here and now.

pub fn dequantize_i2_s(data: &[u8], n_elements: usize) -> Result<Vec<f32>, DequantError> {
    let code_bytes = (n_elements + 3) / 4;
    let need = code_bytes + 4; // codes + trailing f32 scale
    if data.len() < need {
        return Err(DequantError::BufferTooSmall { need, have: data.len() });
    }
    // Scale is the f32 immediately after the packed codes.
    let s = &data[code_bytes..code_bytes + 4];
    let scale = f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
    let mut out = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let byte = data[i / 4];
        let code = (byte >> ((i % 4) * 2)) & 0b11;
        let t: f32 = match code {
            0 => 0.0,
            1 => 1.0,
            2 => -1.0,
            _ => 0.0, // 3 reserved
        };
        out.push(t * scale);
    }
    Ok(out)
}

// ============================================================
// STQ1_0 "g256" — Tencent/AngelSlim "Sherry" sparse-ternary packing
// ============================================================
//
// Triple-verified (model card; the `block_stq1_0` struct + codebook +
// `dequantize_row_stq1_0` kernel in llama.cpp PR #22836 by Tencent's
// AngelSlim team; direct byte inspection of `Hy-MT1.5-1.8B-1.25bit`):
//
//   block = 256 elements, 42 bytes (in struct field order):
//     [0:32]   qs[32]   — 4-bit slot indices, 2 per byte
//     [32:40]  sign[8]  — 1-bit table-select per 4-weight group
//     [40:42]  f16 d    — block scale
//
// "Sherry" enforces 3:4 sparsity: every 4-lane group has exactly one
// zero; the other three are ±d. That gives 4 (zero-position) × 2³
// (sign) = 32 patterns per group, encoded as a 4-bit slot index
// (`stq1_0_codebook` indexes 16 base patterns, all with the first
// non-zero lane fixed to +1) plus a 1-bit table-select (sign=1 flips
// every non-zero lane). Effective 5 bits per 4-weight group +
// f16/256 scale overhead = **1.3125 bpw**.
//
// Per group `g`:
//   code = (qs[g/2] >> (4*(g&1))) & 0xF
//   sign = (sign[g/8] >> (g%8)) & 1
//   qpack = STQ1_0_CODEBOOK[(sign<<4) | code]
//   for lane p in 0..4:
//     q = (qpack >> (2*p)) & 3    // 0b00=-1, 0b01=0, 0b10=+1
//     w = (q - 1) * d
//
// The ggml type id 42 collides with PrismML's Q2_0 — the loader
// resolves the ambiguity by measuring per-tensor byte stride at parse
// time and rewriting `dtype` accordingly. The reference fast-path
// matmul still does it via f32 dequantisation; a packed
// `MatMulSparseTernary` kernel for the energy/bandwidth win is a
// follow-on (the format admits a `vqtbl2q + vdotq_s32` decode per the
// PR — the SIMD pattern is documented).

const STQ1_0_BLOCK_ELEMS: usize = 256;
const STQ1_0_BLOCK_BYTES: usize = 42; // 32 qs + 8 sign + 2 d

// Verbatim from llama.cpp PR #22836 (Tencent AngelSlim).
// Entry index = (sign << 4) | slot ; each entry is `qpack` (4 lanes ×
// 2 bits, LSB lane 0). 2-bit value: 0b00 = -1, 0b01 = 0, 0b10 = +1.
const STQ1_0_CODEBOOK: [u8; 32] = [
    // sign = 0 (first non-zero lane = +1)
    0xA9, 0x89, 0x29, 0x09, 0xA6, 0x86, 0x26, 0x06,
    0x9A, 0x92, 0x1A, 0x12, 0x6A, 0x62, 0x4A, 0x42,
    // sign = 1 (every non-zero lane negated)
    0x01, 0x21, 0x81, 0xA1, 0x04, 0x24, 0x84, 0xA4,
    0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
];

pub fn dequantize_stq1_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, DequantError> {
    if n_elements % STQ1_0_BLOCK_ELEMS != 0 {
        return Err(DequantError::NotMultipleOfBlockSize {
            total: n_elements,
            block: STQ1_0_BLOCK_ELEMS,
        });
    }
    let n_blocks = n_elements / STQ1_0_BLOCK_ELEMS;
    let need = n_blocks * STQ1_0_BLOCK_BYTES;
    if data.len() < need {
        return Err(DequantError::BufferTooSmall { need, have: data.len() });
    }
    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let base = b * STQ1_0_BLOCK_BYTES;
        let qs = &data[base..base + 32];
        let sign = &data[base + 32..base + 40];
        let d = f16_from_le(&data[base + 40..base + 42]);
        // 256 elems / 4 lanes/group = 64 groups
        for g in 0..(STQ1_0_BLOCK_ELEMS / 4) {
            let code = (qs[g / 2] >> (4 * (g & 1))) & 0x0F;
            let sgn = (sign[g / 8] >> (g % 8)) & 0x01;
            let qpack = STQ1_0_CODEBOOK[((sgn as usize) << 4) | code as usize];
            for p in 0..4 {
                let q = (qpack >> (2 * p)) & 0x03;
                out.push((q as i32 - 1) as f32 * d);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod stq1_0_tests {
    use super::*;

    fn f16_le(v: f32) -> [u8; 2] {
        let bits: u16 = match v {
            x if x == 1.0 => 0x3C00,
            x if x == 0.5 => 0x3800,
            x if x == 2.0 => 0x4000,
            _ => unreachable!(),
        };
        bits.to_le_bytes()
    }

    fn one_block(qs: &[u8; 32], sign: &[u8; 8], d_f16: u16) -> Vec<u8> {
        let mut buf = Vec::with_capacity(STQ1_0_BLOCK_BYTES);
        buf.extend_from_slice(qs);
        buf.extend_from_slice(sign);
        buf.extend_from_slice(&d_f16.to_le_bytes());
        buf
    }

    #[test]
    fn stq1_0_codebook_round_trip_sanity() {
        // For every (slot, sign) pair, decode and check exactly 1 zero
        // per group of 4 and the other 3 are ±1.
        for sign in 0..2u8 {
            for slot in 0..16u8 {
                let qpack = STQ1_0_CODEBOOK[((sign as usize) << 4) | slot as usize];
                let lanes: [i32; 4] = std::array::from_fn(|p| {
                    ((qpack >> (2 * p)) & 0x3) as i32 - 1
                });
                let zeros = lanes.iter().filter(|&&v| v == 0).count();
                let plusminus = lanes.iter().filter(|&&v| v == 1 || v == -1).count();
                assert_eq!(zeros, 1,
                    "(slot {}, sign {}) qpack=0x{:02x} → lanes {:?}: must have exactly 1 zero",
                    slot, sign, qpack, lanes);
                assert_eq!(plusminus, 3, "...and 3 ±1");
            }
        }
    }

    #[test]
    fn stq1_0_block_decodes_one_group() {
        // d = 0.5. All qs nibbles = 0 → every group's slot = 0, sign = 0
        // → qpack = STQ1_0_CODEBOOK[0] = 0xA9 = 0b10_10_10_01.
        // Decoded lanes LSB-first: [01, 10, 10, 10] → q={1,2,2,2} →
        // (q-1)*d = [0, +d, +d, +d].
        let qs = [0u8; 32];
        let sign = [0u8; 8];
        let buf = one_block(&qs, &sign, u16::from_le_bytes(f16_le(0.5)));
        let v = dequantize_stq1_0(&buf, STQ1_0_BLOCK_ELEMS).unwrap();
        for g in 0..64 {
            assert_eq!(v[g * 4],     0.0);
            assert_eq!(v[g * 4 + 1], 0.5);
            assert_eq!(v[g * 4 + 2], 0.5);
            assert_eq!(v[g * 4 + 3], 0.5);
        }
    }

    #[test]
    fn stq1_0_sign_bit_flips_pattern() {
        // d = 1.0; group 0 uses slot=4, then test sign=0 vs sign=1.
        //   slot=4, sign=0 → qpack = STQ1_0_CODEBOOK[4]  = 0xA6 = 0b10_10_01_10 → [+1, 0, +1, +1]
        //   slot=4, sign=1 → qpack = STQ1_0_CODEBOOK[20] = 0x04 = 0b00_00_01_00 → [-1, 0, -1, -1]
        let mut qs = [0u8; 32];
        qs[0] = 0x04; // group 0 slot = 4 (low nibble)
        let mut buf0 = one_block(&qs, &[0u8; 8], u16::from_le_bytes(f16_le(1.0)));
        let v0 = dequantize_stq1_0(&buf0, STQ1_0_BLOCK_ELEMS).unwrap();
        assert_eq!(&v0[..4], &[1.0, 0.0, 1.0, 1.0]);

        let mut sign = [0u8; 8]; sign[0] = 0x01;        // group 0 sign = 1
        buf0 = one_block(&qs, &sign, u16::from_le_bytes(f16_le(1.0)));
        let v1 = dequantize_stq1_0(&buf0, STQ1_0_BLOCK_ELEMS).unwrap();
        assert_eq!(&v1[..4], &[-1.0, 0.0, -1.0, -1.0]);
    }

    #[test]
    fn stq1_0_rejects_bad_shape_and_short_buffer() {
        assert!(matches!(
            dequantize_stq1_0(&[0u8; 42], 100),
            Err(DequantError::NotMultipleOfBlockSize { block: 256, .. })
        ));
        assert!(matches!(
            dequantize_stq1_0(&[0u8; 16], 256),
            Err(DequantError::BufferTooSmall { .. })
        ));
    }

}

// ============================================================
// Q1_0 "g128" — PrismML Bonsai 1-bit packing (ggml type 41)
// ============================================================
//
// Triple-verified (model card; the `block_q1_0` struct + kernel in
// PrismML's llama.cpp fork `PrismML-Eng/llama.cpp@prism`; direct byte
// inspection of `Bonsai-1.7B-Q1_0`):
//
//   block = 128 elements, 18 bytes:
//     [0:2]   f16 scale `d`
//     [2:18]  qs[16]  — LSB-first 1-bit codes, 8 elements/byte
//
//   for element j in block:
//     byte = j / 8 ;  bit = j % 8
//     b    = (qs[byte] >> bit) & 1
//     w    = b ? d : -d           // {+d if bit==1, -d if bit==0}
//
// Even simpler than Q2_0 (no zero, no q==3): two-level sign-only.
// Effective 1.125 bpw. The fast-path bit matmul is therefore a pure
// masked-sign-flip + accumulate — *strictly* no FP multiply per
// weight, with a single f16-scale multiply per 128-block.

const Q1_0_BLOCK_ELEMS: usize = 128;
const Q1_0_BLOCK_BYTES: usize = 18; // 2 (f16 d) + 16 (qs)

pub fn dequantize_q1_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, DequantError> {
    if n_elements % Q1_0_BLOCK_ELEMS != 0 {
        return Err(DequantError::NotMultipleOfBlockSize {
            total: n_elements,
            block: Q1_0_BLOCK_ELEMS,
        });
    }
    let n_blocks = n_elements / Q1_0_BLOCK_ELEMS;
    let need = n_blocks * Q1_0_BLOCK_BYTES;
    if data.len() < need {
        return Err(DequantError::BufferTooSmall { need, have: data.len() });
    }
    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let base = b * Q1_0_BLOCK_BYTES;
        let d = f16_from_le(&data[base..base + 2]);
        let qs = &data[base + 2..base + Q1_0_BLOCK_BYTES];
        for j in 0..Q1_0_BLOCK_ELEMS {
            let bit = (qs[j / 8] >> (j % 8)) & 1;
            out.push(if bit == 1 { d } else { -d });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod q1_0_tests {
    use super::*;

    fn f16_le(v: f32) -> [u8; 2] {
        let bits: u16 = match v {
            x if x == 1.0 => 0x3C00,
            x if x == 0.5 => 0x3800,
            x if x == 2.0 => 0x4000,
            _ => unreachable!(),
        };
        bits.to_le_bytes()
    }

    #[test]
    fn q1_0_block_layout_bits_and_scale() {
        // One 128-elem block, d = 0.5, byte0 = 0b1010_0101 = 0xA5
        //   LSB-first bits: 1,0,1,0,0,1,0,1 → +d, -d, +d, -d, -d, +d, -d, +d
        let mut buf = Vec::new();
        buf.extend_from_slice(&f16_le(0.5));
        buf.push(0xA5);
        buf.extend(std::iter::repeat(0u8).take(15)); // rest = bit 0 → -0.5
        let v = dequantize_q1_0(&buf, Q1_0_BLOCK_ELEMS).unwrap();
        assert_eq!(&v[0..8], &[0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5]);
        assert!(v[8..].iter().all(|&x| x == -0.5));
        assert_eq!(v.len(), 128);
    }

    #[test]
    fn q1_0_multi_block_independent_scales() {
        let mut buf = Vec::new();
        // block0: d=1.0, all bits 1 → all +1.0
        buf.extend_from_slice(&f16_le(1.0));
        buf.extend(std::iter::repeat(0xFFu8).take(16));
        // block1: d=2.0, all bits 0 → all -2.0
        buf.extend_from_slice(&f16_le(2.0));
        buf.extend(std::iter::repeat(0x00u8).take(16));
        let v = dequantize_q1_0(&buf, 2 * Q1_0_BLOCK_ELEMS).unwrap();
        assert!(v[..128].iter().all(|&x| x == 1.0));
        assert!(v[128..].iter().all(|&x| x == -2.0));
    }

    #[test]
    fn q1_0_rejects_bad_shape_and_short_buffer() {
        assert!(matches!(
            dequantize_q1_0(&[0u8; 18], 100),
            Err(DequantError::NotMultipleOfBlockSize { block: 128, .. })
        ));
        assert!(matches!(
            dequantize_q1_0(&[0u8; 4], 128),
            Err(DequantError::BufferTooSmall { .. })
        ));
    }
}

// ============================================================
// Q2_0 "g128" — PrismML Bonsai ternary packing (ggml type 42)
// ============================================================
//
// Triple-verified (model card spec; the `block_q2_0` struct +
// `dequantize_row_q2_0` kernel in PrismML's llama.cpp fork
// `PrismML-Eng/llama.cpp@prism`; and direct byte inspection of
// `Ternary-Bonsai-1.7B-Q2_0`):
//
//   block = 128 elements, 34 bytes:
//     [0:2]   f16 scale `d`
//     [2:34]  qs[32]  — LSB-first 2-bit codes, 4 elements/byte
//
//   for element j in block:
//     byte = j / 4 ;  bit = (j % 4) * 2
//     q    = (qs[byte] >> bit) & 0b11        // {0,1,2,3}
//     w    = (q as i32 - 1) * d              // {-1,0,+1,+2} * d
//
// Ternary weights only ever use {0,1,2} (q=3 → +2·d is reserved
// and confirmed absent in the real Bonsai tensors). Unlike I2_S
// (one trailing f32 scale for the whole tensor), Q2_0 carries an
// independent f16 scale per 128-element block. Structure is
// unit-tested here; end-to-end correctness is gated by the
// coherence oracle (Bonsai must emit coherent text), exactly as
// TinyLlama was gated by "Paris".

const Q2_0_BLOCK_ELEMS: usize = 128;
const Q2_0_BLOCK_BYTES: usize = 34; // 2 (f16 d) + 32 (qs)

pub fn dequantize_q2_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, DequantError> {
    if n_elements % Q2_0_BLOCK_ELEMS != 0 {
        return Err(DequantError::NotMultipleOfBlockSize {
            total: n_elements,
            block: Q2_0_BLOCK_ELEMS,
        });
    }
    let n_blocks = n_elements / Q2_0_BLOCK_ELEMS;
    let need = n_blocks * Q2_0_BLOCK_BYTES;
    if data.len() < need {
        return Err(DequantError::BufferTooSmall { need, have: data.len() });
    }
    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let base = b * Q2_0_BLOCK_BYTES;
        let d = f16_from_le(&data[base..base + 2]);
        let qs = &data[base + 2..base + Q2_0_BLOCK_BYTES];
        for j in 0..Q2_0_BLOCK_ELEMS {
            let q = (qs[j / 4] >> ((j % 4) * 2)) & 0b11;
            out.push((q as i32 - 1) as f32 * d);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod q2_0_tests {
    use super::*;

    fn f16_le(v: f32) -> [u8; 2] {
        // minimal round-trippable encodings for the small exact values
        // used in tests (1.0, 0.5, 2.0) — exercises f16_from_le too.
        let bits: u16 = match v {
            x if x == 1.0 => 0x3C00,
            x if x == 0.5 => 0x3800,
            x if x == 2.0 => 0x4000,
            _ => unreachable!("test uses only 1.0/0.5/2.0"),
        };
        bits.to_le_bytes()
    }

    #[test]
    fn q2_0_block_layout_codes_and_scale() {
        // One 128-elem block, scale d = 0.5.
        // First byte packs elements j=0..3, LSB-first:
        //   j0 q=0 (-1), j1 q=1 (0), j2 q=2 (+1), j3 q=3 (+2)
        //   byte = (3<<6)|(2<<4)|(1<<2)|0 = 0b11_10_01_00 = 0xE4
        let mut buf = Vec::new();
        buf.extend_from_slice(&f16_le(0.5));
        buf.push(0xE4);
        buf.extend(std::iter::repeat(0u8).take(31)); // rest of qs (code 0 → -1)
        let v = dequantize_q2_0(&buf, Q2_0_BLOCK_ELEMS).unwrap();
        assert_eq!(&v[0..4], &[-0.5, 0.0, 0.5, 1.0]);
        // every remaining element is code 0 → (0-1)*0.5 = -0.5
        assert!(v[4..].iter().all(|&x| x == -0.5));
        assert_eq!(v.len(), 128);
    }

    #[test]
    fn q2_0_multi_block_independent_scales() {
        // Two blocks with different scales prove per-block scaling.
        let mut buf = Vec::new();
        // block 0: d=1.0, all code 2 (+1) → all +1.0
        buf.extend_from_slice(&f16_le(1.0));
        buf.extend(std::iter::repeat(0xAAu8).take(32)); // 0b10_10_10_10 → all q=2
        // block 1: d=2.0, all code 0 (-1) → all -2.0
        buf.extend_from_slice(&f16_le(2.0));
        buf.extend(std::iter::repeat(0x00u8).take(32));
        let v = dequantize_q2_0(&buf, 2 * Q2_0_BLOCK_ELEMS).unwrap();
        assert!(v[..128].iter().all(|&x| x == 1.0));
        assert!(v[128..].iter().all(|&x| x == -2.0));
    }

    #[test]
    fn q2_0_rejects_bad_shape_and_short_buffer() {
        assert!(matches!(
            dequantize_q2_0(&[0u8; 34], 100),
            Err(DequantError::NotMultipleOfBlockSize { block: 128, .. })
        ));
        assert!(matches!(
            dequantize_q2_0(&[0u8; 10], 128),
            Err(DequantError::BufferTooSmall { .. })
        ));
    }
}

#[cfg(test)]
mod i2s_tests {
    use super::*;

    #[test]
    fn i2s_layout_and_scale_are_applied() {
        // 4 weights in one byte: codes [w0=1(+1), w1=2(-1), w2=0(0), w3=1(+1)]
        // packed low-pair-first: 0b01_00_10_01 = 0x49
        let scale: f32 = 0.5;
        let mut buf = vec![0x49u8];
        buf.extend_from_slice(&scale.to_le_bytes());
        let v = dequantize_i2_s(&buf, 4).unwrap();
        assert_eq!(v, vec![0.5, -0.5, 0.0, 0.5]);
    }

    #[test]
    fn i2s_partial_last_byte_and_buffer_guard() {
        // n=2 → 1 code byte + f32 scale. codes w0=2(-1), w1=1(+1): 0b00_00_01_10=0x06
        let scale: f32 = 2.0;
        let mut buf = vec![0x06u8];
        buf.extend_from_slice(&scale.to_le_bytes());
        let v = dequantize_i2_s(&buf, 2).unwrap();
        assert_eq!(v, vec![-2.0, 2.0]);
        // too-small buffer rejected, not panicked
        assert!(matches!(
            dequantize_i2_s(&[0x00], 4),
            Err(DequantError::BufferTooSmall { .. })
        ));
    }
}
