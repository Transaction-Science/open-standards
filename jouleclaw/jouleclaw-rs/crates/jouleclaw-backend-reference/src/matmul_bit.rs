//! Reference 1-bit matmul: `Y = A @ W^T`, W packed PrismML Q1_0 g128.
//!
//! Stricter version of [`crate::matmul_ternary`] — every weight is
//! ±d (no zero, no 4th code point), so the inner product is pure
//! sign-flip + accumulate:
//!
//!   per 128-element block (18 bytes = f16 d + 16 code bytes):
//!     s = Σ ( +a[k] where bit==1
//!            -a[k] where bit==0 )
//!     acc += d * s        // ONE f32 multiply per 128 weights
//!
//! Like the ternary kernel: deterministic and bit-reproducible with a
//! fixed reduction order, output columns parallelised across cores via
//! `thread::scope`, aarch64 path uses a NEON LUT + FMA per code byte
//! (16 lanes per byte = 2 × `float32x4_t`), scalar elsewhere stays the
//! numeric reference. Cost-modelled at the ternary rates because the
//! arithmetic is the same shape (the 1-bit format has even better
//! memory bandwidth, but charging less than the ternary path would be
//! a separate, calibrated change).

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

const QK: usize = 128;
const BLK: usize = 18; // 2 (f16 d) + 16 (qs)

pub struct MatMulBitRef;

#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let s = (h >> 15) & 1;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;
    let v = if e == 0 {
        (m as f32 / 1024.0) * 2f32.powi(-14)
    } else if e == 31 {
        if m == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15)
    };
    if s == 1 { -v } else { v }
}

/// `byte → [s0,…,s7]` where `sj = +1.0` if `bitj==1` else `-1.0`
/// (LSB first). Lets the NEON path multiply-accumulate 8 weights per
/// byte with two vector loads + two FMAs.
const fn build_bit_lut() -> [[f32; 8]; 256] {
    let mut t = [[0.0f32; 8]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut j = 0usize;
        while j < 8 {
            t[b][j] = if (b >> j) & 1 == 1 { 1.0 } else { -1.0 };
            j += 1;
        }
        b += 1;
    }
    t
}
static BIT_LUT: [[f32; 8]; 256] = build_bit_lut();

/// Dot of one full 128-element Q1_0 block: `Σ (sign_bitⱼ)·a[j]`, scale
/// applied by the caller. `a` is exactly 128 f32; `codes` is exactly
/// 16 bytes. aarch64 → NEON; everything else → branchless scalar
/// (the numeric reference).
#[cfg(target_arch = "aarch64")]
#[inline]
fn block_dot_bit(a: &[f32], codes: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    // SAFETY: NEON/ASIMD is baseline on aarch64; `a` ≥ 128 f32 and
    // `codes` ≥ 16 bytes per the full-block precondition.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let ap = a.as_ptr();
        for byte_idx in 0..16 {
            let base = byte_idx * 8;
            let lut = BIT_LUT[codes[byte_idx] as usize].as_ptr();
            // Lanes 0..3 then 4..7 from the same byte.
            let av0 = vld1q_f32(ap.add(base));
            let av1 = vld1q_f32(ap.add(base + 4));
            let m0  = vld1q_f32(lut);
            let m1  = vld1q_f32(lut.add(4));
            acc = vfmaq_f32(acc, av0, m0);
            acc = vfmaq_f32(acc, av1, m1);
        }
        vaddvq_f32(acc)
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn block_dot_bit(a: &[f32], codes: &[u8]) -> f32 {
    const SIGN: u32 = 0x8000_0000;
    let mut s = 0f32;
    for byte_idx in 0..16 {
        let cb = codes[byte_idx];
        let base = byte_idx * 8;
        for j in 0..8 {
            let av = a[base + j];
            let ab = av.to_bits();
            // bit==1 → +av (mask 0xFFFF_FFFF → contribute ab)
            // bit==0 → -av (mask 0xFFFF_FFFF → contribute ab^SIGN)
            let bit_set = ((cb >> j) & 1) as u32;
            let pos_mask = bit_set.wrapping_neg();
            let neg_mask = (!pos_mask) & u32::MAX;
            let pos = pos_mask & ab;
            let neg = neg_mask & (ab ^ SIGN);
            s += f32::from_bits(pos) + f32::from_bits(neg);
        }
    }
    s
}

impl Kernel for MatMulBitRef {
    fn op_kind(&self) -> OpKind { OpKind::MatMulBit }
    fn backend(&self) -> BackendId { BACKEND_ID }
    fn determinism(&self) -> DeterminismClass { DeterminismClass::Deterministic }

    fn execute(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        attrs: &OpAttrs,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorViewMut<'_>],
    ) -> Result<KernelResult, ExecutionError> {
        let start = Instant::now();
        let (out_dim, k) = match attrs {
            OpAttrs::MatMulBit { out, k } => (*out, *k),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulBit, backend: BACKEND_ID,
                reason: "MatMulBit kernel requires OpAttrs::MatMulBit".into(),
            }),
        };
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulBit, backend: BACKEND_ID,
                reason: format!("expects 2 inputs / 1 output, got {} / {}",
                    inputs.len(), outputs.len()),
            });
        }

        let a = inputs[0].as_f32_vec();
        let a_shape = &inputs[0].meta.shape;
        let a_rank = a_shape.len();
        if a_shape[a_rank - 1] != k {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulBit, backend: BACKEND_ID,
                reason: format!("A last dim {} != k {}", a_shape[a_rank - 1], k),
            });
        }
        let total_m: usize = a_shape.iter().take(a_rank - 1).product::<usize>().max(1);

        let n_blocks = (k + QK - 1) / QK;
        let row_stride = n_blocks * BLK;
        let w = inputs[1].bytes;
        let need = out_dim * row_stride;
        if w.len() < need {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulBit, backend: BACKEND_ID,
                reason: format!("packed W too small: need {} bytes, have {}",
                    need, w.len()),
            });
        }

        let col = |o: usize, dst: &mut [f32]| {
            let wbase = o * row_stride;
            for i in 0..total_m {
                let a_row = &a[i * k..i * k + k];
                let mut acc = 0f32;
                for blk in 0..n_blocks {
                    let bb = wbase + blk * BLK;
                    let d = f16_to_f32(u16::from_le_bytes([w[bb], w[bb + 1]]));
                    let codes = &w[bb + 2..bb + BLK]; // exactly 16 bytes
                    let k0 = blk * QK;
                    let kn = (k0 + QK).min(k);
                    let blk_len = kn - k0;
                    let a_blk = &a_row[k0..kn];
                    let s = if blk_len == QK {
                        block_dot_bit(a_blk, codes)
                    } else {
                        // Partial trailing block (rare; kept correct).
                        let mut acc_p = 0f32;
                        for local in 0..blk_len {
                            let bit = (codes[local / 8] >> (local % 8)) & 1;
                            acc_p += if bit == 1 { a_blk[local] } else { -a_blk[local] };
                        }
                        acc_p
                    };
                    acc += d * s;
                }
                dst[i] = acc;
            }
        };

        let mut c_om = vec![0f32; out_dim * total_m];
        let nthreads = std::thread::available_parallelism()
            .map(|n| n.get()).unwrap_or(1).min(out_dim.max(1));

        if nthreads <= 1 || out_dim < 64 {
            for o in 0..out_dim {
                col(o, &mut c_om[o * total_m..(o + 1) * total_m]);
            }
        } else {
            let per = out_dim.div_ceil(nthreads);
            let col_ref = &col;
            std::thread::scope(|sc| {
                for (t, chunk) in c_om.chunks_mut(per * total_m).enumerate() {
                    let o0 = t * per;
                    sc.spawn(move || {
                        let cols = chunk.len() / total_m;
                        for j in 0..cols {
                            col_ref(o0 + j,
                                &mut chunk[j * total_m..(j + 1) * total_m]);
                        }
                    });
                }
            });
        }

        let mut c = vec![0f32; total_m * out_dim];
        for o in 0..out_dim {
            for i in 0..total_m {
                c[i * out_dim + o] = c_om[o * total_m + i];
            }
        }
        outputs[0].write_f32(&c);

        let elapsed = start.elapsed();
        // 1-bit memory traffic is even better than ternary's 2-bit, but
        // we charge the same per-add rate — calibration (R37+) will
        // separate these honestly. One scale-multiply per 128-block.
        let adds = (total_m * out_dim * k) as f64;
        let scale_muls = (total_m * out_dim * n_blocks) as f64;
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: adds * 2.5e-11 + scale_muls * 1e-10,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: (a.len() * 4 + need) as u64,
            bytes_written: (c.len() * 4) as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_core::tensor::{Dtype, TensorMeta};

    fn pack_row(bits: &[i8], d_f16: u16) -> Vec<u8> {
        // bits[i] ∈ {-1, +1}: +1 → bit 1, -1 → bit 0.
        let k = bits.len();
        let n_blocks = (k + QK - 1) / QK;
        let mut out = vec![0u8; n_blocks * BLK];
        for blk in 0..n_blocks {
            let bb = blk * BLK;
            out[bb..bb + 2].copy_from_slice(&d_f16.to_le_bytes());
            for local in 0..QK {
                let idx = blk * QK + local;
                if idx >= k { break; }
                if bits[idx] == 1 {
                    out[bb + 2 + (local >> 3)] |= 1 << (local & 7);
                }
            }
        }
        out
    }

    fn run(a: &[f32], m: usize, k: usize, w: &[u8], out: usize) -> Vec<f32> {
        let a_meta = TensorMeta::new(Dtype::F32, &[m, k]);
        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_meta = TensorMeta::new(Dtype::U8, &[w.len()]);
        let o_meta = TensorMeta::new(Dtype::F32, &[m, out]);
        let mut o_bytes = vec![0u8; m * out * 4];
        let iv = [
            TensorView { meta: &a_meta, bytes: &a_bytes },
            TensorView { meta: &w_meta, bytes: w },
        ];
        let mut ov = [TensorViewMut { meta: &o_meta, bytes: &mut o_bytes }];
        let mut scratch = vec![0u8; 1024];
        let mut ctx = ExecutionContext {
            backend: BACKEND_ID, deterministic: true,
            seed: None, scratch: &mut scratch,
        };
        MatMulBitRef.execute(&mut ctx,
            &OpAttrs::MatMulBit { out, k }, &iv, &mut ov).unwrap();
        (0..m * out).map(|i| f32::from_le_bytes(
            [o_bytes[i*4], o_bytes[i*4+1], o_bytes[i*4+2], o_bytes[i*4+3]]))
            .collect()
    }

    #[test]
    fn bit_matches_explicit_dot() {
        // d=0.5, row0 = [+1,-1,+1,-1] (rest +1) over k=128.
        let d: u16 = 0x3800;
        let mut v = vec![1i8; 128];
        v[1] = -1; v[3] = -1;
        let w = pack_row(&v, d);
        let mut a = vec![0f32; 128];
        a[0] = 2.0; a[1] = 3.0; a[2] = 4.0; a[3] = 5.0;
        // Σ = d*( +a0 -a1 +a2 -a3 +0...0 ) = 0.5*(2-3+4-5) = -1.0
        let y = run(&a, 1, 128, &w, 1);
        assert!((y[0] - (-1.0)).abs() < 1e-6, "got {}", y[0]);
    }

    #[test]
    fn bit_parallel_path_is_correct() {
        // out=200 forces threading. Random-ish but deterministic.
        let m = 2;
        let k = 256;
        let out = 200;
        let mut a = vec![0f32; m * k];
        for (i, x) in a.iter_mut().enumerate() {
            *x = (((i * 7 + 1) % 13) as f32) - 6.0;
        }
        let scales = [0x3800u16, 0x3C00, 0x4000];
        let mut w = Vec::new();
        for o in 0..out {
            let mut row = vec![1i8; k];
            for (j, r) in row.iter_mut().enumerate() {
                *r = if (o + j) % 2 == 0 { 1 } else { -1 };
            }
            w.extend(pack_row(&row, scales[o % 3]));
        }
        let got = run(&a, m, k, &w, out);

        // Serial reference: dequantise + plain dot.
        let mut expect = vec![0f32; m * out];
        for o in 0..out {
            let d = super::f16_to_f32(scales[o % 3]);
            for i in 0..m {
                let mut acc = 0f32;
                for j in 0..k {
                    let sign = if (o + j) % 2 == 0 { 1.0 } else { -1.0 };
                    acc += a[i * k + j] * sign * d;
                }
                expect[i * out + o] = acc;
            }
        }
        for idx in 0..m * out {
            assert!((got[idx] - expect[idx]).abs() < 1e-3,
                "idx {} got {} expect {}", idx, got[idx], expect[idx]);
        }
    }
}
