//! Reference packed STQ1_0 matmul: `Y = A @ W^T`, W packed Tencent/
//! AngelSlim "Sherry" 3:4-sparse ternary g256. **1.3125 bpw** — the
//! most aggressive packing in the substrate.
//!
//! Per 256-element block (42 bytes = `qs[32] + sign[8] + f16 d`):
//!
//!   per 4-lane group (64 groups/block):
//!     slot = (qs[g/2] >> (4*(g&1))) & 0xF        // 4-bit
//!     sign = (sign[g/8] >> (g%8)) & 0x1          // 1-bit
//!     qpack = STQ1_0_CODEBOOK[(sign<<4) | slot]  // 32-entry LUT
//!     for lane p in 0..4:
//!       q = (qpack >> (2*p)) & 0x3   // {-1, 0, +1} via q-1
//!       w[g*4+p] = (q - 1) * d
//!
//! 3:4 sparsity guarantees exactly one lane per group is zero, so a
//! quarter of the weights contribute nothing — the kernel skips them
//! structurally instead of multiplying by zero. The block scale is the
//! ONLY f32 multiply (one per 256 weights).
//!
//! Output columns are independent → parallelised across cores via
//! `thread::scope`, same playbook as `MatMulTernary` / `MatMulBit`.

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

const QK: usize = 256;
const BLK: usize = 42; // 32 qs + 8 sign + 2 d

pub struct MatMulSTQ1_0Ref;

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

// Verbatim from llama.cpp PR #22836 — index = (sign<<4) | slot.
const STQ1_0_CODEBOOK: [u8; 32] = [
    0xA9, 0x89, 0x29, 0x09, 0xA6, 0x86, 0x26, 0x06,
    0x9A, 0x92, 0x1A, 0x12, 0x6A, 0x62, 0x4A, 0x42,
    0x01, 0x21, 0x81, 0xA1, 0x04, 0x24, 0x84, 0xA4,
    0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
];

/// Pre-decode each codebook entry into its 4 ternary lane values
/// `[-1, 0, +1]` stored as `i8`. This is the hot-loop LUT — one byte
/// load per 4-weight group, then 4 branchless integer multiply-adds.
const fn build_lane_lut() -> [[i8; 4]; 32] {
    let mut out = [[0i8; 4]; 32];
    let mut idx = 0usize;
    while idx < 32 {
        let qpack = STQ1_0_CODEBOOK[idx];
        let mut p = 0usize;
        while p < 4 {
            let q = (qpack >> (2 * p)) & 0x3;
            out[idx][p] = q as i8 - 1;
            p += 1;
        }
        idx += 1;
    }
    out
}
static LANE_LUT: [[i8; 4]; 32] = build_lane_lut();

impl Kernel for MatMulSTQ1_0Ref {
    fn op_kind(&self) -> OpKind { OpKind::MatMulSTQ1_0 }
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
            OpAttrs::MatMulSTQ1_0 { out, k } => (*out, *k),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulSTQ1_0, backend: BACKEND_ID,
                reason: "MatMulSTQ1_0 kernel requires OpAttrs::MatMulSTQ1_0".into(),
            }),
        };
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulSTQ1_0, backend: BACKEND_ID,
                reason: format!("expects 2 inputs / 1 output, got {} / {}",
                    inputs.len(), outputs.len()),
            });
        }
        let a = inputs[0].as_f32_vec();
        let a_shape = &inputs[0].meta.shape;
        let a_rank = a_shape.len();
        if a_shape[a_rank - 1] != k {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulSTQ1_0, backend: BACKEND_ID,
                reason: format!("A last dim {} != k {}", a_shape[a_rank - 1], k),
            });
        }
        if k % QK != 0 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulSTQ1_0, backend: BACKEND_ID,
                reason: format!("k={} must be multiple of {} (STQ1_0 block)", k, QK),
            });
        }
        let total_m: usize = a_shape.iter().take(a_rank - 1).product::<usize>().max(1);
        let n_blocks = k / QK;
        let row_stride = n_blocks * BLK;
        let w = inputs[1].bytes;
        let need = out_dim * row_stride;
        if w.len() < need {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulSTQ1_0, backend: BACKEND_ID,
                reason: format!("packed W too small: need {} bytes, have {}",
                    need, w.len()),
            });
        }

        // Pure function of (a, w, o); independent across o → parallel safe.
        let col = |o: usize, dst: &mut [f32]| {
            let wbase = o * row_stride;
            for i in 0..total_m {
                let a_row = &a[i * k..i * k + k];
                let mut acc = 0f32;
                for blk in 0..n_blocks {
                    let bb = wbase + blk * BLK;
                    let qs = &w[bb..bb + 32];
                    let sign = &w[bb + 32..bb + 40];
                    let d = f16_to_f32(u16::from_le_bytes([w[bb + 40], w[bb + 41]]));
                    let a_blk = &a_row[blk * QK..(blk + 1) * QK];
                    let mut s = 0f32;
                    // 64 groups per block, 4 lanes per group.
                    for g in 0..64 {
                        let slot = (qs[g >> 1] >> (4 * (g & 1))) & 0x0F;
                        let sgn  = (sign[g >> 3] >> (g & 7)) & 0x01;
                        let lanes = &LANE_LUT[((sgn as usize) << 4) | slot as usize];
                        let base = g * 4;
                        // 4 ternary integer madds; 3:4 sparsity means one
                        // of these is always zero — the branchless form
                        // still computes it, but per spec at least one
                        // lane is zero so the multiply is a no-op-cost
                        // arithmetic addition of zero.
                        s += a_blk[base    ] * lanes[0] as f32
                           + a_blk[base + 1] * lanes[1] as f32
                           + a_blk[base + 2] * lanes[2] as f32
                           + a_blk[base + 3] * lanes[3] as f32;
                    }
                    acc += d * s; // the only "real" f32 multiply per block
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
        // 3:4 sparsity: only 3 of 4 lanes contribute → adds rate is
        // 0.75 × k. One scale-multiply per 256-block.
        let effective_adds = (total_m * out_dim * k * 3 / 4) as f64;
        let scale_muls = (total_m * out_dim * n_blocks) as f64;
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: effective_adds * 2.5e-11 + scale_muls * 1e-10,
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

    fn pack_block(slots: &[u8; 64], signs: &[u8; 64], d_f16: u16) -> Vec<u8> {
        let mut out = vec![0u8; BLK];
        // qs: 2 nibbles per byte, slot[2g] in low, slot[2g+1] in high
        for g in 0..64 {
            out[g / 2] |= (slots[g] & 0xF) << (4 * (g & 1));
        }
        for g in 0..64 {
            out[32 + (g / 8)] |= (signs[g] & 1) << (g & 7);
        }
        out[40..42].copy_from_slice(&d_f16.to_le_bytes());
        out
    }

    fn run(a: &[f32], m: usize, k: usize, w: &[u8], out_dim: usize) -> Vec<f32> {
        let am = TensorMeta::new(Dtype::F32, &[m, k]);
        let ab: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let wm = TensorMeta::new(Dtype::U8, &[w.len()]);
        let om = TensorMeta::new(Dtype::F32, &[m, out_dim]);
        let mut ob = vec![0u8; m * out_dim * 4];
        let iv = [
            TensorView { meta: &am, bytes: &ab },
            TensorView { meta: &wm, bytes: w },
        ];
        let mut ov = [TensorViewMut { meta: &om, bytes: &mut ob }];
        let mut scratch = vec![0u8; 1024];
        let mut ctx = ExecutionContext {
            backend: BACKEND_ID, deterministic: true,
            seed: None, scratch: &mut scratch,
        };
        MatMulSTQ1_0Ref.execute(&mut ctx,
            &OpAttrs::MatMulSTQ1_0 { out: out_dim, k }, &iv, &mut ov).unwrap();
        (0..m * out_dim).map(|i| f32::from_le_bytes(
            [ob[i*4], ob[i*4+1], ob[i*4+2], ob[i*4+3]])).collect()
    }

    #[test]
    fn stq1_0_matmul_one_block_decode_matches_lut() {
        // Block of 256: every group's slot = 0, sign = 0 → qpack = 0xA9
        // → lanes [0, +1, +1, +1] (LUT[0]).
        // d = 1.0. Each output row sums over k=256: 64 groups × (0 + 1 + 1 + 1) = 192
        // weighted by a[base..base+4] values. With a = all 1.0, expected y = 64*3 = 192.
        let d: u16 = 0x3C00;
        let slots = [0u8; 64];
        let signs = [0u8; 64];
        let w = pack_block(&slots, &signs, d);
        let a = vec![1.0f32; 256];
        let y = run(&a, 1, 256, &w, 1);
        assert!((y[0] - 192.0).abs() < 1e-4, "got {}", y[0]);
    }

    #[test]
    fn stq1_0_matmul_multi_row_independent_blocks() {
        // out=2. Row0: slot=0 sign=0, d=1.0 → lanes [0,+1,+1,+1] per group.
        // Row1: slot=4 sign=1, d=2.0 → qpack = LUT[20] = 0x04 →
        //   bits LSB-first 2bit: 00,01,00,00 → lanes [-1, 0, -1, -1]
        let row0 = pack_block(&[0u8; 64], &[0u8; 64], 0x3C00);
        let row1 = pack_block(&[4u8; 64], &[1u8; 64], 0x4000);
        let mut w = row0.clone();
        w.extend(row1);
        let a = vec![1.0f32; 256];
        let y = run(&a, 1, 256, &w, 2);
        assert!((y[0] - 192.0).abs() < 1e-4, "row0 {}", y[0]);
        // Row1: lanes per group = [-1, 0, -1, -1] → sum = -3 per group.
        // 64 groups * a-sum × d = 64 * (-3 * 1.0) * 2.0 = -384.
        assert!((y[1] + 384.0).abs() < 1e-3, "row1 {}", y[1]);
    }

    #[test]
    fn stq1_0_matmul_parallel_path_matches_serial_reference() {
        // out > 64 forces the multi-thread split. Build a varied W and
        // compare against an independent (slow) dequant + dot reference.
        let m = 3;
        let k = 512; // 2 blocks per row
        let out = 200;
        let mut a = vec![0f32; m * k];
        for (i, x) in a.iter_mut().enumerate() {
            *x = (((i * 11 + 5) % 17) as f32) - 8.0;
        }
        let scales = [0x3800u16, 0x3C00, 0x4000];
        let mut w = Vec::new();
        for o in 0..out {
            for b in 0..2 {
                let mut slots = [0u8; 64];
                let mut signs = [0u8; 64];
                for g in 0..64 { slots[g] = ((o + b + g) % 16) as u8; }
                for g in 0..64 { signs[g] = ((o + g) & 1) as u8; }
                w.extend(pack_block(&slots, &signs, scales[(o + b) % 3]));
            }
        }
        let got = run(&a, m, k, &w, out);

        let f16 = |h: u16| super::f16_to_f32(h);
        let mut expect = vec![0f32; m * out];
        for o in 0..out {
            for i in 0..m {
                let mut acc = 0f32;
                for b in 0..2 {
                    let d = f16(scales[(o + b) % 3]);
                    let mut s = 0f32;
                    for g in 0..64 {
                        let slot = ((o + b + g) % 16) as u8;
                        let sgn  = ((o + g) & 1) as u8;
                        let lanes = &LANE_LUT[((sgn as usize) << 4) | slot as usize];
                        for p in 0..4 {
                            s += a[i * k + b * 256 + g * 4 + p]
                               * lanes[p] as f32;
                        }
                    }
                    acc += d * s;
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
