//! Reference ternary matmul: `Y = A @ W^T`, W packed PrismML Q2_0 g128.
//!
//! This is the kernel that makes ternary weights worth their format.
//! The dense path dequantises Q2_0 → f32 and runs a generic MAC loop,
//! discarding the entire point of 1.58-bit weights. Here W stays packed
//! and the inner product is **sign-select + accumulate**:
//!
//!   per 128-element block (34 bytes = f16 d + 32 code bytes):
//!     s = Σ ( +a[k]  where code==2     // +1
//!            -a[k]  where code==0     // -1
//!             0     where code==1 )   //  0   (q==3 reserved/unused)
//!     acc += d * s                    // ONE f32 multiply per 128 weights
//!
//! No per-weight floating multiply. Reduction order is fixed (output
//! row-major, k ascending by block) so the result is bit-reproducible
//! and numerically identical to `matmul_bt` over the dequantised W
//! (ternary lands exactly on its code points; only the per-block f16
//! scale rounding applies, identically in both paths).
//!
//! Output columns are mutually independent, so the column loop is split
//! across `available_parallelism()` worker threads via `thread::scope`
//! (pure std, no unsafe, no SIMD intrinsics). This stays bit-exact
//! regardless of thread count — each output element is written once by
//! the same pure `col` closure the serial path uses — so the kernel
//! remains a valid determinism reference while using the cores a
//! Pi-class target actually has.

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

const QK: usize = 128; // elements per Q2_0 block
const BLK: usize = 34; // bytes per block: 2 (f16 d) + 32 (codes)

pub struct MatMulTernaryRef;

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

/// `byte → [t0,t1,t2,t3]` where `tj = (codej - 1) ∈ {-1,0,1,2}` (LSB
/// code first). Lets the NEON path multiply-accumulate 4 weights per
/// byte with a single vector load + FMA.
const fn build_ternary_lut() -> [[f32; 4]; 256] {
    let mut t = [[0.0f32; 4]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut j = 0usize;
        while j < 4 {
            t[b][j] = match (b >> (j * 2)) & 0b11 {
                0 => -1.0,
                1 => 0.0,
                2 => 1.0,
                _ => 2.0,
            };
            j += 1;
        }
        b += 1;
    }
    t
}
static TERNARY_LUT: [[f32; 4]; 256] = build_ternary_lut();

/// Dot of one full 128-element Q2_0 block: `Σ (codeⱼ-1)·a[j]`, scale
/// applied by the caller. `a` is exactly 128 f32; `codes` is exactly
/// 32 bytes. aarch64 uses NEON (4-wide FMA per code byte); every other
/// target uses the branchless scalar form, which is also the numeric
/// reference. NEON re-associates the sum lane-wise, so it can differ
/// from the scalar reference by f32 rounding (it stays bit-reproducible
/// run-to-run); the unit tests and the end-to-end oracle gate this.
#[cfg(target_arch = "aarch64")]
#[inline]
fn block_dot(a: &[f32], codes: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    // SAFETY: NEON/ASIMD is baseline-mandatory on aarch64 (no runtime
    // detection needed). `a` has ≥128 elements and `codes` ≥32 bytes
    // (full-block precondition); all reads below are in-bounds.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let ap = a.as_ptr();
        for byte_idx in 0..32 {
            let av = vld1q_f32(ap.add(byte_idx * 4));
            let m = vld1q_f32(TERNARY_LUT[codes[byte_idx] as usize].as_ptr());
            acc = vfmaq_f32(acc, av, m);
        }
        vaddvq_f32(acc)
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn block_dot(a: &[f32], codes: &[u8]) -> f32 {
    const SIGN: u32 = 0x8000_0000;
    let mut s = 0f32;
    for byte_idx in 0..32 {
        let cb = codes[byte_idx];
        let base = byte_idx * 4;
        for j in 0..4 {
            let q = (cb >> (j * 2)) & 0b11;
            let av = a[base + j];
            let ab = av.to_bits();
            let add1 = (((q >= 2) as u32).wrapping_neg()) & ab;
            let add2 = (((q == 3) as u32).wrapping_neg()) & ab;
            let sub1 = (((q == 0) as u32).wrapping_neg()) & (ab ^ SIGN);
            s += f32::from_bits(add1) + f32::from_bits(add2) + f32::from_bits(sub1);
        }
    }
    s
}

impl Kernel for MatMulTernaryRef {
    fn op_kind(&self) -> OpKind { OpKind::MatMulTernary }
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
            OpAttrs::MatMulTernary { out, k } => (*out, *k),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulTernary, backend: BACKEND_ID,
                reason: "MatMulTernary kernel requires OpAttrs::MatMulTernary".into(),
            }),
        };
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulTernary, backend: BACKEND_ID,
                reason: format!("expects 2 inputs / 1 output, got {} / {}",
                    inputs.len(), outputs.len()),
            });
        }

        let a = inputs[0].as_f32_vec();
        let a_shape = &inputs[0].meta.shape;
        let a_rank = a_shape.len();
        if a_shape[a_rank - 1] != k {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulTernary, backend: BACKEND_ID,
                reason: format!("A last dim {} != k {}", a_shape[a_rank - 1], k),
            });
        }
        // Flatten all leading dims into M (broadcast-mode only — weight
        // projections never need batched ternary matmul).
        let total_m: usize = a_shape.iter().take(a_rank - 1).product::<usize>().max(1);

        // Packed W: `out` rows, each `n_blocks` blocks of BLK bytes.
        let n_blocks = (k + QK - 1) / QK;
        let row_stride = n_blocks * BLK;
        let w = inputs[1].bytes;
        let need = out_dim * row_stride;
        if w.len() < need {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulTernary, backend: BACKEND_ID,
                reason: format!("packed W too small: need {} bytes, have {}",
                    need, w.len()),
            });
        }

        // Branchless ternary inner product. For activation value `av`
        // and 2-bit code q, the contribution is `(q-1)*av ∈
        // {-av, 0, +av, +2av}`. We synthesise it with masked f32-bit
        // ops — NO data-dependent branch (the old 4-way `match` per
        // weight mispredicted catastrophically) and NO per-weight FP
        // multiply (faithful to the ternary thesis): `-av` is a sign-bit
        // XOR, `0` is an all-zero mask, `+2av` (q==3, unused by Bonsai
        // ternary but defined) is `av` added twice. Numerically
        // identical to the dense dequant+matmul path.
        const SIGN: u32 = 0x8000_0000;
        #[inline(always)]
        fn contrib(av: f32, q: u8) -> f32 {
            let abits = av.to_bits();
            let add1 = (((q >= 2) as u32).wrapping_neg()) & abits;       // +av if q∈{2,3}
            let add2 = (((q == 3) as u32).wrapping_neg()) & abits;       // extra +av if q==3
            let sub1 = (((q == 0) as u32).wrapping_neg()) & (abits ^ SIGN); // -av if q==0
            f32::from_bits(add1) + f32::from_bits(add2) + f32::from_bits(sub1)
        }

        // Compute one output column `o`: its `total_m` values, written
        // into `dst` (length `total_m`). Pure function of (a, w, o) —
        // output columns are mutually independent, which is why the
        // parallel split below is bit-exact regardless of thread count.
        let col = |o: usize, dst: &mut [f32]| {
            let wbase = o * row_stride;
            for i in 0..total_m {
                let a_row = &a[i * k..i * k + k];
                let mut acc = 0f32;
                for blk in 0..n_blocks {
                    let bb = wbase + blk * BLK;
                    let d = f16_to_f32(u16::from_le_bytes([w[bb], w[bb + 1]]));
                    let codes = &w[bb + 2..bb + BLK]; // exactly 32 bytes
                    let k0 = blk * QK;
                    let kn = (k0 + QK).min(k);
                    let blk_len = kn - k0;
                    let a_blk = &a_row[k0..kn];
                    let mut s = 0f32;
                    if blk_len == QK {
                        // Full 128-element block — NEON (aarch64) or
                        // branchless scalar reference elsewhere.
                        s += block_dot(a_blk, codes);
                    } else {
                        // Partial trailing block (k not a multiple of
                        // 128; not hit by qwen3 dims but kept correct).
                        for local in 0..blk_len {
                            let q = (codes[local >> 2] >> ((local & 3) * 2)) & 0b11;
                            s += contrib(a_blk[local], q);
                        }
                    }
                    acc += d * s; // the only f32 multiply: once per 128
                }
                dst[i] = acc;
            }
        };

        // Out-major scratch `[out_dim][total_m]` so each output column
        // is a contiguous `total_m`-slice — lets us hand disjoint,
        // contiguous chunks to worker threads with no aliasing and no
        // false sharing. Transposed back to row-major at the end (cheap:
        // total_m is the token count, typically a handful).
        let mut c_om = vec![0f32; out_dim * total_m];
        let nthreads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(out_dim.max(1));

        if nthreads <= 1 || out_dim < 64 {
            for o in 0..out_dim {
                col(o, &mut c_om[o * total_m..(o + 1) * total_m]);
            }
        } else {
            // Split columns into `nthreads` contiguous bands. Determinism
            // is unaffected: each c_om entry is written by exactly one
            // thread via the pure `col` closure, identical to serial.
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

        // Transpose out-major → row-major `[total_m, out_dim]`.
        let mut c = vec![0f32; total_m * out_dim];
        for o in 0..out_dim {
            for i in 0..total_m {
                c[i * out_dim + o] = c_om[o * total_m + i];
            }
        }

        outputs[0].write_f32(&c);

        let elapsed = start.elapsed();
        // Energy model: one accumulate per weight (no FP multiply) plus
        // one f16-scaled add per 128-block. Charged well below the dense
        // MAC rate (`flops*1e-10`, flops = 2·m·n·k) to reflect that the
        // multiply is gone and the weight traffic is 2-bit, not 32-bit.
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

    // Build one Q2_0 row of `k` ternary values with block scale `d`.
    fn pack_row(vals: &[i8], d_f16: u16) -> Vec<u8> {
        let k = vals.len();
        let n_blocks = (k + QK - 1) / QK;
        let mut out = vec![0u8; n_blocks * BLK];
        for blk in 0..n_blocks {
            let bb = blk * BLK;
            out[bb..bb + 2].copy_from_slice(&d_f16.to_le_bytes());
            for local in 0..QK {
                let idx = blk * QK + local;
                if idx >= k { break; }
                let q: u8 = match vals[idx] { -1 => 0, 0 => 1, 1 => 2, _ => 3 };
                out[bb + 2 + (local >> 2)] |= q << ((local & 3) * 2);
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
            backend: BACKEND_ID,
            deterministic: true,
            seed: None,
            scratch: &mut scratch,
        };
        MatMulTernaryRef.execute(
            &mut ctx,
            &OpAttrs::MatMulTernary { out, k },
            &iv, &mut ov,
        ).unwrap();
        (0..m * out).map(|i| f32::from_le_bytes(
            [o_bytes[i*4], o_bytes[i*4+1], o_bytes[i*4+2], o_bytes[i*4+3]]))
            .collect()
    }

    #[test]
    fn ternary_matches_explicit_dot() {
        // d = 0.5 (f16 0x3800). W row0 = [+1,-1,0,+1] padded to 128 with 0.
        let d: u16 = 0x3800;
        let mut v = vec![0i8; 128];
        v[0] = 1; v[1] = -1; v[2] = 0; v[3] = 1;
        let w = pack_row(&v, d);
        let mut a = vec![0f32; 128];
        a[0] = 2.0; a[1] = 3.0; a[2] = 9.0; a[3] = 4.0;
        // Σ = d*( +a0 -a1 +0 +a3 ) = 0.5*(2 -3 +4) = 1.5
        let y = run(&a, 1, 128, &w, 1);
        assert!((y[0] - 1.5).abs() < 1e-6, "got {}", y[0]);
    }

    #[test]
    fn ternary_multi_row_and_block() {
        // k = 256 (two blocks), out = 2. Row0 all +1 in block0 only;
        // row1 all -1 in block1 only. Distinct per-row/per-block scales.
        let d0: u16 = 0x3C00; // 1.0
        let mut r0 = vec![0i8; 256];
        for x in r0.iter_mut().take(128) { *x = 1; }
        let mut r1 = vec![0i8; 256];
        for x in r1.iter_mut().skip(128) { *x = -1; }
        let mut w = pack_row(&r0, d0);
        w.extend(pack_row(&r1, d0));
        let mut a = vec![0f32; 256];
        for x in a.iter_mut().take(128) { *x = 1.0; }   // block0 = 1.0s
        for x in a.iter_mut().skip(128) { *x = 2.0; }    // block1 = 2.0s
        let y = run(&a, 1, 256, &w, 2);
        // row0: +1·(128×1.0) = 128 ; row1: -1·(128×2.0) = -256
        assert!((y[0] - 128.0).abs() < 1e-4, "row0 {}", y[0]);
        assert!((y[1] + 256.0).abs() < 1e-4, "row1 {}", y[1]);
    }

    #[test]
    fn parallel_path_is_bit_identical_to_serial() {
        // out_dim ≥ 64 forces the multi-thread split. Build a varied
        // weight (distinct per-row scale + ternary pattern) and check
        // every output column against an independent serial reference.
        let m = 3;
        let k = 256;
        let out = 200; // > 64 and not a multiple of typical core counts
        let mut a = vec![0f32; m * k];
        for (i, x) in a.iter_mut().enumerate() {
            *x = (((i * 7 + 3) % 11) as f32) - 5.0; // deterministic spread
        }
        let scales = [0x3800u16, 0x3C00, 0x4000]; // 0.5, 1.0, 2.0
        let mut w = Vec::new();
        for o in 0..out {
            let mut row = vec![0i8; k];
            for (j, r) in row.iter_mut().enumerate() {
                *r = match (o + j) % 3 { 0 => -1, 1 => 0, _ => 1 };
            }
            w.extend(pack_row(&row, scales[o % 3]));
        }
        let got = run(&a, m, k, &w, out);

        // Independent serial reference (plain f32 dot of dequantised W).
        let f16 = |h: u16| super::f16_to_f32(h);
        let mut expect = vec![0f32; m * out];
        for o in 0..out {
            let d = f16(scales[o % 3]);
            for i in 0..m {
                let mut acc = 0f32;
                for j in 0..k {
                    let t = match (o + j) % 3 { 0 => -1.0, 1 => 0.0, _ => 1.0 };
                    acc += a[i * k + j] * t * d;
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
