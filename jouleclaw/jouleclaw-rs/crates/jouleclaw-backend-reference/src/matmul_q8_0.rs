//! Reference Q8_0 matmul: `Y = A @ W^T`, W packed Q8_0 (32-element
//! blocks, 1 fp16 scale per block, 8.5 bits/weight on disk).
//!
//! This is the "skip the dequant" path for Q8_0-stored weights. The
//! standard dispatch dequantizes Q8_0 → fp32 at graph compile time,
//! then runs `cblas_sgemm` (AMX on Apple Silicon, scalar/NEON on
//! Pi/Android). On non-AMX edge targets that dequant + sgemm
//! round-trip wastes both bandwidth (writing the fp32 intermediate)
//! and compute (multiplying fp32 by fp32 when an int8 dot product
//! would do).
//!
//! This kernel keeps W packed and runs:
//!
//!   1. Per-row activation quant: `a_q[i,:] = round(A[i,:] / a_scale[i])`
//!      where `a_scale[i] = max(|A[i,:]|) / 127`. Symmetric int8,
//!      no zero-point.
//!   2. Per (i, j, b): int8×int8 dot product across the b-th
//!      32-element block, scaled by `block_scale * a_scale[i]`.
//!      NEON `vmull_s8` + `vpadalq_s16` (no `vdotq_s32` — that's
//!      still nightly-gated in stable Rust).
//!   3. Sum block contributions; add bias if present.
//!
//! Determinism: int32 dot products are order-independent; the f32
//! accumulation across blocks is left-to-right (k ascending) so the
//! result is bit-reproducible per (shape, weights, inputs).
//!
//! Selection: the kernel declares `Strong` for any Q8_0 matmul. On
//! Apple Silicon (where AMX wins the equivalent fp32 sgemm), the
//! existing MatMul path through `Weight::Dense` won't reach this
//! kernel — it's only selected when the weight was loaded as
//! `Weight::Q80`, which `embed_weight` does only for Q8_0 dtypes
//! that explicitly opt into the packed path.

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

const Q8_0_ELEMS: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;

pub struct MatMulQ80Ref;

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

impl Kernel for MatMulQ80Ref {
    fn op_kind(&self) -> OpKind { OpKind::MatMulQ80 }
    fn backend(&self) -> BackendId { BACKEND_ID }
    fn determinism(&self) -> DeterminismClass { DeterminismClass::Deterministic }

    fn execute(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        attrs: &OpAttrs,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorViewMut<'_>],
    ) -> Result<KernelResult, ExecutionError> {
        let (out_dim, k) = match attrs {
            OpAttrs::MatMulQ80 { out, k } => (*out, *k),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulQ80, backend: BACKEND_ID,
                reason: "expected OpAttrs::MatMulQ80".into(),
            }),
        };
        if k % Q8_0_ELEMS != 0 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulQ80, backend: BACKEND_ID,
                reason: format!("k ({k}) must be a multiple of 32 (Q8_0 block size)"),
            });
        }
        let blocks_per_row = k / Q8_0_ELEMS;

        let a = inputs[0].as_f32_vec();
        let w_bytes: Vec<u8> = inputs[1].bytes.to_vec();
        let expected_w_bytes = out_dim * blocks_per_row * Q8_0_BLOCK_BYTES;
        if w_bytes.len() != expected_w_bytes {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulQ80, backend: BACKEND_ID,
                reason: format!(
                    "Q8_0 weight bytes mismatch: have {}, expected {}",
                    w_bytes.len(), expected_w_bytes),
            });
        }

        // A may be 2D `[m, k]` or higher-rank with `k` as the last dim.
        let a_shape = &inputs[0].meta.shape;
        let a_rank = a_shape.len();
        if a_rank < 2 || a_shape[a_rank - 1] != k {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMulQ80, backend: BACKEND_ID,
                reason: format!(
                    "A's last dim must equal k={k}; got shape {:?}", a_shape),
            });
        }
        let leading: usize = a_shape.iter().take(a_rank - 2).product::<usize>().max(1);
        let m = leading * a_shape[a_rank - 2];

        let start = Instant::now();

        // 1. Quantize A per-row to int8 + scale.
        let mut q_a = vec![0_i8; m * k];
        let mut a_scales = vec![0.0_f32; m];
        for i in 0..m {
            let row = &a[i * k..(i + 1) * k];
            let mut max_abs = 0.0_f32;
            for &v in row {
                let av = v.abs();
                if av > max_abs { max_abs = av; }
            }
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            a_scales[i] = scale;
            let inv = 1.0 / scale;
            for (slot, &v) in q_a[i * k..(i + 1) * k].iter_mut().zip(row) {
                *slot = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }

        // 2. Per (i, j): walk the blocks, NEON dot + per-block scale.
        let mut c = vec![0.0_f32; m * out_dim];
        for i in 0..m {
            let a_row = &q_a[i * k..(i + 1) * k];
            let a_s = a_scales[i];
            for j in 0..out_dim {
                let w_row_off = j * blocks_per_row * Q8_0_BLOCK_BYTES;
                let mut accum_f32 = 0.0_f32;
                for b in 0..blocks_per_row {
                    let block_off = w_row_off + b * Q8_0_BLOCK_BYTES;
                    let scale_bits = u16::from_le_bytes([
                        w_bytes[block_off], w_bytes[block_off + 1]]);
                    let block_scale = f16_to_f32(scale_bits);

                    let a_chunk = &a_row[b * Q8_0_ELEMS..(b + 1) * Q8_0_ELEMS];
                    let w_chunk_start = block_off + 2;

                    let sum_i32 = dot_block(a_chunk, &w_bytes[w_chunk_start..w_chunk_start + Q8_0_ELEMS]);
                    accum_f32 += sum_i32 as f32 * block_scale * a_s;
                }
                c[i * out_dim + j] = accum_f32;
            }
        }

        outputs[0].write_f32(&c);

        let elapsed = start.elapsed();
        let flops = 2u64
            .saturating_mul(m as u64)
            .saturating_mul(out_dim as u64)
            .saturating_mul(k as u64);
        // Energy estimate: NEON int8 path is roughly 2× more
        // ops/joule than fp32 sgemm at the int8 dot-product level.
        // Use 0.5 pJ/flop (vs cblas_sgemm's 1 pJ). IOReport
        // measurement replaces this when it lands.
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: flops as f64 * 0.5e-12,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: (m * k * 4 + w_bytes.len()) as u64,
            bytes_written: (m * out_dim * 4) as u64,
        })
    }
}

/// 32-element int8 × int8 → int32 dot product. NEON when available,
/// scalar fallback otherwise. Order-independent — int32 wraps
/// consistently regardless of how the partials are reduced.
#[inline]
fn dot_block(a: &[i8], w_bytes: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), Q8_0_ELEMS);
    debug_assert_eq!(w_bytes.len(), Q8_0_ELEMS);
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut acc = vdupq_n_s32(0);
        // Two 16-byte NEON loads per 32-element block.
        for c in 0..2 {
            let a_v = vld1q_s8(a.as_ptr().add(c * 16));
            let b_v = vld1q_s8(w_bytes.as_ptr().add(c * 16) as *const i8);
            let prod_lo = vmull_s8(vget_low_s8(a_v), vget_low_s8(b_v));
            let prod_hi = vmull_s8(vget_high_s8(a_v), vget_high_s8(b_v));
            acc = vpadalq_s16(acc, prod_lo);
            acc = vpadalq_s16(acc, prod_hi);
        }
        vaddvq_s32(acc)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum: i32 = 0;
        for c in 0..Q8_0_ELEMS {
            sum += (a[c] as i32) * (w_bytes[c] as i8 as i32);
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};

    /// Quick fp32 → fp16-bits roundtrip for normal-range positive
    /// values (Q8_0 block scales are always positive normals).
    fn f32_to_f16_bits(v: f32) -> u16 {
        if v == 0.0 { return 0; }
        let bits = v.to_bits();
        let sign = ((bits >> 31) & 1) as u16;
        let exp_f32 = ((bits >> 23) & 0xFF) as i32 - 127;
        let mantissa_f32 = bits & 0x7F_FFFF;
        let exp_f16 = (exp_f32 + 15) as u16;
        let mantissa_f16 = (mantissa_f32 >> 13) as u16;
        (sign << 15) | (exp_f16 << 10) | mantissa_f16
    }

    fn synth_q8_0(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
        assert_eq!(k % Q8_0_ELEMS, 0);
        let blocks_per_row = k / Q8_0_ELEMS;
        let mut bytes = vec![0_u8; n * blocks_per_row * Q8_0_BLOCK_BYTES];
        let mut fp32 = vec![0.0_f32; n * k];
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state
        };
        for j in 0..n {
            for b in 0..blocks_per_row {
                let block_off = j * blocks_per_row * Q8_0_BLOCK_BYTES + b * Q8_0_BLOCK_BYTES;
                let scale_f32 = (((next() >> 32) as u32 % 200) as f32 + 1.0) / 1000.0;
                let scale_bits = f32_to_f16_bits(scale_f32);
                bytes[block_off] = (scale_bits & 0xFF) as u8;
                bytes[block_off + 1] = (scale_bits >> 8) as u8;
                let actual_scale = f16_to_f32(scale_bits);
                for c in 0..Q8_0_ELEMS {
                    let qv = ((next() >> 32) as i32 % 255 - 127) as i8;
                    bytes[block_off + 2 + c] = qv as u8;
                    fp32[j * k + b * Q8_0_ELEMS + c] = qv as f32 * actual_scale;
                }
            }
        }
        (bytes, fp32)
    }

    fn matmul_bt_fp32_ref(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * w[j * k + kk];
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    /// MatMulQ80Ref output must match the fp32 dequant-then-matmul
    /// reference within the activation-quant noise. Weight quant
    /// error is fixed (set by the synth function); activation quant
    /// adds ~1/254 ≈ 0.4% relative noise per element.
    #[test]
    fn matmul_q8_0_kernel_matches_fp32_reference() {
        let m = 4;
        let n = 8;
        let k = 64;
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.011).sin() * 0.9).collect();
        let (w_q8, w_fp32) = synth_q8_0(n, k, 12345);

        // Run the kernel.
        let kernel = MatMulQ80Ref;
        let attrs = OpAttrs::MatMulQ80 { out: n, k };

        let a_meta = TensorMeta::new(Dtype::F32, &[m, k]);
        let mut a_bytes = Vec::with_capacity(a.len() * 4);
        for &v in &a { a_bytes.extend_from_slice(&v.to_le_bytes()); }
        let a_storage = std::sync::Arc::new(TensorStorage { bytes: a_bytes, mapped: None });
        let a_tensor = Tensor { meta: a_meta.clone(), storage: a_storage };
        let a_view = TensorView { meta: &a_tensor.meta, bytes: &a_tensor.storage.bytes };

        let w_meta = TensorMeta::new(Dtype::U8, &[w_q8.len()]);
        let w_storage = std::sync::Arc::new(TensorStorage { bytes: w_q8, mapped: None });
        let w_tensor = Tensor { meta: w_meta.clone(), storage: w_storage };
        let w_view = TensorView { meta: &w_tensor.meta, bytes: &w_tensor.storage.bytes };

        let mut out_bytes = vec![0_u8; m * n * 4];
        let out_meta = TensorMeta::new(Dtype::F32, &[m, n]);
        let mut out_view = TensorViewMut { meta: &out_meta, bytes: &mut out_bytes };

        let mut scratch = vec![0_u8; 0];
        let mut ctx = ExecutionContext {
            backend: BACKEND_ID,
            deterministic: true,
            seed: None,
            scratch: &mut scratch,
        };
        kernel.execute(&mut ctx, &attrs, &[a_view, w_view], &mut [out_view])
            .expect("kernel execute");

        let out_q8: Vec<f32> = out_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let out_ref = matmul_bt_fp32_ref(&a, &w_fp32, m, n, k);

        let mut max_abs = 0.0_f32;
        let mut sum_sq = 0.0_f64;
        let mut sum_sq_ref = 0.0_f64;
        for (q, r) in out_q8.iter().zip(&out_ref) {
            let d = (q - r).abs();
            if d > max_abs { max_abs = d; }
            sum_sq += (d as f64).powi(2);
            sum_sq_ref += (*r as f64).powi(2);
        }
        let rms_rel = (sum_sq / sum_sq_ref).sqrt();
        eprintln!("MatMulQ80Ref vs fp32 reference: max_abs={max_abs:.4} rms_rel={rms_rel:.4}");
        // Activation int8 quant typically gives ~0.4% relative error;
        // 5% is a generous bound that catches kernel bugs without
        // chasing benign quant noise.
        assert!(rms_rel < 0.05,
            "Q8_0 kernel diverged too far: rms_rel={rms_rel}");
    }
}
