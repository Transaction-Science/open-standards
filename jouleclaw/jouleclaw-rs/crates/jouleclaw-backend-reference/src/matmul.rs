//! Reference matrix multiply.
//!
//! Two modes, both with strictly fixed reduction order for determinism:
//!
//! 1. **Broadcast** — A: `[..., m, k]`, B: `[k, n]`. B is broadcast across
//!    A's batch dimensions (which are flattened into M).
//! 2. **Batched** — A: `[batch, m, k]`, B: `[batch, k, n]`. Per-batch
//!    independent matmul; output is `[batch, m, n]`. Required for multi-head
//!    attention's QK^T and attn*V products.
//!
//! Transpose flags apply to the inner `(m, k)` and `(k, n)` dims, not to
//! batch.

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

pub struct MatMulRef;

impl Kernel for MatMulRef {
    fn op_kind(&self) -> OpKind { OpKind::MatMul }
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
        let (transpose_a, transpose_b, alpha, b_n_valid) = match attrs {
            OpAttrs::MatMul { transpose_a, transpose_b, alpha, b_n_valid } =>
                (*transpose_a, *transpose_b, *alpha, *b_n_valid),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMul, backend: BACKEND_ID,
                reason: "MatMul kernel requires OpAttrs::MatMul".into(),
            }),
        };
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMul, backend: BACKEND_ID,
                reason: format!("MatMul expects 2 inputs and 1 output, got {} and {}",
                    inputs.len(), outputs.len()),
            });
        }
        let a = inputs[0].as_f32_vec();
        let b = inputs[1].as_f32_vec();
        let a_shape = &inputs[0].meta.shape;
        let b_shape = &inputs[1].meta.shape;
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        // Resolve inner dims.
        let (m, k) = if transpose_a {
            (a_shape[a_rank - 1], a_shape[a_rank - 2])
        } else {
            (a_shape[a_rank - 2], a_shape[a_rank - 1])
        };
        let (kb, n_full) = if transpose_b {
            (b_shape[b_rank - 1], b_shape[b_rank - 2])
        } else {
            (b_shape[b_rank - 2], b_shape[b_rank - 1])
        };
        if k != kb {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMul, backend: BACKEND_ID,
                reason: format!("inner dim mismatch: A k={}, B k={}", k, kb),
            });
        }
        // Logical N — if b_n_valid is set, we iterate only that many output
        // columns; otherwise the full B's N.
        let n = match b_n_valid {
            Some(nv) => {
                if nv > n_full {
                    return Err(ExecutionError::KernelFailed {
                        op: OpKind::MatMul, backend: BACKEND_ID,
                        reason: format!(
                            "b_n_valid {} exceeds B's N axis size {}", nv, n_full),
                    });
                }
                nv
            }
            None => n_full,
        };

        // Decide mode.
        let mode = if b_rank == 2 {
            MatMulMode::Broadcast
        } else if b_rank == a_rank && a_rank >= 3 {
            for d in 0..a_rank - 2 {
                if a_shape[d] != b_shape[d] {
                    return Err(ExecutionError::KernelFailed {
                        op: OpKind::MatMul, backend: BACKEND_ID,
                        reason: format!(
                            "batched matmul requires matching leading dims; A={:?}, B={:?}",
                            a_shape, b_shape),
                    });
                }
            }
            MatMulMode::Batched
        } else {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMul, backend: BACKEND_ID,
                reason: format!(
                    "unsupported rank combination: A.rank={}, B.rank={}",
                    a_rank, b_rank),
            });
        };

        let mn = m * n;
        // For B's storage stride we must always use the FULL N (`n_full`),
        // even when iterating only the first `n` columns. The reason: B is
        // laid out in memory with its full N axis; truncating only changes
        // how many positions we *use*, not how the bytes are addressed.
        let n_stride_b = n_full;
        let (c, total_m_eff) = match mode {
            MatMulMode::Broadcast => {
                let leading: usize = a_shape.iter().take(a_rank - 2).product::<usize>().max(1);
                let total_m = leading * m;

                let mut c = vec![0f32; total_m * n];
                for i in 0..total_m {
                    for j in 0..n {
                        let mut acc: f32 = 0.0;
                        for kk in 0..k {
                            let av = if transpose_a {
                                a[kk * total_m + i]
                            } else {
                                a[i * k + kk]
                            };
                            let bv = if transpose_b {
                                b[j * k + kk]
                            } else {
                                b[kk * n_stride_b + j]
                            };
                            acc += av * bv;
                        }
                        c[i * n + j] = acc * alpha;
                    }
                }
                (c, total_m)
            }
            MatMulMode::Batched => {
                let batch: usize = a_shape.iter().take(a_rank - 2).product();
                let a_per_batch = m * k;
                // B's per-batch storage uses full N stride.
                let b_per_batch = k * n_stride_b;
                let c_per_batch = mn;

                let mut c = vec![0f32; batch * c_per_batch];

                for bi in 0..batch {
                    let a_off = bi * a_per_batch;
                    let b_off = bi * b_per_batch;
                    let c_off = bi * c_per_batch;

                    let a_slice = &a[a_off..a_off + a_per_batch];
                    let b_slice = &b[b_off..b_off + b_per_batch];

                    for i in 0..m {
                        for j in 0..n {
                            let mut acc: f32 = 0.0;
                            for kk in 0..k {
                                let av = if transpose_a {
                                    a_slice[kk * m + i]
                                } else {
                                    a_slice[i * k + kk]
                                };
                                let bv = if transpose_b {
                                    b_slice[j * k + kk]
                                } else {
                                    b_slice[kk * n_stride_b + j]
                                };
                                acc += av * bv;
                            }
                            c[c_off + i * n + j] = acc * alpha;
                        }
                    }
                }
                (c, batch * m)
            }
        };

        outputs[0].write_f32(&c);

        let elapsed = start.elapsed();
        let flops = (total_m_eff * n * k * 2) as f64;
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: flops * 1e-10,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: ((a.len() + b.len()) * 4) as u64,
            bytes_written: (c.len() * 4) as u64,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum MatMulMode {
    /// B is 2D, broadcast across A's batch dims.
    Broadcast,
    /// A and B both have matching leading batch dims; per-batch matmul.
    Batched,
}
