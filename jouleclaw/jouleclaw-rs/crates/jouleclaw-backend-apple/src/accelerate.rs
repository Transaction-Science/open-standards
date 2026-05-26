//! Accelerate-backed F32 MatMul.
//!
//! Calls into Apple's Accelerate framework via `cblas_sgemm`. On M-series
//! chips this routes through the AMX coprocessor automatically.
//!
//! Determinism: `cblas_sgemm` is deterministic for fixed inputs and shapes
//! on a single thread. We pin to single-threaded BLAS by setting
//! `OPENBLAS_NUM_THREADS=1` is not relevant — Apple's Accelerate uses
//! its own scheduler, which on M-series chips happens to be deterministic
//! for F32. We validate that empirically via the `jouleclaw-runtime::validate`
//! harness against the reference backend.

use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

pub struct AccelerateMatMul;

impl AccelerateMatMul {
    /// Construct on macOS aarch64; returns `None` on other platforms.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn new() -> Option<Self> { Some(Self) }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    pub fn new() -> Option<Self> { None }
}

impl Kernel for AccelerateMatMul {
    fn op_kind(&self) -> OpKind { OpKind::MatMul }
    fn backend(&self) -> BackendId { BackendId::AppleAmx }
    fn determinism(&self) -> DeterminismClass { DeterminismClass::Deterministic }

    /// Adaptive shape preference. The Accelerate/AMX path has per-call
    /// dispatch overhead that dominates the actual compute on tiny
    /// matmuls (e.g. attention's per-head `[1, d_head] @ [d_head,
    /// ≤ctx]` scores or `attn × V` at decode). Empirically, below
    /// roughly 1M flops the reference scalar loop wins; above it
    /// Accelerate wins by orders of magnitude. The picker uses this
    /// to route tiny matmuls to reference and large ones to AMX
    /// automatically — the per-tier `Runtime::reference_only` pin
    /// PrismTier used to need is now unnecessary.
    fn prefers(
        &self,
        attrs: &OpAttrs,
        input_metas: &[&jouleclaw_core::tensor::TensorMeta],
    ) -> jouleclaw_core::kernel::KernelPreference {
        use jouleclaw_core::kernel::KernelPreference;
        if input_metas.len() != 2 { return KernelPreference::Acceptable; }
        let (transpose_a, transpose_b) = match attrs {
            OpAttrs::MatMul { transpose_a, transpose_b, .. } =>
                (*transpose_a, *transpose_b),
            _ => return KernelPreference::Refuse,
        };
        let a_shape = &input_metas[0].shape;
        let b_shape = &input_metas[1].shape;
        if a_shape.len() < 2 || b_shape.len() < 2 {
            return KernelPreference::Acceptable;
        }
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();
        let (m, k) = if transpose_a {
            (a_shape[a_rank - 1], a_shape[a_rank - 2])
        } else {
            (a_shape[a_rank - 2], a_shape[a_rank - 1])
        };
        let n = if transpose_b {
            b_shape[b_rank - 2]
        } else {
            b_shape[b_rank - 1]
        };
        let leading: usize = a_shape.iter().take(a_rank - 2).product::<usize>().max(1);
        let flops = 2usize.saturating_mul(leading).saturating_mul(m).saturating_mul(n).saturating_mul(k);
        // Threshold calibrated against the prior PrismTier measurement
        // (reference_only beat boot by ~40% on tiny attention matmuls).
        // Above ~1M flops Accelerate's tiled SGEMM is a clear win.
        if flops >= 1_000_000 {
            KernelPreference::Strong
        } else {
            KernelPreference::Weak
        }
    }

    fn execute(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        attrs: &OpAttrs,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorViewMut<'_>],
    ) -> Result<KernelResult, ExecutionError> {
        execute_impl(attrs, inputs, outputs)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn execute_impl(
    attrs: &OpAttrs,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorViewMut<'_>],
) -> Result<KernelResult, ExecutionError> {
    let start = Instant::now();
    let (transpose_a, transpose_b, alpha) = match attrs {
        OpAttrs::MatMul { transpose_a, transpose_b, alpha, b_n_valid: _ } =>
            (*transpose_a, *transpose_b, *alpha),
        _ => return Err(ExecutionError::KernelFailed {
            op: OpKind::MatMul, backend: BackendId::AppleAmx,
            reason: "expected OpAttrs::MatMul".into(),
        }),
    };

    let a = inputs[0].as_f32_vec();
    let b = inputs[1].as_f32_vec();
    let a_shape = &inputs[0].meta.shape;
    let b_shape = &inputs[1].meta.shape;
    let a_rank = a_shape.len();
    let b_rank = b_shape.len();

    // Resolve inner dims. The last two axes are always (m,k) or (k,n)
    // (or their transposes); leading axes are batch.
    let (m, k) = if transpose_a {
        (a_shape[a_rank - 1], a_shape[a_rank - 2])
    } else {
        (a_shape[a_rank - 2], a_shape[a_rank - 1])
    };
    let (kb, n) = if transpose_b {
        (b_shape[b_rank - 1], b_shape[b_rank - 2])
    } else {
        (b_shape[b_rank - 2], b_shape[b_rank - 1])
    };
    if k != kb {
        return Err(ExecutionError::KernelFailed {
            op: OpKind::MatMul, backend: BackendId::AppleAmx,
            reason: format!("inner dim mismatch: A k={}, B k={}", k, kb),
        });
    }

    // Two dispatch modes, matching the reference kernel's semantics:
    //   Broadcast — B is 2D, broadcast across A's batch dims.
    //   Batched   — A and B both have matching leading batch dims; one
    //               sgemm per batch slice.
    let mode = if b_rank == 2 {
        MatMulMode::Broadcast
    } else if b_rank == a_rank && a_rank >= 3 {
        for d in 0..a_rank - 2 {
            if a_shape[d] != b_shape[d] {
                return Err(ExecutionError::KernelFailed {
                    op: OpKind::MatMul, backend: BackendId::AppleAmx,
                    reason: format!(
                        "batched matmul requires matching leading dims; A={:?}, B={:?}",
                        a_shape, b_shape),
                });
            }
        }
        MatMulMode::Batched
    } else {
        return Err(ExecutionError::KernelFailed {
            op: OpKind::MatMul, backend: BackendId::AppleAmx,
            reason: format!(
                "unsupported rank combination: A.rank={}, B.rank={}",
                a_rank, b_rank),
        });
    };

    let trans_a = if transpose_a { CblasTranspose::Trans } else { CblasTranspose::NoTrans };
    let trans_b = if transpose_b { CblasTranspose::Trans } else { CblasTranspose::NoTrans };

    let (c, flops, total_m_eff) = match mode {
        MatMulMode::Broadcast => {
            let leading: usize = a_shape.iter().take(a_rank - 2).product::<usize>().max(1);
            let total_m = leading * m;
            let mut c = vec![0f32; total_m * n];

            let lda = if transpose_a { total_m } else { k };
            let ldb = if transpose_b { k } else { n };
            let ldc = n;

            unsafe {
                cblas_sgemm(
                    CblasLayout::RowMajor as i32,
                    trans_a as i32, trans_b as i32,
                    total_m as i32, n as i32, k as i32,
                    alpha,
                    a.as_ptr(), lda as i32,
                    b.as_ptr(), ldb as i32,
                    0.0,
                    c.as_mut_ptr(), ldc as i32,
                );
            }
            (c, (total_m * n * k * 2) as f64, total_m)
        }
        MatMulMode::Batched => {
            let batch: usize = a_shape.iter().take(a_rank - 2).product();
            let a_per_batch = m * k;
            let b_per_batch = k * n;
            let c_per_batch = m * n;
            let mut c = vec![0f32; batch * c_per_batch];

            let lda = if transpose_a { m } else { k };
            let ldb = if transpose_b { k } else { n };
            let ldc = n;

            for bi in 0..batch {
                let a_off = bi * a_per_batch;
                let b_off = bi * b_per_batch;
                let c_off = bi * c_per_batch;
                unsafe {
                    cblas_sgemm(
                        CblasLayout::RowMajor as i32,
                        trans_a as i32, trans_b as i32,
                        m as i32, n as i32, k as i32,
                        alpha,
                        a.as_ptr().add(a_off), lda as i32,
                        b.as_ptr().add(b_off), ldb as i32,
                        0.0,
                        c.as_mut_ptr().add(c_off), ldc as i32,
                    );
                }
            }
            (c, (batch * m * n * k * 2) as f64, batch * m)
        }
    };
    let _ = total_m_eff;

    outputs[0].write_f32(&c);

    let elapsed = start.elapsed();
    Ok(KernelResult {
        joules: JouleMeasurement {
            // ~1 pJ/FLOP on M-series AMX is a defensible estimate; replaced
            // with measured values once the IOReport integration lands.
            joules: flops * 1e-12,
            energy_source: EnergySourceId(0),
            measurement_window: elapsed,
            attribution_confidence: 0.0,
        },
        wall_clock: elapsed,
        bytes_read: ((a.len() + b.len()) * 4) as u64,
        bytes_written: (c.len() * 4) as u64,
    })
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Clone, Copy)]
enum MatMulMode { Broadcast, Batched }

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn execute_impl(
    _attrs: &OpAttrs,
    _inputs: &[TensorView<'_>],
    _outputs: &mut [TensorViewMut<'_>],
) -> Result<KernelResult, ExecutionError> {
    Err(ExecutionError::KernelFailed {
        op: OpKind::MatMul,
        backend: BackendId::AppleAmx,
        reason: "AccelerateMatMul is only available on Apple Silicon (macos + aarch64)".into(),
    })
}

// ---- Accelerate FFI ----

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[repr(i32)]
#[allow(dead_code)]
enum CblasLayout { RowMajor = 101, ColMajor = 102 }

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[repr(i32)]
#[allow(dead_code)]
#[derive(Clone, Copy)]
enum CblasTranspose { NoTrans = 111, Trans = 112, ConjTrans = 113 }

// Rust 2024 edition requires extern blocks to be marked `unsafe`.
// See https://doc.rust-lang.org/edition-guide/rust-2024/unsafe-extern.html
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe extern "C" {
    fn cblas_sgemm(
        layout: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32, n: i32, k: i32,
        alpha: f32,
        a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32,
        c: *mut f32, ldc: i32,
    );
}
