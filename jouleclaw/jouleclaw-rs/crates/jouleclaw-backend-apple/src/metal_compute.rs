//! Metal compute kernels for Apple GPU.
//!
//! Phase 1.3 ships the structural skeleton; the actual MSL kernel sources
//! and Rust-side command-buffer dispatch land when the `metal` crate is
//! added as a target-gated dependency. The skeleton is here so:
//! 1. The validation harness can already enumerate "Metal kernels exist"
//!    and produce informative errors.
//! 2. The shape of dispatch calls is fixed before adding GPU drivers,
//!    so the integration is mechanical.
//!
//! When wiring up:
//!
//! ```toml
//! [target.'cfg(target_os = "macos")'.dependencies]
//! metal = "0.29"
//! objc2 = "0.5"
//! objc2-foundation = "0.2"
//! ```
//!
//! Then implement `execute_impl` below using:
//!
//! ```ignore
//! let device = metal::Device::system_default()?;
//! let queue = device.new_command_queue();
//! let lib = device.new_library_with_source(MSL_SOURCE, &metal::CompileOptions::new())?;
//! let function = lib.get_function("matmul_f32", None)?;
//! let pipeline = device.new_compute_pipeline_state_with_function(&function)?;
//! // upload buffers, dispatch threadgroups, copy back
//! ```
//!
//! Determinism note: GPU reduction ordering is not automatically
//! deterministic. The MSL kernel must use a fixed work-distribution scheme
//! (e.g., one threadgroup per output row, in-thread reduction) to match
//! the reference backend bit-for-bit. Validation against `MatMulRef` is
//! mandatory.
//!
//! ## M5 Metal 4 Tensor APIs — the lever-3 upgrade path
//!
//! Apple shipped GPU-side tensor cores ("Neural Accelerators") in
//! every M5 GPU core in October 2025, exposed via the
//! `MTLTensor` / Tensor Operations API set in Metal 4. From the
//! edge-architecture notes survey: "1024 FLOPS per GPU core per
//! cycle for FP16 matrix operations, ~2048 OPS for INT8 matrix
//! operations, optimal tile 32×32." On M5, INT8 matmul runs at
//! literally 2× the FP16 rate via dedicated silicon — a different
//! economic regime from M3/M4 where INT8 saved only memory.
//!
//! The skeleton above (basic Metal compute via MSL kernel) is the
//! M3/M4 path. The M5 upgrade is to add a SECOND kernel that uses
//! the Tensor APIs:
//!
//! ```ignore
//! // In MSL (Metal 4+):
//! #include <metal_tensor>
//! using namespace metal;
//!
//! kernel void matmul_tensor_i8(
//!     tensor<int8_t,  dextents<int, 2>> a,
//!     tensor<int8_t,  dextents<int, 2>> b,
//!     tensor<int32_t, dextents<int, 2>> c,
//!     uint2 gid [[thread_position_in_grid]])
//! {
//!     // Apple-blessed matmul over 32×32 tiles using the
//!     // per-GPU-core Neural Accelerator. Maps to native silicon
//!     // on M5; falls back to SIMD-group on M3/M4 (and is then
//!     // slower than the regular MSL kernel — that's why this is
//!     // a separate dispatch path with a hardware-gating check).
//! }
//! ```
//!
//! Hardware gating: query `MTLDevice.supportsFamily(.metal4)` and
//! `MTLDevice.supportsNeuralAccelerators` (or the runtime-detect
//! equivalent in Metal 4) at backend init; register the Tensor
//! kernel only when both succeed. On M3/M4, fall through to the
//! AMX path via Accelerate — which is already optimal there.
//!
//! ### Measured baseline (M3 Ultra vs M5 Max)
//!
//! M5 hardware is now reachable via SSH (Tailscale 100.86.20.59).
//! The existing Bonsai-1.7B cascade test, unchanged source, run
//! on both:
//!
//!   M3 Ultra (dev):  wall ≈ 3.8-4.0 s, 1107 mJ measured, "Paris."
//!   M5 Max (ssh):    wall ≈ 2.72-2.88 s, same answer, same joule
//!                    receipt (the static estimate doesn't depend
//!                    on host).
//!
//! Free ~30-40% wall-clock from running on M5 silicon — entirely
//! via the already-shipped Accelerate (AMX) path. M5's CPU matrix
//! coprocessor is genuinely faster than M3's at fp32 cblas_sgemm.
//!
//! ### What's left to capture from M5 silicon
//!
//! The CPU-side AMX win is automatic. The GPU-side Neural Accelerators
//! require an explicit Metal/MPS dispatch path because Accelerate runs
//! on the CPU. See `mps_matmul.rs` for the substrate.
//!
//! ### Measured: MPS vs cblas_sgemm on M5 Max, fp32
//!
//! From `mps_matmul::tests::bench_mps_vs_cblas_sgemm`:
//!
//!   Shape (m,n,k)            MPS (GPU)    cblas (AMX)   Winner
//!   (17, 1024, 1024)         5.45 ms      0.058 ms      AMX  95×
//!   (256, 4096, 4096)        12.69 ms     6.33 ms       AMX  2×
//!   (1024, 4096, 4096)       14.23 ms     20.84 ms      MPS  1.47×
//!
//! Crossover at ~m=512. Below that, AMX dominates — Metal's
//! command-buffer dispatch overhead crushes any compute advantage on
//! small matmul. Above ~m=1024 at k=n=4096, MPS (and on M5 the GPU
//! Tensor Cores it dispatches to) starts winning.
//!
//! ### Where adaptive routing helps which workloads
//!
//!   - **Bonsai single-token decode** (m=17 per layer): stay on AMX.
//!     MPS is 95× slower at this scale.
//!   - **DeBERTa entail_batch** (m ≈ 17 × N pairs after batching):
//!     once N is large enough that effective m > 512, switch to MPS.
//!   - **Long-prompt prefill** (m = seq_len, often > 1024): switch
//!     to MPS for the verify-stage compute.
//!   - **Batched LLM inference** (m = batch × seq): same threshold.
//!
//! Wiring this into adaptive kernel selection (`Runtime::boot`) is
//! the production follow-up. The cost model already exists for tier
//! selection; per-kernel shape-dependent dispatch is a similar
//! decision layer.

use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};

pub struct MetalMatMul;

impl MetalMatMul {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn new() -> Option<Self> {
        // Phase 1.3: registration disabled until the kernel body is implemented.
        // Returning None means the runtime ignores it and uses the next-best
        // candidate (Accelerate). This keeps the validation harness from
        // failing on a stub.
        None
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    pub fn new() -> Option<Self> { None }
}

impl Kernel for MetalMatMul {
    fn op_kind(&self) -> OpKind { OpKind::MatMul }
    fn backend(&self) -> BackendId { BackendId::AppleGpuMetal }
    fn determinism(&self) -> DeterminismClass { DeterminismClass::Deterministic }

    fn execute(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _attrs: &OpAttrs,
        _inputs: &[TensorView<'_>],
        _outputs: &mut [TensorViewMut<'_>],
    ) -> Result<KernelResult, ExecutionError> {
        Err(ExecutionError::KernelFailed {
            op: OpKind::MatMul,
            backend: BackendId::AppleGpuMetal,
            reason: "MetalMatMul body not yet implemented (Phase 1.3 follow-up)".into(),
        })
    }
}

/// Metal Shading Language kernel source for F32 matmul.
///
/// Designed for determinism: one threadgroup per output row, in-thread
/// reduction in fixed `kk = 0..k` order, no atomic adds, no parallel sums.
/// This is the slow-but-correct reference; future revisions can add
/// SIMD-group matrix instructions once the validation oracle confirms drift
/// is bounded.
#[allow(dead_code)]
const MSL_MATMUL_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matmul_f32(
    device const float* A      [[ buffer(0) ]],
    device const float* B      [[ buffer(1) ]],
    device       float* C      [[ buffer(2) ]],
    constant     uint&  M      [[ buffer(3) ]],
    constant     uint&  N      [[ buffer(4) ]],
    constant     uint&  K      [[ buffer(5) ]],
    constant     uint&  ta     [[ buffer(6) ]],   // transpose_a flag
    constant     uint&  tb     [[ buffer(7) ]],   // transpose_b flag
    uint2 gid                  [[ thread_position_in_grid ]])
{
    if (gid.x >= N || gid.y >= M) return;
    uint row = gid.y;
    uint col = gid.x;
    float acc = 0.0;
    // Fixed reduction order kk = 0..K; deterministic.
    for (uint kk = 0; kk < K; kk++) {
        float a = ta ? A[kk * M + row] : A[row * K + kk];
        float b = tb ? B[col * K + kk] : B[kk * N + col];
        acc += a * b;
    }
    C[row * N + col] = acc;
}
"#;
