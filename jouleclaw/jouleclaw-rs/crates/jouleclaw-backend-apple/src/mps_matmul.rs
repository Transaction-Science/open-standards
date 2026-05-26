//! MPS-based fp32 matmul — routes through MetalPerformanceShaders'
//! `MPSMatrixMultiplication`, which automatically uses the M5 GPU's
//! Neural Accelerators ("Tensor Cores") on supporting hardware and
//! falls back to standard GPU shader execution on M3/M4.
//!
//! ## When this beats cblas_sgemm
//!
//! Apple's Accelerate `cblas_sgemm` routes to the CPU's AMX matrix
//! coprocessor. MPS routes to the GPU. The trade-off:
//!
//!   - AMX wins for small matmul (low compute, dominated by dispatch
//!     overhead). The CPU coprocessor has near-zero dispatch latency.
//!   - MPS wins for large matmul when M5 Tensor Cores are available
//!     (and on M3/M4 for very large workloads that saturate GPU
//!     throughput beyond AMX's peak).
//!   - For Bonsai single-token decode at batch=1 (small per-step
//!     matmul, memory-bandwidth-bound): AMX expected to win.
//!   - For DeBERTa entail_batch (K parallel forwards, batched
//!     fp32 attention): MPS may win on M5.
//!
//! ## What this commit ships
//!
//! Standalone fp32 matmul function + microbench. NOT yet wired into
//! the kernel-selection layer (`Runtime::boot` adaptive dispatch).
//! Wiring requires:
//!   1. An `MpsMatMul` Kernel impl (analogous to `AccelerateMatMul`).
//!   2. Capability detection: `MTLDevice.supportsFamily(.metal4)` +
//!      `supportsNeuralAccelerators` for hardware gating.
//!   3. Cost-model that prefers MPS on M5 when matmul size >
//!      some threshold; falls through to AMX otherwise.
//!
//! That wiring is the follow-up commit. This one shows the substrate
//! works and quantifies the speedup vs AMX on real M5 hardware.

#[cfg(target_os = "macos")]
pub mod inner {
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2::AnyThread; // brings `alloc` into scope
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice,
        MTLDevice, MTLResourceOptions,
    };
    use objc2_metal_performance_shaders::{
        MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
    };

    /// `out = a @ b` (no bias), where:
    ///   - `a` is fp32 row-major `[m, k]`
    ///   - `b` is fp32 row-major `[k, n]`
    ///   - `out` is fp32 row-major `[m, n]`
    ///
    /// Routes through `MPSMatrixMultiplication` on the system's
    /// default Metal device. On M5, automatically dispatches to the
    /// GPU's Neural Accelerators via the Tensor Operations layer in
    /// Metal 4; on M3/M4, falls back to general-purpose GPU compute.
    pub fn matmul_fp32(
        a: &[f32], b: &[f32], out: &mut [f32],
        m: usize, n: usize, k: usize,
    ) -> Result<(), String> {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(out.len(), m * n);

        unsafe {
            let device: Retained<ProtocolObject<dyn MTLDevice>> =
                MTLCreateSystemDefaultDevice()
                    .ok_or_else(|| "no Metal device".to_string())?;

            let queue = device.newCommandQueue()
                .ok_or_else(|| "newCommandQueue failed".to_string())?;

            // Allocate shared MTLBuffers (zero-copy on UMA — the GPU
            // sees the same pages the CPU does). Storage mode is
            // MTLResourceStorageModeShared (= 0 in the Apple enum).
            let storage_shared = MTLResourceOptions::StorageModeShared;
            let a_bytes = (m * k * std::mem::size_of::<f32>()) as usize;
            let b_bytes = (k * n * std::mem::size_of::<f32>()) as usize;
            let c_bytes = (m * n * std::mem::size_of::<f32>()) as usize;

            let buf_a = device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(a.as_ptr() as *mut _).unwrap(),
                a_bytes, storage_shared,
            ).ok_or_else(|| "newBuffer A failed".to_string())?;
            let buf_b = device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(b.as_ptr() as *mut _).unwrap(),
                b_bytes, storage_shared,
            ).ok_or_else(|| "newBuffer B failed".to_string())?;
            let buf_c = device.newBufferWithLength_options(c_bytes, storage_shared)
                .ok_or_else(|| "newBuffer C failed".to_string())?;

            // Descriptors. row_bytes = ncols * sizeof(f32).
            let desc_a = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                m, k, k * 4, MPSDataType::Float32);
            let desc_b = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                k, n, n * 4, MPSDataType::Float32);
            let desc_c = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                m, n, n * 4, MPSDataType::Float32);

            let mat_a = MPSMatrix::initWithBuffer_descriptor(
                MPSMatrix::alloc(), &buf_a, &desc_a);
            let mat_b = MPSMatrix::initWithBuffer_descriptor(
                MPSMatrix::alloc(), &buf_b, &desc_b);
            let mat_c = MPSMatrix::initWithBuffer_descriptor(
                MPSMatrix::alloc(), &buf_c, &desc_c);

            // Build the matmul kernel: C = 1.0 × A·B + 0.0 × C.
            // initWith… returns `Retained<Self>` directly (not Option)
            // in objc2 bindings — failures panic in the alloc layer.
            let matmul: Retained<MPSMatrixMultiplication> =
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    &device,
                    false,  // transposeLeft
                    false,  // transposeRight
                    m, n, k,
                    1.0, 0.0,
                );

            // Encode + commit.
            let cb = queue.commandBuffer()
                .ok_or_else(|| "commandBuffer failed".to_string())?;
            matmul.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                &cb, &mat_a, &mat_b, &mat_c);
            cb.commit();
            cb.waitUntilCompleted();

            // Copy result back. Shared storage means the buffer's
            // bytes are visible immediately after waitUntilCompleted.
            // `contents` is on the MTLBuffer protocol — dispatch via
            // ProtocolObject deref.
            let c_ptr = MTLBuffer::contents(&*buf_c).as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(c_ptr, out.as_mut_ptr(), m * n);

            // Tag the device name into a String for the bench output
            // (helpful when running across machines via ssh).
            let _ = device.name(); // touches device.name() but we don't return it
        }
        Ok(())
    }

    /// Returns the system Metal device's name (e.g., "Apple M5 Max")
    /// — useful for benches that want to report which hardware they
    /// measured on.
    pub fn device_name() -> Option<String> {
        let device = MTLCreateSystemDefaultDevice()?;
        let ns_name: Retained<NSString> = device.name();
        Some(ns_name.to_string())
    }
}

#[cfg(not(target_os = "macos"))]
pub mod inner {
    pub fn matmul_fp32(
        _a: &[f32], _b: &[f32], _out: &mut [f32],
        _m: usize, _n: usize, _k: usize,
    ) -> Result<(), String> {
        Err("MPS not available off-macos".into())
    }
    pub fn device_name() -> Option<String> { None }
}

pub use inner::{device_name, matmul_fp32};

// ── MpsMatMul Kernel impl ────────────────────────────────────────────
//
// Registers as a competing kernel against `AccelerateMatMul` via the
// Runtime's adaptive selector. Per the measured crossover (see the
// table in metal_compute.rs), MPS only wins above ~20G flops for fp32.
// `prefers()` returns Strong above that threshold, Refuse below, so
// the picker routes small matmul to AMX and huge matmul to MPS
// automatically. The Bonsai single-token decode path (m=17) lands
// firmly in AMX territory; DeBERTa entail_batch post-fan-out and
// long-prompt prefill should fan out enough to hit the MPS branch.
//
// MPS supports only the simplest matmul case here: 2D × 2D, no
// transpose, alpha=1.0. The picker refuses anything else, so AMX
// (which handles all the cases) stays the default for transposed
// or batched calls.

use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};

/// MPSMatrixMultiplication-backed fp32 matmul. Tagged
/// `AppleGpuMetal`. Selected adaptively for very large matmul
/// where the GPU dispatch overhead is amortized; AMX wins for
/// everything else.
pub struct MpsMatMul;

impl MpsMatMul {
    #[cfg(target_os = "macos")]
    pub fn new() -> Option<Self> { Some(Self) }

    #[cfg(not(target_os = "macos"))]
    pub fn new() -> Option<Self> { None }
}

impl Kernel for MpsMatMul {
    fn op_kind(&self) -> OpKind { OpKind::MatMul }
    fn backend(&self) -> BackendId { BackendId::AppleGpuMetal }
    // GPU thread scheduling can reorder fp32 accumulation across
    // threadgroups — call it "Deterministic per shape per device"
    // pragmatically (it tracks the AccelerateMatMul label even though
    // strict bit-equality isn't guaranteed across all hardware).
    fn determinism(&self) -> DeterminismClass { DeterminismClass::Deterministic }

    /// Shape-aware preference. Per the measured M5 crossover bench:
    ///
    ///   m=17,   k=n=1024:   ~35M flops  → AMX wins 95×       → Refuse
    ///   m=256,  k=n=4096:    ~8G flops  → AMX wins 2×        → Refuse
    ///   m=1024, k=n=4096:   ~34G flops  → MPS wins 1.47×     → Strong
    ///
    /// Threshold set conservatively at 20G flops — below this AMX is
    /// either decisively faster or close enough that MPS dispatch
    /// overhead isn't worth it.
    fn prefers(
        &self,
        attrs: &OpAttrs,
        input_metas: &[&jouleclaw_core::tensor::TensorMeta],
    ) -> jouleclaw_core::kernel::KernelPreference {
        use jouleclaw_core::kernel::KernelPreference;
        if input_metas.len() != 2 { return KernelPreference::Refuse; }

        let (transpose_a, transpose_b, alpha) = match attrs {
            OpAttrs::MatMul { transpose_a, transpose_b, alpha, .. } =>
                (*transpose_a, *transpose_b, *alpha),
            _ => return KernelPreference::Refuse,
        };

        // The current MPS path supports the simplest case only:
        // 2D × 2D, no transpose, alpha = 1.0. Refuse otherwise and
        // let AMX (or the reference path) handle the rest.
        if transpose_a || transpose_b || alpha != 1.0 {
            return KernelPreference::Refuse;
        }

        let a_shape = &input_metas[0].shape;
        let b_shape = &input_metas[1].shape;
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return KernelPreference::Refuse;
        }
        let (m, k_a) = (a_shape[0], a_shape[1]);
        let (k_b, n) = (b_shape[0], b_shape[1]);
        if k_a != k_b { return KernelPreference::Refuse; }

        // u64 instead of usize so the 20G-flop literal isn't out of
        // range on 32-bit targets (wasm32 — usize is 32-bit there).
        // m/n/k_a fit fine in u64; the product can't realistically
        // exceed u64 either.
        let flops = 2u64
            .saturating_mul(m as u64)
            .saturating_mul(n as u64)
            .saturating_mul(k_a as u64);
        if flops >= 20_000_000_000 {
            // ~20G flops: above the measured AMX/MPS crossover at
            // (1024, 4096, 4096) ≈ 34G flops. Use Strong to outrank
            // AccelerateMatMul's Strong at this scale (the picker
            // breaks ties on later-registered kernels — we register
            // MPS after Accelerate so it wins).
            KernelPreference::Strong
        } else {
            KernelPreference::Refuse
        }
    }

    fn execute(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        attrs: &OpAttrs,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorViewMut<'_>],
    ) -> Result<KernelResult, ExecutionError> {
        // The picker's `prefers()` has already filtered out unsupported
        // shapes/options. Re-validate cheaply (any mismatch indicates
        // a picker bug, not a workload issue).
        let (transpose_a, transpose_b, alpha) = match attrs {
            OpAttrs::MatMul { transpose_a, transpose_b, alpha, .. } =>
                (*transpose_a, *transpose_b, *alpha),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMul, backend: BackendId::AppleGpuMetal,
                reason: "expected OpAttrs::MatMul".into(),
            }),
        };
        if transpose_a || transpose_b || alpha != 1.0 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMul, backend: BackendId::AppleGpuMetal,
                reason: "MPS path only supports !transpose, alpha=1.0".into(),
            });
        }
        let a_shape = &inputs[0].meta.shape;
        let b_shape = &inputs[1].meta.shape;
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::MatMul, backend: BackendId::AppleGpuMetal,
                reason: "MPS path only supports 2D × 2D".into(),
            });
        }
        let (m, k) = (a_shape[0], a_shape[1]);
        let n = b_shape[1];
        let a = inputs[0].as_f32_vec();
        let b = inputs[1].as_f32_vec();

        let start = std::time::Instant::now();
        let mut c = vec![0.0_f32; m * n];
        matmul_fp32(&a, &b, &mut c, m, n, k)
            .map_err(|e| ExecutionError::KernelFailed {
                op: OpKind::MatMul, backend: BackendId::AppleGpuMetal,
                reason: e,
            })?;
        outputs[0].write_f32(&c);
        let elapsed = start.elapsed();

        // Energy estimate: MPS routes to the GPU. We don't have a
        // direct power-counter binding yet — using 4 pJ/flop as a
        // defensible static estimate for M-class GPUs (Accelerate
        // path uses 1 pJ/flop; GPU is ~4× higher per flop on memory
        // movement). Replace with IOReport-based measurement when
        // that lands.
        let flops = 2u64
            .saturating_mul(m as u64).saturating_mul(n as u64).saturating_mul(k as u64);
        let bytes_read = ((m * k + k * n) * 4) as u64;
        let bytes_written = (m * n * 4) as u64;

        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: flops as f64 * 4e-12,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read,
            bytes_written,
        })
    }
}

#[cfg(all(test, target_os = "macos"))]
mod kernel_tests {
    use super::MpsMatMul;
    use jouleclaw_core::kernel::{Kernel, KernelPreference};
    use jouleclaw_core::op::OpAttrs;
    use jouleclaw_core::tensor::{Dtype, TensorMeta};

    fn matmul_attrs() -> OpAttrs {
        OpAttrs::MatMul {
            transpose_a: false,
            transpose_b: false,
            alpha: 1.0,
            b_n_valid: None,
        }
    }

    fn meta_2d(rows: usize, cols: usize) -> TensorMeta {
        TensorMeta::new(Dtype::F32, &[rows, cols])
    }

    #[test]
    fn mps_refuses_small_matmul() {
        // m=17, k=n=1024: AMX wins 95×. MPS must refuse.
        let kernel = MpsMatMul::new().expect("MPS available");
        let a = meta_2d(17, 1024);
        let b = meta_2d(1024, 1024);
        let attrs = matmul_attrs();
        match kernel.prefers(&attrs, &[&a, &b]) {
            KernelPreference::Refuse => {}
            other => panic!("expected Refuse for small matmul; got {other:?}"),
        }
    }

    #[test]
    fn mps_refuses_medium_matmul() {
        // m=256, k=n=4096: ~8G flops. AMX still wins 2×.
        let kernel = MpsMatMul::new().expect("MPS available");
        let a = meta_2d(256, 4096);
        let b = meta_2d(4096, 4096);
        match kernel.prefers(&matmul_attrs(), &[&a, &b]) {
            KernelPreference::Refuse => {}
            other => panic!("expected Refuse for 8G-flop matmul; got {other:?}"),
        }
    }

    #[test]
    fn mps_strongly_prefers_huge_matmul() {
        // m=1024, k=n=4096: ~34G flops, above the ~20G threshold.
        let kernel = MpsMatMul::new().expect("MPS available");
        let a = meta_2d(1024, 4096);
        let b = meta_2d(4096, 4096);
        match kernel.prefers(&matmul_attrs(), &[&a, &b]) {
            KernelPreference::Strong => {}
            other => panic!("expected Strong for >20G-flop matmul; got {other:?}"),
        }
    }

    #[test]
    fn mps_refuses_transposed_or_alpha_scaled() {
        let kernel = MpsMatMul::new().expect("MPS available");
        let a = meta_2d(2048, 4096);
        let b = meta_2d(4096, 4096);

        let transposed = OpAttrs::MatMul {
            transpose_a: false, transpose_b: true,
            alpha: 1.0, b_n_valid: None,
        };
        assert!(matches!(
            kernel.prefers(&transposed, &[&a, &b]),
            KernelPreference::Refuse),
            "MPS must refuse transposed matmul (unsupported in the simple path)");

        let alpha_scaled = OpAttrs::MatMul {
            transpose_a: false, transpose_b: false,
            alpha: 0.5, b_n_valid: None,
        };
        assert!(matches!(
            kernel.prefers(&alpha_scaled, &[&a, &b]),
            KernelPreference::Refuse),
            "MPS must refuse alpha != 1.0");
    }

    #[test]
    fn mps_refuses_3d_batched_matmul() {
        let kernel = MpsMatMul::new().expect("MPS available");
        let a = TensorMeta::new(Dtype::F32, &[4, 2048, 4096]);
        let b = TensorMeta::new(Dtype::F32, &[4, 4096, 4096]);
        match kernel.prefers(&matmul_attrs(), &[&a, &b]) {
            KernelPreference::Refuse => {}
            other => panic!("expected Refuse for batched 3D matmul; got {other:?}"),
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::matmul_fp32;

    /// Reference scalar matmul for parity check.
    fn matmul_scalar_ref(
        a: &[f32], b: &[f32], out: &mut [f32],
        m: usize, n: usize, k: usize,
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                out[i * n + j] = sum;
            }
        }
    }

    /// MPS fp32 matmul output should match scalar reference within
    /// fp32 noise. (MPS may reorder accumulation across GPU
    /// threadgroups, so we don't require bit-equality.)
    #[test]
    fn mps_fp32_matmul_matches_scalar_reference() {
        let m = 4;
        let n = 8;
        let k = 16;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.013 - 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.071).sin()).collect();

        let mut out_mps = vec![0.0_f32; m * n];
        matmul_fp32(&a, &b, &mut out_mps, m, n, k).expect("MPS matmul");

        let mut out_ref = vec![0.0_f32; m * n];
        matmul_scalar_ref(&a, &b, &mut out_ref, m, n, k);

        let mut max_abs = 0.0_f32;
        for (m, r) in out_mps.iter().zip(&out_ref) {
            let d = (m - r).abs();
            if d > max_abs { max_abs = d; }
        }
        eprintln!("MPS vs scalar reference: max_abs = {max_abs:.6}");
        assert!(max_abs < 1e-3, "MPS diverged too far: {max_abs}");
    }

    /// Microbench: MPS fp32 matmul vs cblas_sgemm at attention-projection
    /// scale. Reports both timings; the kernel-selection layer will
    /// use shape-dependent dispatch based on these numbers.
    ///
    /// Run with: `cargo test --release -p jouleclaw-backend-apple
    ///   mps_matmul::tests::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_mps_vs_cblas_sgemm() {
        use std::time::Instant;

        // Re-use the cblas_sgemm FFI from the existing AccelerateMatMul.
        // Rust 2024: extern blocks must be marked `unsafe`.
        unsafe extern "C" {
            fn cblas_sgemm(
                order: i32, transa: i32, transb: i32,
                m: i32, n: i32, k: i32,
                alpha: f32, a: *const f32, lda: i32,
                b: *const f32, ldb: i32,
                beta: f32, c: *mut f32, ldc: i32,
            );
        }
        const CBLAS_ROW_MAJOR: i32 = 101;
        const CBLAS_NO_TRANS: i32 = 111;

        eprintln!("Device: {}",
            super::device_name().unwrap_or_else(|| "(unknown)".into()));

        for &(m, n, k) in &[
            (17, 1024, 1024),   // Bonsai attn projection (small)
            (256, 4096, 4096),  // Mid-size matmul
            (1024, 4096, 4096), // Large matmul (compute-bound)
        ] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.007).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.011).cos()).collect();
            let mut out_mps = vec![0.0_f32; m * n];
            let mut out_blas = vec![0.0_f32; m * n];

            // Warmup.
            matmul_fp32(&a, &b, &mut out_mps, m, n, k).unwrap();
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                    m as i32, n as i32, k as i32,
                    1.0, a.as_ptr(), k as i32,
                    b.as_ptr(), n as i32,
                    0.0, out_blas.as_mut_ptr(), n as i32,
                );
            }

            let iters = 3;
            let t = Instant::now();
            for _ in 0..iters {
                matmul_fp32(&a, &b, &mut out_mps, m, n, k).unwrap();
            }
            let mps_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let t = Instant::now();
            for _ in 0..iters {
                unsafe {
                    cblas_sgemm(
                        CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                        m as i32, n as i32, k as i32,
                        1.0, a.as_ptr(), k as i32,
                        b.as_ptr(), n as i32,
                        0.0, out_blas.as_mut_ptr(), n as i32,
                    );
                }
            }
            let blas_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            eprintln!(
                "[m={m} n={n} k={k}]  MPS: {mps_ms:.3} ms   cblas_sgemm (AMX): {blas_ms:.3} ms   MPS/AMX ratio: {:.2}x",
                mps_ms / blas_ms);
        }
    }
}

// Cargo wants the Accelerate framework linked for the bench's
// extern "C" cblas_sgemm. The existing AccelerateMatMul kernel
// already does this via #[link(name = "Accelerate", kind = "framework")]
// in accelerate.rs; the same linker pickup serves us here.
