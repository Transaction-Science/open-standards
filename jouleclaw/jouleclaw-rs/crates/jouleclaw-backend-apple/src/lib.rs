//! # jouleclaw-backend-apple
//!
//! Apple Silicon backend for the Joule runtime.
//!
//! Phase 1.2 shipped:
//! - `AccelerateMatMul` — F32 `cblas_sgemm` via the Accelerate framework
//!   (AMX-backed on M-series). Tagged `Deterministic`.
//!
//! Phase 1.3 ships:
//! - `MetalMatMul` — skeleton with MSL kernel source. Body fills in when
//!   the `metal` crate is wired up as a target-gated dependency.
//!
//! Phase 1.3 follow-up will add:
//! - Metal compute kernels for the other primitives (Softmax, Norm,
//!   Activation, Add, Mul, Lookup)
//! - ANE dispatch via CoreML for inference-only paths
//! - IOReport-based energy counter integration
//!
//! On non-Apple-Silicon platforms this crate compiles, but `all_kernels()`
//! returns empty and the kernel constructors return `None`. The runtime
//! falls back to the reference backend.

mod accelerate;
mod metal_compute;
pub mod mps_matmul;

pub use accelerate::AccelerateMatMul;
pub use metal_compute::MetalMatMul;
pub use mps_matmul::MpsMatMul;

use jouleclaw_core::kernel::Kernel;
use std::sync::Arc;

/// Return all Apple-backend kernels available on the running platform.
/// Empty on non-Apple-Silicon hosts.
pub fn all_kernels() -> Vec<Arc<dyn Kernel>> {
    let mut out: Vec<Arc<dyn Kernel>> = Vec::new();

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if let Some(k) = AccelerateMatMul::new() { out.push(Arc::new(k)); }
        if let Some(k) = MetalMatMul::new()      { out.push(Arc::new(k)); }
        // MpsMatMul registered AFTER AccelerateMatMul so ties at
        // `Strong` go to MPS — the picker tie-breaker uses
        // registration order. The shape-aware `prefers()` ensures MPS
        // only declares Strong above the measured ~20G flops
        // crossover; below that it Refuses and AMX wins by default.
        if let Some(k) = MpsMatMul::new()        { out.push(Arc::new(k)); }
    }

    out
}
