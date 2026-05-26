//! Backend identification.
//!
//! A backend is a place where kernels can run. Examples: a CPU SIMD ISA,
//! a GPU compute API, a neural-engine accelerator, a custom silicon path.

use crate::op::OpKind;

/// Stable identifier for a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendId {
    // ---- Apple Silicon (Phase 1 targets) ----
    /// ARM64 CPU with NEON SIMD.
    AppleCpuNeon,
    /// Apple AMX matrix coprocessor (accessible via Accelerate framework).
    AppleAmx,
    /// Apple GPU via Metal compute.
    AppleGpuMetal,
    /// Apple Neural Engine via CoreML/MPSGraph.
    AppleAne,

    // ---- Other backends (Phase 2+) ----
    X86Avx512,
    X86Amx,
    NvidiaCuda,
    NvidiaTensorRt,
    AmdHip,
    IntelXmx,

    /// Reserved for custom-silicon targets.
    Custom(u16),
}

/// Capabilities advertised by a backend.
#[derive(Debug, Clone)]
pub struct Capabilities {
    pub supported_ops: Vec<OpKind>,
    pub supports_deterministic_mode: bool,
    pub supports_async: bool,
    pub max_tensor_bytes: u64,
}
