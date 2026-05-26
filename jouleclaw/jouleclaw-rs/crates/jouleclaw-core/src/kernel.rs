//! Kernels are concrete implementations of operations on specific backends.
//!
//! `Op` is the abstract operation. `Kernel` is what actually runs.
//! A backend (Apple CPU+AMX, Apple GPU/Metal, ANE, x86 AVX-512, NVIDIA CUDA, ...)
//! registers kernels for the ops it can execute.

use crate::backend::BackendId;
use crate::determinism::DeterminismClass;
use crate::energy::JouleMeasurement;
use crate::error::ExecutionError;
use crate::op::{OpAttrs, OpKind};
use crate::tensor::{TensorView, TensorViewMut};
use std::time::Duration;

/// Stable identifier for a registered kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelId(pub u64);

/// Result of a kernel execution.
#[derive(Debug, Clone)]
pub struct KernelResult {
    pub joules: JouleMeasurement,
    pub wall_clock: Duration,
    pub bytes_read: u64,
    pub bytes_written: u64,
}

/// Execution context passed to kernels.
pub struct ExecutionContext<'a> {
    pub backend: BackendId,
    pub deterministic: bool,
    pub seed: Option<u64>,
    pub scratch: &'a mut [u8],
}

/// How strongly a kernel wants to handle a specific op invocation.
/// Higher variants win during picking; `Refuse` removes the kernel
/// from consideration for this call entirely.
///
/// The default impl of [`Kernel::prefers`] returns `Acceptable`, so
/// kernels that don't care about shape inherit the previous "any
/// non-reference backend wins, else reference" behaviour. Vendor
/// kernels with per-call dispatch overhead (Apple Accelerate /
/// cuBLAS / AppleAmx) should return `Weak` for shapes where the
/// reference scalar loop is empirically faster — the picker then
/// falls through to reference automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KernelPreference {
    /// "Don't pick me for this call."
    Refuse = 0,
    /// "Reference is probably faster than me here."
    Weak = 1,
    /// Default — no opinion.
    Acceptable = 2,
    /// "This is exactly the shape I'm fast at."
    Strong = 3,
}

/// The contract every concrete kernel must satisfy.
pub trait Kernel: Send + Sync {
    fn op_kind(&self) -> OpKind;
    fn backend(&self) -> BackendId;
    fn determinism(&self) -> DeterminismClass;

    /// Shape-aware preference for this specific op invocation. Default
    /// `Acceptable`. Override to express e.g. "AppleAmx is slow on
    /// tiny matmuls due to dispatch overhead." The picker uses
    /// `prefers` to rank candidate kernels for each Op node at
    /// compile time.
    fn prefers(
        &self,
        _attrs: &crate::op::OpAttrs,
        _input_metas: &[&crate::tensor::TensorMeta],
    ) -> KernelPreference {
        KernelPreference::Acceptable
    }

    /// Execute the kernel. The engine passes the op's attributes alongside
    /// the input/output views.
    fn execute(
        &self,
        ctx: &mut ExecutionContext<'_>,
        attrs: &OpAttrs,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorViewMut<'_>],
    ) -> Result<KernelResult, ExecutionError>;
}
