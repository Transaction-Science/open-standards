//! # jouleclaw-backend-reference
//!
//! Pure-Rust scalar reference kernels. F32 only.
//!
//! These kernels are slow by design. They exist to:
//! 1. Provide a portable fallback that runs on any platform.
//! 2. Serve as the *determinism oracle*: every other backend's deterministic
//!    kernel must produce bit-identical output to this one for the same
//!    inputs.
//! 3. Document the canonical semantics of each operation.
//!
//! Reduction order, accumulation precision, and tie-breaking are all fixed.
//! No parallelism. No SIMD. No fused operations.

mod matmul;
mod matmul_ternary;
mod matmul_bit;
#[allow(non_snake_case)]
mod matmul_stq1_0;
mod matmul_q8_0;
mod lookup_ternary;
mod lookup_bit;
mod conv1d;
mod softmax;
mod norm;
mod activation;
mod elementwise;
mod lookup;
mod sample;
mod shape;
mod rope;

pub use matmul::MatMulRef;
pub use matmul_ternary::MatMulTernaryRef;
pub use matmul_bit::MatMulBitRef;
pub use matmul_stq1_0::MatMulSTQ1_0Ref;
pub use matmul_q8_0::MatMulQ80Ref;
pub use lookup_ternary::LookupTernaryRef;
pub use lookup_bit::LookupBitRef;
pub use conv1d::Conv1DDepthwiseCausalRef;
pub use softmax::SoftmaxRef;
pub use norm::NormRef;
pub use activation::ActivationRef;
pub use elementwise::{AddRef, MulRef};
pub use lookup::LookupRef;
pub use sample::SampleRef;
pub use shape::{ReshapeRef, TransposeRef, ConcatRef, RepeatRef, ScatterRef, SliceRef};
pub use rope::RopeRef;

use jouleclaw_core::backend::BackendId;
use jouleclaw_core::kernel::Kernel;
use std::sync::Arc;

/// The reference backend identifier — a `Custom` slot reserved for it.
pub const BACKEND_ID: BackendId = BackendId::Custom(0);

/// Return all reference kernels, ready to register with a runtime.
pub fn all_kernels() -> Vec<Arc<dyn Kernel>> {
    vec![
        Arc::new(MatMulRef),
        Arc::new(MatMulTernaryRef),
        Arc::new(MatMulBitRef),
        Arc::new(MatMulSTQ1_0Ref),
        Arc::new(MatMulQ80Ref),
        Arc::new(LookupTernaryRef),
        Arc::new(LookupBitRef),
        Arc::new(Conv1DDepthwiseCausalRef),
        Arc::new(SoftmaxRef),
        Arc::new(NormRef),
        Arc::new(ActivationRef),
        Arc::new(AddRef),
        Arc::new(MulRef),
        Arc::new(LookupRef),
        Arc::new(SampleRef),
        Arc::new(ReshapeRef),
        Arc::new(TransposeRef),
        Arc::new(ConcatRef),
        Arc::new(RepeatRef),
        Arc::new(ScatterRef),
        Arc::new(SliceRef),
        Arc::new(RopeRef),
    ]
}
