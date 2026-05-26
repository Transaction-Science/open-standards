//! Kernel validation harness.
//!
//! Every kernel for every backend must produce bit-identical output to the
//! reference backend for the same inputs (in deterministic mode). This module
//! provides the harness for that check.
//!
//! Used at:
//! - CI time: every PR runs validation across all registered kernels.
//! - Boot time (optional): runtime validates new backends against the
//!   reference oracle before accepting them into the kernel registry.

use jouleclaw_core::backend::BackendId;
use jouleclaw_core::error::{ExecutionError, Result};
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::OpAttrs;
use jouleclaw_core::tensor::{Tensor, TensorMeta, TensorView, TensorViewMut};
use std::sync::Arc;

/// Result of validating one kernel against a reference.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub kernel_backend: BackendId,
    pub reference_backend: BackendId,
    pub status: ValidationStatus,
    /// Bytes the kernel produced (for diagnostic output).
    pub kernel_bytes: Vec<u8>,
    /// Bytes the reference produced.
    pub reference_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum ValidationStatus {
    /// Bit-identical output. Kernel passes the strict determinism contract.
    Identical,
    /// Outputs differ but the differences are within an explicit
    /// floating-point tolerance. Vendor-optimised matmul kernels
    /// (Accelerate, cuBLAS, AppleAmx, any tiled GEMM) reassociate
    /// partial sums and cannot produce bit-identical output to a
    /// scalar reference; their honest contract is "deterministic
    /// within a backend, numerically equivalent across backends within
    /// FP tolerance." A real correctness bug exceeds any sensible
    /// tolerance by orders of magnitude, so this is still a strong
    /// gate.
    ToleratedDrift {
        max_abs_diff_f32: f32,
        max_rel_diff_f32: f32,
    },
    /// Outputs differ beyond tolerance. The kernel violates the contract.
    Drift {
        /// Index of the first differing byte.
        first_diff_byte: usize,
        /// Number of bytes that differ.
        differing_bytes: usize,
        /// Maximum absolute difference, when interpreting as F32.
        max_abs_diff_f32: Option<f32>,
        /// Maximum relative difference, when interpreting as F32.
        max_rel_diff_f32: Option<f32>,
    },
    /// Kernel returned an error during execution.
    KernelError { message: String },
    /// Reference returned an error during execution.
    ReferenceError { message: String },
}

/// Floating-point tolerance for cross-backend numerical equivalence.
/// Both bounds must hold for `ToleratedDrift` classification.
#[derive(Debug, Clone, Copy)]
pub struct FpTolerance {
    pub max_abs: f32,
    pub max_rel: f32,
}

impl FpTolerance {
    /// Sensible default for f32 matmul on shapes up to ~512×512: ULP
    /// reassociation drift on N partial sums grows ~N·2^-23, ≤ ~6e-5
    /// for N=512. We allow a margin: 1e-3 abs / 1e-3 rel. Real bugs
    /// (transposed indices, wrong attrs, etc.) show drift in the
    /// 0.1–1.0 range, easily caught.
    pub const fn matmul_default() -> Self {
        Self { max_abs: 1e-3, max_rel: 1e-3 }
    }
}

impl ValidationReport {
    /// `true` iff outputs were bit-identical OR within the FP
    /// tolerance the caller supplied. Strict bit-identity callers
    /// should check `matches!(.status, ValidationStatus::Identical)`
    /// directly.
    pub fn passed(&self) -> bool {
        matches!(self.status,
            ValidationStatus::Identical | ValidationStatus::ToleratedDrift { .. })
    }

    pub fn pretty(&self) -> String {
        match &self.status {
            ValidationStatus::Identical => format!(
                "OK  {:?} matches {:?} (bit-identical, {} bytes)",
                self.kernel_backend, self.reference_backend, self.kernel_bytes.len()
            ),
            ValidationStatus::ToleratedDrift { max_abs_diff_f32, max_rel_diff_f32 } => format!(
                "OK  {:?} matches {:?} within FP tolerance (max |diff|={:.3e}, max rel={:.3e})",
                self.kernel_backend, self.reference_backend,
                max_abs_diff_f32, max_rel_diff_f32),
            ValidationStatus::Drift { first_diff_byte, differing_bytes,
                max_abs_diff_f32, max_rel_diff_f32 } => {
                let mut s = format!(
                    "FAIL  {:?} drifts from {:?}: first diff at byte {}, {} bytes differ",
                    self.kernel_backend, self.reference_backend,
                    first_diff_byte, differing_bytes
                );
                if let Some(d) = max_abs_diff_f32 {
                    s.push_str(&format!(", max |diff| (f32): {:.6e}", d));
                }
                if let Some(d) = max_rel_diff_f32 {
                    s.push_str(&format!(", max rel diff (f32): {:.6e}", d));
                }
                s
            }
            ValidationStatus::KernelError { message } => format!(
                "FAIL  {:?} kernel error: {}", self.kernel_backend, message),
            ValidationStatus::ReferenceError { message } => format!(
                "ERROR {:?} reference error: {}", self.reference_backend, message),
        }
    }
}

/// Run a single kernel on the given inputs, returning the output tensor and
/// the kernel's reported result metadata.
pub fn run_kernel(
    kernel: &dyn Kernel,
    attrs: &OpAttrs,
    inputs: &[&Tensor],
    output_meta: TensorMeta,
    seed: Option<u64>,
) -> std::result::Result<(Tensor, KernelResult), ExecutionError> {
    let mut output = Tensor::zeros(output_meta);
    let mut scratch = vec![0u8; 64 * 1024];

    let in_views: Vec<TensorView<'_>> = inputs.iter().map(|t| t.view()).collect();

    let storage = std::sync::Arc::get_mut(&mut output.storage)
        .expect("fresh tensor storage must be uniquely owned");
    let mut out_view = TensorViewMut {
        meta: &output.meta,
        bytes: &mut storage.bytes,
    };

    let mut ctx = ExecutionContext {
        backend: kernel.backend(),
        deterministic: true,
        seed,
        scratch: &mut scratch,
    };

    let result = kernel.execute(
        &mut ctx,
        attrs,
        &in_views,
        std::slice::from_mut(&mut out_view),
    )?;

    Ok((output, result))
}

/// Validate one candidate kernel against a reference kernel for the same op.
///
/// Both kernels are run with identical inputs; the candidate is expected to
/// produce bit-identical output.
pub fn validate_against_reference(
    candidate: &Arc<dyn Kernel>,
    reference: &Arc<dyn Kernel>,
    attrs: &OpAttrs,
    inputs: &[&Tensor],
    output_meta: TensorMeta,
    seed: Option<u64>,
) -> ValidationReport {
    assert_eq!(candidate.op_kind(), reference.op_kind(),
        "validate_against_reference: op kind mismatch");

    let kernel_backend = candidate.backend();
    let reference_backend = reference.backend();

    let ref_result = run_kernel(reference.as_ref(), attrs, inputs, output_meta.clone(), seed);
    let ref_tensor = match ref_result {
        Ok((t, _)) => t,
        Err(e) => return ValidationReport {
            kernel_backend, reference_backend,
            status: ValidationStatus::ReferenceError { message: format!("{:?}", e) },
            kernel_bytes: Vec::new(),
            reference_bytes: Vec::new(),
        },
    };

    let cand_result = run_kernel(candidate.as_ref(), attrs, inputs, output_meta, seed);
    let cand_tensor = match cand_result {
        Ok((t, _)) => t,
        Err(e) => return ValidationReport {
            kernel_backend, reference_backend,
            status: ValidationStatus::KernelError { message: format!("{:?}", e) },
            kernel_bytes: Vec::new(),
            reference_bytes: ref_tensor.storage.bytes.clone(),
        },
    };

    let kbytes = cand_tensor.storage.bytes.clone();
    let rbytes = ref_tensor.storage.bytes.clone();

    let status = if kbytes == rbytes {
        ValidationStatus::Identical
    } else {
        diff_status(&kbytes, &rbytes, &cand_tensor.meta)
    };

    ValidationReport {
        kernel_backend, reference_backend,
        status,
        kernel_bytes: kbytes,
        reference_bytes: rbytes,
    }
}

/// Like [`validate_against_reference`] but with an explicit
/// floating-point tolerance for cross-backend equivalence. Drift
/// within `tol` is classified as [`ValidationStatus::ToleratedDrift`]
/// (a passing status); drift beyond `tol` stays
/// [`ValidationStatus::Drift`] (failing). This is the correct contract
/// for vendor-tiled GEMM kernels — they reassociate partial sums and
/// will never be bit-identical to a scalar reference, but real bugs
/// produce drift orders of magnitude beyond any sensible tolerance and
/// are still caught.
pub fn validate_against_reference_fp(
    candidate: &Arc<dyn Kernel>,
    reference: &Arc<dyn Kernel>,
    attrs: &OpAttrs,
    inputs: &[&Tensor],
    output_meta: TensorMeta,
    seed: Option<u64>,
    tol: FpTolerance,
) -> ValidationReport {
    let mut report = validate_against_reference(
        candidate, reference, attrs, inputs, output_meta, seed);
    if let ValidationStatus::Drift {
        max_abs_diff_f32: Some(abs),
        max_rel_diff_f32: Some(rel), .. } = report.status
    {
        if abs <= tol.max_abs && rel <= tol.max_rel {
            report.status = ValidationStatus::ToleratedDrift {
                max_abs_diff_f32: abs,
                max_rel_diff_f32: rel,
            };
        }
    }
    report
}

fn diff_status(k: &[u8], r: &[u8], meta: &TensorMeta) -> ValidationStatus {
    let len = k.len().min(r.len());
    let first_diff_byte = (0..len).find(|&i| k[i] != r[i]).unwrap_or(len);
    let differing_bytes = (0..len).filter(|&i| k[i] != r[i]).count()
        + k.len().abs_diff(r.len());

    // If F32 dtype, compute float-level diffs for diagnostic output.
    let (max_abs, max_rel) = if meta.dtype == jouleclaw_core::tensor::Dtype::F32
        && k.len() % 4 == 0 && r.len() % 4 == 0 && k.len() == r.len() {
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for i in (0..k.len()).step_by(4) {
            let mut bk = [0u8; 4]; bk.copy_from_slice(&k[i..i+4]);
            let mut br = [0u8; 4]; br.copy_from_slice(&r[i..i+4]);
            let fk = f32::from_le_bytes(bk);
            let fr = f32::from_le_bytes(br);
            let d = (fk - fr).abs();
            if d > max_abs { max_abs = d; }
            let denom = fr.abs().max(1e-12);
            let rd = d / denom;
            if rd > max_rel { max_rel = rd; }
        }
        (Some(max_abs), Some(max_rel))
    } else {
        (None, None)
    };

    ValidationStatus::Drift {
        first_diff_byte,
        differing_bytes,
        max_abs_diff_f32: max_abs,
        max_rel_diff_f32: max_rel,
    }
}

/// Validate every (candidate, reference) pair across a kernel set, for a
/// fixed op + attrs + inputs.
///
/// Useful for end-of-build CI: register all kernels, run this for every
/// op and a battery of input shapes, fail the build on any drift.
pub fn validate_all_against_reference<'a>(
    op_kernels: &'a [Arc<dyn Kernel>],
    reference_backend: BackendId,
    attrs: &OpAttrs,
    inputs: &[&Tensor],
    output_meta: TensorMeta,
    seed: Option<u64>,
) -> Result<Vec<ValidationReport>> {
    let reference = op_kernels.iter()
        .find(|k| k.backend() == reference_backend)
        .ok_or_else(|| jouleclaw_core::error::Error::Execution(
            ExecutionError::KernelFailed {
                op: op_kernels[0].op_kind(),
                backend: reference_backend,
                reason: "reference backend not in kernel set".into(),
            }))?;

    let mut reports = Vec::new();
    for cand in op_kernels {
        if cand.backend() == reference_backend { continue; }
        reports.push(validate_against_reference(
            cand, reference, attrs, inputs, output_meta.clone(), seed,
        ));
    }
    Ok(reports)
}
