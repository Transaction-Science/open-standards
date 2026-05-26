//! Boot-time validation across all registered backends.
//!
//! For every (op, candidate-backend) pair in the runtime's kernel
//! registry, verify the candidate produces output equivalent to the
//! reference within a documented FP tolerance for a battery of input
//! shapes. Drift beyond tolerance is a build-breaking failure.
//!
//! **Contract.** Vendor-tiled GEMM kernels (Apple Accelerate, AppleAmx,
//! cuBLAS, any optimised BLAS) reassociate partial sums; their output
//! cannot be bit-identical to a scalar reference. Their honest contract
//! is "deterministic within a backend, numerically equivalent across
//! backends within FP tolerance." A real correctness bug — wrong
//! transpose, wrong shape, swapped indices — produces drift orders of
//! magnitude beyond `FpTolerance::matmul_default()` (1e-3 abs/rel) and
//! is still caught here; the prior strict bit-identity contract was
//! unattainable for any FP kernel and was just lying.
//!
//! On Linux this test usually finds zero non-reference candidates and
//! passes trivially. On Apple Silicon it exercises every accelerated
//! kernel registered by `jouleclaw-backend-apple`.

use jouleclaw_core::kernel::Kernel;
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta};
use jouleclaw_runtime::{
    validate_against_reference_fp, FpTolerance, Runtime, ValidationStatus,
};
use std::sync::Arc;

fn det_random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (s >> 40) as u32;
        (bits as f32) * (1.0 / (1u32 << 24) as f32) - 0.5
    }).collect()
}

#[test]
fn every_backend_matches_reference_for_matmul() {
    let runtime = Runtime::boot();
    let ref_backend = jouleclaw_backend_reference::BACKEND_ID;
    let kernels: Vec<Arc<dyn Kernel>> = runtime.kernels.iter().cloned().collect();

    let reference = kernels.iter()
        .find(|k| k.op_kind() == OpKind::MatMul && k.backend() == ref_backend)
        .expect("reference MatMul must be registered");

    let candidates: Vec<&Arc<dyn Kernel>> = kernels.iter()
        .filter(|k| k.op_kind() == OpKind::MatMul && k.backend() != ref_backend)
        .collect();

    println!("Found {} non-reference MatMul candidates", candidates.len());

    // Battery of shapes covering the typical inference patterns.
    let shapes = [
        (4, 4, 4),       // tiny smoke test
        (8, 16, 8),      // small square-ish
        (1, 512, 512),   // single-token ffn proj
        (32, 64, 128),   // batched
    ];

    let attrs = OpAttrs::MatMul { transpose_a: false, transpose_b: false, alpha: 1.0, b_n_valid: None };
    let tol = FpTolerance::matmul_default();

    for cand in candidates {
        for &(m, k, n) in &shapes {
            let a = Tensor::from_f32(
                TensorMeta::new(Dtype::F32, &[m, k]), &det_random(m * k, 1));
            let b = Tensor::from_f32(
                TensorMeta::new(Dtype::F32, &[k, n]), &det_random(k * n, 2));
            let out_meta = TensorMeta::new(Dtype::F32, &[m, n]);

            let report = validate_against_reference_fp(
                cand, reference, &attrs, &[&a, &b], out_meta, None, tol);

            // Pass on bit-identity OR drift within FP tolerance.
            // Anything else — kernel error, drift beyond tolerance — is
            // a build-breaking failure that catches real bugs.
            match &report.status {
                ValidationStatus::Identical => {
                    println!("OK  {:?} matmul[{}x{}x{}]: bit-identical",
                        cand.backend(), m, k, n);
                }
                ValidationStatus::ToleratedDrift {
                    max_abs_diff_f32, max_rel_diff_f32 } => {
                    println!("OK  {:?} matmul[{}x{}x{}]: drift within FP tolerance \
                              (max |diff|={:.3e}, max rel={:.3e}; tol abs={:.0e}, rel={:.0e})",
                        cand.backend(), m, k, n,
                        max_abs_diff_f32, max_rel_diff_f32, tol.max_abs, tol.max_rel);
                }
                _ => panic!("VALIDATION FAILED for {:?} on shape {}x{}x{}: {}",
                    cand.backend(), m, k, n, report.pretty()),
            }
        }
    }
}
