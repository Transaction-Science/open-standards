//! Validation harness tests.
//!
//! Two things must be true:
//! 1. A correct kernel matches the reference bit-for-bit -> Identical.
//! 2. An incorrect kernel is detected and reported with diagnostic info -> Drift.

use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorView, TensorViewMut};
use jouleclaw_runtime::{validate_against_reference, ValidationStatus};
use std::sync::Arc;
use std::time::Duration;

/// A deliberately-broken matmul kernel that perturbs one output element.
/// Same backend ID family as reference but tagged Custom(99) for clarity.
struct BrokenMatMul;

impl Kernel for BrokenMatMul {
    fn op_kind(&self) -> OpKind { OpKind::MatMul }
    fn backend(&self) -> BackendId { BackendId::Custom(99) }
    fn determinism(&self) -> DeterminismClass { DeterminismClass::Deterministic }

    fn execute(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _attrs: &OpAttrs,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorViewMut<'_>],
    ) -> Result<KernelResult, ExecutionError> {
        // Compute correctly, then perturb output[0] by a small amount.
        let a = inputs[0].as_f32_vec();
        let b = inputs[1].as_f32_vec();
        let a_shape = &inputs[0].meta.shape;
        let b_shape = &inputs[1].meta.shape;
        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[1];

        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for kk in 0..k { acc += a[i * k + kk] * b[kk * n + j]; }
                c[i * n + j] = acc;
            }
        }
        // The drift: nudge one element by epsilon.
        if !c.is_empty() { c[0] += 1e-6; }

        outputs[0].write_f32(&c);
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: 0.0, energy_source: EnergySourceId(0),
                measurement_window: Duration::from_secs(0),
                attribution_confidence: 0.0,
            },
            wall_clock: Duration::from_secs(0),
            bytes_read: 0, bytes_written: (c.len() * 4) as u64,
        })
    }
}

fn make_inputs(m: usize, k: usize, n: usize, seed: u64) -> (Tensor, Tensor) {
    fn det_random(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (s >> 40) as u32;
            (bits as f32) * (1.0 / (1u32 << 24) as f32) - 0.5
        }).collect()
    }
    let a = Tensor::from_f32(TensorMeta::new(Dtype::F32, &[m, k]), &det_random(m * k, seed));
    let b = Tensor::from_f32(TensorMeta::new(Dtype::F32, &[k, n]), &det_random(k * n, seed + 1));
    (a, b)
}

#[test]
fn correct_kernel_passes_validation() {
    let reference: Arc<dyn Kernel> = Arc::new(jouleclaw_backend_reference::MatMulRef);
    // The same kernel under a different backend tag must pass validation.
    let candidate: Arc<dyn Kernel> = Arc::new(jouleclaw_backend_reference::MatMulRef);

    let (a, b) = make_inputs(8, 16, 8, 7);
    let attrs = OpAttrs::MatMul { transpose_a: false, transpose_b: false, alpha: 1.0, b_n_valid: None };
    let out_meta = TensorMeta::new(Dtype::F32, &[8, 8]);

    let report = validate_against_reference(
        &candidate, &reference, &attrs, &[&a, &b], out_meta, None);

    assert!(report.passed(), "expected pass, got: {}", report.pretty());
    println!("{}", report.pretty());
}

#[test]
fn broken_kernel_is_detected_with_diagnostics() {
    let reference: Arc<dyn Kernel> = Arc::new(jouleclaw_backend_reference::MatMulRef);
    let broken: Arc<dyn Kernel> = Arc::new(BrokenMatMul);

    let (a, b) = make_inputs(8, 16, 8, 7);
    let attrs = OpAttrs::MatMul { transpose_a: false, transpose_b: false, alpha: 1.0, b_n_valid: None };
    let out_meta = TensorMeta::new(Dtype::F32, &[8, 8]);

    let report = validate_against_reference(
        &broken, &reference, &attrs, &[&a, &b], out_meta, None);

    assert!(!report.passed(), "broken kernel should fail validation");

    match report.status {
        ValidationStatus::Drift { first_diff_byte, differing_bytes,
            max_abs_diff_f32, max_rel_diff_f32 } => {
            // Drift is in element 0 (bytes 0..4).
            assert!(first_diff_byte < 4,
                "drift should appear in first f32 element, got byte {}", first_diff_byte);
            assert!(differing_bytes >= 1, "at least one byte must differ");
            // The perturbation we injected was 1e-6. F32 rounding can shift
            // the observed delta slightly; allow ±10%.
            let abs = max_abs_diff_f32.expect("F32 diff should be reported");
            assert!(abs > 5e-7 && abs < 2e-6,
                "max abs diff should be ~1e-6 (the injected perturbation); got {:.3e}", abs);
            assert!(max_rel_diff_f32.is_some());
        }
        ref other => panic!("expected Drift, got {:?}", other),
    }

    println!("{}", report.pretty());
}
