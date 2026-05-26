//! Reference element-wise Add and Mul.

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

pub struct AddRef;

impl Kernel for AddRef {
    fn op_kind(&self) -> OpKind { OpKind::Add }
    fn backend(&self) -> BackendId { BACKEND_ID }
    fn determinism(&self) -> DeterminismClass { DeterminismClass::Deterministic }

    fn execute(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _attrs: &OpAttrs,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorViewMut<'_>],
    ) -> Result<KernelResult, ExecutionError> {
        elementwise(OpKind::Add, inputs, outputs, |a, b| a + b)
    }
}

pub struct MulRef;

impl Kernel for MulRef {
    fn op_kind(&self) -> OpKind { OpKind::Mul }
    fn backend(&self) -> BackendId { BACKEND_ID }
    fn determinism(&self) -> DeterminismClass { DeterminismClass::Deterministic }

    fn execute(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _attrs: &OpAttrs,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorViewMut<'_>],
    ) -> Result<KernelResult, ExecutionError> {
        elementwise(OpKind::Mul, inputs, outputs, |a, b| a * b)
    }
}

fn elementwise<F: Fn(f32, f32) -> f32>(
    op: OpKind,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorViewMut<'_>],
    f: F,
) -> Result<KernelResult, ExecutionError> {
    let start = Instant::now();
    let a = inputs[0].as_f32_vec();
    let b = inputs[1].as_f32_vec();
    if a.len() != b.len() {
        return Err(ExecutionError::KernelFailed {
            op, backend: BACKEND_ID,
            reason: format!("shape mismatch: {} vs {}", a.len(), b.len()),
        });
    }
    let mut y = vec![0f32; a.len()];
    for i in 0..a.len() { y[i] = f(a[i], b[i]); }
    outputs[0].write_f32(&y);
    let elapsed = start.elapsed();
    Ok(KernelResult {
        joules: JouleMeasurement {
            joules: (a.len() as f64) * 1e-10,
            energy_source: EnergySourceId(0),
            measurement_window: elapsed,
            attribution_confidence: 0.0,
        },
        wall_clock: elapsed,
        bytes_read: ((a.len() + b.len()) * 4) as u64,
        bytes_written: (y.len() * 4) as u64,
    })
}
