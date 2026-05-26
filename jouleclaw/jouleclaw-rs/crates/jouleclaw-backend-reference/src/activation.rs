//! Reference element-wise activation.

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{ActivationKind, OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

pub struct ActivationRef;

impl Kernel for ActivationRef {
    fn op_kind(&self) -> OpKind { OpKind::Activation }
    fn backend(&self) -> BackendId { BACKEND_ID }
    fn determinism(&self) -> DeterminismClass { DeterminismClass::Deterministic }

    fn execute(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        attrs: &OpAttrs,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorViewMut<'_>],
    ) -> Result<KernelResult, ExecutionError> {
        let start = Instant::now();
        let kind = match attrs {
            OpAttrs::Activation { kind } => *kind,
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::Activation, backend: BACKEND_ID,
                reason: "Activation kernel requires OpAttrs::Activation".into(),
            }),
        };
        let x = inputs[0].as_f32_vec();
        let mut y = vec![0f32; x.len()];
        for i in 0..x.len() {
            y[i] = match kind {
                ActivationKind::SiLU => x[i] * sigmoid(x[i]),
                ActivationKind::GELU => 0.5 * x[i] * (1.0 + libm_tanh(
                    0.7978845608_f32 * (x[i] + 0.044715 * x[i] * x[i] * x[i])
                )),
                ActivationKind::ReLU => if x[i] > 0.0 { x[i] } else { 0.0 },
                ActivationKind::Tanh => libm_tanh(x[i]),
                ActivationKind::Sigmoid => sigmoid(x[i]),
            };
        }
        outputs[0].write_f32(&y);
        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: (x.len() as f64) * 3e-10,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: (x.len() * 4) as u64,
            bytes_written: (y.len() * 4) as u64,
        })
    }
}

#[inline] fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// `f32::tanh` is in std but not always considered deterministic across
/// platforms; for Phase 1.1 we trust the platform. Phase 2+ swap for a
/// deterministic series implementation.
#[inline] fn libm_tanh(x: f32) -> f32 { x.tanh() }
