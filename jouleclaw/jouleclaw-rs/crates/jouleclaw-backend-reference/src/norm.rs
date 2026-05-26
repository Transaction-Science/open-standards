//! Reference normalization. RMSNorm (Llama-style) and LayerNorm.
//!
//! `(x: [..,d], weight: [d]) -> [..,d]`
//!
//! RMS:   y = x / sqrt(mean(x^2) + eps) * weight
//! Layer: y = (x - mean) / sqrt(var + eps) * weight

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{NormKind, OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

pub struct NormRef;

impl Kernel for NormRef {
    fn op_kind(&self) -> OpKind { OpKind::Norm }
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
        let (kind, eps) = match attrs {
            OpAttrs::Norm { kind, eps } => (*kind, *eps),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::Norm, backend: BACKEND_ID,
                reason: "Norm kernel requires OpAttrs::Norm".into(),
            }),
        };

        let x = inputs[0].as_f32_vec();
        let w = inputs[1].as_f32_vec();
        let shape = &inputs[0].meta.shape;
        let last = shape[shape.len() - 1];
        let outer: usize = shape.iter().take(shape.len() - 1).product::<usize>().max(1);

        let mut y = vec![0f32; x.len()];
        for row in 0..outer {
            let base = row * last;
            match kind {
                NormKind::Rms => {
                    let mut sumsq = 0f32;
                    for j in 0..last { sumsq += x[base + j] * x[base + j]; }
                    let denom = ((sumsq / last as f32) + eps).sqrt();
                    for j in 0..last {
                        y[base + j] = (x[base + j] / denom) * w[j];
                    }
                }
                NormKind::Layer => {
                    let mut sum = 0f32;
                    for j in 0..last { sum += x[base + j]; }
                    let mean = sum / last as f32;
                    let mut var = 0f32;
                    for j in 0..last {
                        let d = x[base + j] - mean;
                        var += d * d;
                    }
                    let denom = ((var / last as f32) + eps).sqrt();
                    for j in 0..last {
                        y[base + j] = ((x[base + j] - mean) / denom) * w[j];
                    }
                }
            }
        }
        outputs[0].write_f32(&y);
        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: (x.len() as f64) * 8e-10,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: ((x.len() + w.len()) * 4) as u64,
            bytes_written: (y.len() * 4) as u64,
        })
    }
}
