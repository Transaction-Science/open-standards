//! Reference softmax along the last axis.
//!
//! Numerically stable: subtract row max before exp.
//! Fixed iteration order. No parallel reductions.

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

pub struct SoftmaxRef;

impl Kernel for SoftmaxRef {
    fn op_kind(&self) -> OpKind { OpKind::Softmax }
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
        let (causal, mut causal_offset) = match attrs {
            OpAttrs::Softmax { axis, causal, causal_offset }
                if *axis == -1 || *axis == (inputs[0].meta.shape.len() as i32 - 1) =>
                (*causal, *causal_offset),
            OpAttrs::Softmax { .. } => return Err(ExecutionError::KernelFailed {
                op: OpKind::Softmax, backend: BACKEND_ID,
                reason: "only last-axis softmax supported".into(),
            }),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::Softmax, backend: BACKEND_ID,
                reason: "Softmax kernel requires OpAttrs::Softmax".into(),
            }),
        };
        // Dynamic causal offset: a second input (I32 [1]) overrides
        // the attr. Lets one compiled decode graph serve every step.
        if inputs.len() >= 2 {
            let b = inputs[1].bytes;
            if b.len() >= 4 {
                causal_offset = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }

        let x = inputs[0].as_f32_vec();
        let shape = &inputs[0].meta.shape;
        let last = shape[shape.len() - 1];
        let outer: usize = shape.iter().take(shape.len() - 1).product::<usize>().max(1);

        // For causal masking, we treat each row of the last two dims as one
        // softmax row indexed by query position. shape[-2] is `seq_q`; the
        // query at relative position q within that block can attend to keys
        // 0..=q + causal_offset.
        let seq_q = if shape.len() >= 2 { shape[shape.len() - 2] } else { 1 };
        if causal && (last as i32) < (seq_q as i32 + causal_offset) {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Softmax, backend: BACKEND_ID,
                reason: format!(
                    "causal softmax: key dim ({}) must be >= seq_q ({}) + causal_offset ({})",
                    last, seq_q, causal_offset),
            });
        }

        let mut y = vec![0f32; x.len()];
        for row in 0..outer {
            let base = row * last;
            // Query position within the attention block.
            let q_rel = row % seq_q;
            // Causal cutoff: this query may attend to keys 0..=cutoff (inclusive).
            let cutoff: i32 = q_rel as i32 + causal_offset;

            // Stable max over visible positions.
            let mut max = f32::NEG_INFINITY;
            for j in 0..last {
                if causal && (j as i32) > cutoff { continue; }
                if x[base + j] > max { max = x[base + j]; }
            }
            if max == f32::NEG_INFINITY { max = 0.0; }

            let mut sum = 0f32;
            for j in 0..last {
                if causal && (j as i32) > cutoff {
                    y[base + j] = 0.0;
                } else {
                    let e = (x[base + j] - max).exp();
                    y[base + j] = e;
                    sum += e;
                }
            }
            for j in 0..last {
                if !(causal && (j as i32) > cutoff) {
                    y[base + j] /= sum;
                }
            }
        }
        outputs[0].write_f32(&y);
        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: (x.len() as f64) * 5e-10,
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
