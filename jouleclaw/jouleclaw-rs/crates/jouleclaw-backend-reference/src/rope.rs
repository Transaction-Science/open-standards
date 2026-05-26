//! Reference rotary position embedding (RoPE).
//!
//! Adjacent-pair convention (matches Llama 1/2 GGUF):
//! For input `x` of shape `[..., seq, d]` (d even), each consecutive pair
//! `(x[2i], x[2i+1])` at sequence position `p` is rotated by an angle
//! `m = (p + position_offset) * theta_i`, where
//! `theta_i = base^(-2i / d)`.
//!
//! The rotation:
//!   y[2i]   = x[2i] * cos(m) - x[2i+1] * sin(m)
//!   y[2i+1] = x[2i] * sin(m) + x[2i+1] * cos(m)
//!
//! Phase 1.7 implements the adjacent-pair form; some models (GPT-NeoX,
//! Llama with `neox=true`) use the half-split form. Adding it is a small
//! delta — switch the index pattern from `(2i, 2i+1)` to `(i, i+d/2)`.

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

pub struct RopeRef;

impl Kernel for RopeRef {
    fn op_kind(&self) -> OpKind { OpKind::Rope }
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
        let (base, mut pos_offset) = match attrs {
            OpAttrs::Rope { base, position_offset } => (*base, *position_offset as usize),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::Rope, backend: BACKEND_ID,
                reason: "Rope kernel requires OpAttrs::Rope".into(),
            }),
        };
        // Dynamic position: a second input (I32 [1]) overrides the
        // build-time attr. This is what makes the decode graph
        // compile-once / reuse-every-step (rope position is the only
        // per-token variable).
        if inputs.len() >= 2 {
            let b = inputs[1].bytes;
            if b.len() >= 4 {
                pos_offset = i32::from_le_bytes([b[0], b[1], b[2], b[3]]).max(0) as usize;
            }
        }

        let x = inputs[0].as_f32_vec();
        let shape = &inputs[0].meta.shape;
        if shape.len() < 2 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Rope, backend: BACKEND_ID,
                reason: format!("rope requires rank >= 2, got {:?}", shape),
            });
        }
        let d = shape[shape.len() - 1];
        let seq = shape[shape.len() - 2];
        if d % 2 != 0 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Rope, backend: BACKEND_ID,
                reason: format!("rope d must be even, got {}", d),
            });
        }
        let outer: usize = shape.iter().take(shape.len() - 2).product::<usize>().max(1);

        let mut y = vec![0f32; x.len()];
        for o in 0..outer {
            for s in 0..seq {
                let p = (s + pos_offset) as f32;
                let row_off = o * seq * d + s * d;
                for i in 0..d / 2 {
                    let theta = base.powf(-2.0 * (i as f32) / (d as f32));
                    let m = p * theta;
                    let cos = m.cos();
                    let sin = m.sin();
                    let x0 = x[row_off + 2 * i];
                    let x1 = x[row_off + 2 * i + 1];
                    y[row_off + 2 * i]     = x0 * cos - x1 * sin;
                    y[row_off + 2 * i + 1] = x0 * sin + x1 * cos;
                }
            }
        }
        outputs[0].write_f32(&y);

        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: (x.len() as f64) * 1e-9,
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
