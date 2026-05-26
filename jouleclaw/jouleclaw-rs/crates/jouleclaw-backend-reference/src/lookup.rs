//! Reference Lookup: index a [V, d] table with [n] indices, producing [n, d].

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{Dtype, TensorView, TensorViewMut};
use std::time::Instant;

pub struct LookupRef;

impl Kernel for LookupRef {
    fn op_kind(&self) -> OpKind { OpKind::Lookup }
    fn backend(&self) -> BackendId { BACKEND_ID }
    fn determinism(&self) -> DeterminismClass { DeterminismClass::Deterministic }

    fn execute(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _attrs: &OpAttrs,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorViewMut<'_>],
    ) -> Result<KernelResult, ExecutionError> {
        let start = Instant::now();
        // idx is I32; table is F32 of shape [V, d].
        if inputs[0].meta.dtype != Dtype::I32 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Lookup, backend: BACKEND_ID,
                reason: format!("idx must be I32, got {:?}", inputs[0].meta.dtype),
            });
        }
        let n = inputs[0].meta.numel();
        let table_shape = &inputs[1].meta.shape;
        if table_shape.len() != 2 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Lookup, backend: BACKEND_ID,
                reason: format!("table must be 2D, got shape {:?}", table_shape),
            });
        }
        let v = table_shape[0];
        let d = table_shape[1];

        // Read I32 indices.
        let mut idx = Vec::with_capacity(n);
        for i in 0..n {
            let mut b = [0u8; 4];
            b.copy_from_slice(&inputs[0].bytes[i * 4..i * 4 + 4]);
            idx.push(i32::from_le_bytes(b));
        }
        let table = inputs[1].as_f32_vec();

        let mut y = vec![0f32; n * d];
        for i in 0..n {
            let id = idx[i];
            if id < 0 || (id as usize) >= v {
                return Err(ExecutionError::KernelFailed {
                    op: OpKind::Lookup, backend: BACKEND_ID,
                    reason: format!("index {} out of range [0, {})", id, v),
                });
            }
            let src = (id as usize) * d;
            let dst = i * d;
            y[dst..dst + d].copy_from_slice(&table[src..src + d]);
        }
        outputs[0].write_f32(&y);
        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: (n * d) as f64 * 1e-10,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: ((n * 4) + (table.len() * 4)) as u64,
            bytes_written: (y.len() * 4) as u64,
        })
    }
}
