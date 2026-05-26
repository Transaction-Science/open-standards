//! Reference Reshape, Transpose, and Concat.
//!
//! Reshape: total-element-preserving shape change. The byte layout is
//! identical between input and output; we copy verbatim. (A future allocator
//! with stride support will make this a metadata-only no-op; for Phase 1.5
//! the physical copy is fine.)
//!
//! Transpose: permute axes. The output's element at logical index
//! `(o_0, o_1, ..., o_{r-1})` reads from the input at
//! `(i_0, ..., i_{r-1})` where `i_{permutation[k]} = o_k`. Implemented via
//! a fixed iteration order for determinism.
//!
//! Concat: concatenate two tensors along a single axis. All other axes
//! must match. Output's shape on the concat axis is `a.shape[axis] +
//! b.shape[axis]`.

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

pub struct ReshapeRef;

impl Kernel for ReshapeRef {
    fn op_kind(&self) -> OpKind { OpKind::Reshape }
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
        let new_shape = match attrs {
            OpAttrs::Reshape { new_shape } => new_shape,
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::Reshape, backend: BACKEND_ID,
                reason: "Reshape kernel requires OpAttrs::Reshape".into(),
            }),
        };
        let in_numel: usize = inputs[0].meta.shape.iter().product();
        let out_numel: usize = new_shape.iter().product();
        if in_numel != out_numel {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Reshape, backend: BACKEND_ID,
                reason: format!("element count mismatch: {} vs {}", in_numel, out_numel),
            });
        }
        // Verbatim byte copy.
        let bytes_in = inputs[0].bytes;
        outputs[0].bytes.copy_from_slice(bytes_in);

        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: (bytes_in.len() as f64) * 1e-12,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: bytes_in.len() as u64,
            bytes_written: bytes_in.len() as u64,
        })
    }
}

pub struct TransposeRef;

impl Kernel for TransposeRef {
    fn op_kind(&self) -> OpKind { OpKind::Transpose }
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
        let permutation = match attrs {
            OpAttrs::Transpose { permutation } => permutation,
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::Transpose, backend: BACKEND_ID,
                reason: "Transpose kernel requires OpAttrs::Transpose".into(),
            }),
        };

        let in_shape = &inputs[0].meta.shape;
        let rank = in_shape.len();
        if permutation.len() != rank {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Transpose, backend: BACKEND_ID,
                reason: format!("permutation rank {} != tensor rank {}",
                    permutation.len(), rank),
            });
        }
        // Validate permutation: each value 0..rank exactly once.
        let mut seen = vec![false; rank];
        for &p in permutation {
            if p >= rank || seen[p] {
                return Err(ExecutionError::KernelFailed {
                    op: OpKind::Transpose, backend: BACKEND_ID,
                    reason: format!("invalid permutation {:?}", permutation),
                });
            }
            seen[p] = true;
        }

        // Element size in bytes.
        let elem_size = inputs[0].meta.dtype.size_bytes();

        // Compute strides for input (row-major).
        let mut in_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        }

        // Output shape: out_shape[i] = in_shape[permutation[i]].
        let out_shape: Vec<usize> = permutation.iter().map(|&p| in_shape[p]).collect();

        // Iterate over output linearly; for each output index, compute the
        // input multi-index by inverting the permutation.
        let total_elems: usize = out_shape.iter().product();
        let in_bytes = inputs[0].bytes;
        let out_bytes = &mut outputs[0].bytes;

        // Precompute output strides for index decoding.
        let mut out_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }

        for out_linear in 0..total_elems {
            // Decode out_linear into multi-index using out_strides.
            let mut rem = out_linear;
            let mut out_multi = vec![0usize; rank];
            for i in 0..rank {
                out_multi[i] = rem / out_strides[i];
                rem %= out_strides[i];
            }
            // Map to input multi-index: in_multi[permutation[i]] = out_multi[i].
            let mut in_multi = vec![0usize; rank];
            for i in 0..rank {
                in_multi[permutation[i]] = out_multi[i];
            }
            // Encode input multi-index using in_strides.
            let mut in_linear = 0usize;
            for i in 0..rank {
                in_linear += in_multi[i] * in_strides[i];
            }
            // Copy element.
            let in_byte = in_linear * elem_size;
            let out_byte = out_linear * elem_size;
            out_bytes[out_byte..out_byte + elem_size]
                .copy_from_slice(&in_bytes[in_byte..in_byte + elem_size]);
        }

        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: (total_elems as f64) * 2e-11,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: in_bytes.len() as u64,
            bytes_written: out_bytes.len() as u64,
        })
    }
}

pub struct ConcatRef;

impl Kernel for ConcatRef {
    fn op_kind(&self) -> OpKind { OpKind::Concat }
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
        let axis_raw = match attrs {
            OpAttrs::Concat { axis } => *axis,
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::Concat, backend: BACKEND_ID,
                reason: "Concat kernel requires OpAttrs::Concat".into(),
            }),
        };
        if inputs.len() != 2 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Concat, backend: BACKEND_ID,
                reason: format!("Concat expects 2 inputs, got {}", inputs.len()),
            });
        }
        let a_shape = &inputs[0].meta.shape;
        let b_shape = &inputs[1].meta.shape;
        if a_shape.len() != b_shape.len() {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Concat, backend: BACKEND_ID,
                reason: format!("rank mismatch: {} vs {}", a_shape.len(), b_shape.len()),
            });
        }
        let rank = a_shape.len();
        let axis = if axis_raw < 0 {
            (rank as i32 + axis_raw) as usize
        } else {
            axis_raw as usize
        };
        if axis >= rank {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Concat, backend: BACKEND_ID,
                reason: format!("axis {} out of range for rank {}", axis_raw, rank),
            });
        }
        for d in 0..rank {
            if d != axis && a_shape[d] != b_shape[d] {
                return Err(ExecutionError::KernelFailed {
                    op: OpKind::Concat, backend: BACKEND_ID,
                    reason: format!("shape mismatch on non-concat axis {}: {:?} vs {:?}",
                        d, a_shape, b_shape),
                });
            }
        }

        let elem_size = inputs[0].meta.dtype.size_bytes();

        // Strategy: split the rank into "outer" (dims before axis) and
        // "inner-block" (axis dim × dims after axis). For each outer
        // index, copy a's inner-block then b's inner-block contiguously.
        let outer: usize = a_shape.iter().take(axis).product::<usize>().max(1);
        let inner_after: usize = a_shape.iter().skip(axis + 1).product::<usize>().max(1);
        let a_axis = a_shape[axis];
        let b_axis = b_shape[axis];

        let a_block_bytes = a_axis * inner_after * elem_size;
        let b_block_bytes = b_axis * inner_after * elem_size;
        let out_block_bytes = a_block_bytes + b_block_bytes;

        let a_bytes = inputs[0].bytes;
        let b_bytes = inputs[1].bytes;
        let out_bytes = &mut outputs[0].bytes;

        for o in 0..outer {
            let a_off = o * a_block_bytes;
            let b_off = o * b_block_bytes;
            let dst_off = o * out_block_bytes;
            out_bytes[dst_off..dst_off + a_block_bytes]
                .copy_from_slice(&a_bytes[a_off..a_off + a_block_bytes]);
            out_bytes[dst_off + a_block_bytes..dst_off + out_block_bytes]
                .copy_from_slice(&b_bytes[b_off..b_off + b_block_bytes]);
        }

        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: ((a_bytes.len() + b_bytes.len()) as f64) * 1e-12,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: (a_bytes.len() + b_bytes.len()) as u64,
            bytes_written: out_bytes.len() as u64,
        })
    }
}

pub struct RepeatRef;

impl Kernel for RepeatRef {
    fn op_kind(&self) -> OpKind { OpKind::Repeat }
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
        let (axis_raw, repeats) = match attrs {
            OpAttrs::Repeat { axis, repeats } => (*axis, *repeats),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::Repeat, backend: BACKEND_ID,
                reason: "Repeat kernel requires OpAttrs::Repeat".into(),
            }),
        };
        let rank = inputs[0].meta.shape.len();
        let axis = if axis_raw < 0 {
            (rank as i32 + axis_raw) as usize
        } else {
            axis_raw as usize
        };
        if axis >= rank {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Repeat, backend: BACKEND_ID,
                reason: format!("axis {} out of range for rank {}", axis_raw, rank),
            });
        }
        if repeats == 0 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Repeat, backend: BACKEND_ID,
                reason: "repeats must be >= 1".into(),
            });
        }

        let in_shape = &inputs[0].meta.shape;
        let elem_size = inputs[0].meta.dtype.size_bytes();
        let outer: usize = in_shape.iter().take(axis).product::<usize>().max(1);
        let axis_size = in_shape[axis];
        let inner: usize = in_shape.iter().skip(axis + 1).product::<usize>().max(1);

        let in_bytes = inputs[0].bytes;
        let out_bytes = &mut outputs[0].bytes;

        let row_bytes = axis_size * inner * elem_size;
        for o in 0..outer {
            let src_off = o * row_bytes;
            for r in 0..repeats {
                let dst_off = o * row_bytes * repeats + r * row_bytes;
                out_bytes[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&in_bytes[src_off..src_off + row_bytes]);
            }
        }

        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: (in_bytes.len() as f64) * (repeats as f64) * 1e-12,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: in_bytes.len() as u64,
            bytes_written: out_bytes.len() as u64,
        })
    }
}

pub struct ScatterRef;

impl Kernel for ScatterRef {
    fn op_kind(&self) -> OpKind { OpKind::Scatter }
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
        let (axis_raw, mut offset) = match attrs {
            OpAttrs::Scatter { axis, offset } => (*axis, *offset),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::Scatter, backend: BACKEND_ID,
                reason: "Scatter kernel requires OpAttrs::Scatter".into(),
            }),
        };
        // 2 inputs = static (dst, src). 3 inputs = (dst, src, pos):
        // the I32 [1] `pos` overrides `offset` at execute time, so a
        // single compiled decode graph scatters this step's K/V at the
        // right buffer slot without a rebuild.
        if inputs.len() == 3 {
            let b = inputs[2].bytes;
            if b.len() >= 4 {
                offset = i32::from_le_bytes([b[0], b[1], b[2], b[3]]).max(0) as usize;
            }
        } else if inputs.len() != 2 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Scatter, backend: BACKEND_ID,
                reason: format!("Scatter expects 2 or 3 inputs, got {}", inputs.len()),
            });
        }
        let dst_shape = &inputs[0].meta.shape;
        let src_shape = &inputs[1].meta.shape;
        if dst_shape.len() != src_shape.len() {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Scatter, backend: BACKEND_ID,
                reason: format!("rank mismatch: dst {} vs src {}",
                    dst_shape.len(), src_shape.len()),
            });
        }
        let rank = dst_shape.len();
        let axis = if axis_raw < 0 {
            (rank as i32 + axis_raw) as usize
        } else {
            axis_raw as usize
        };
        if axis >= rank {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Scatter, backend: BACKEND_ID,
                reason: format!("axis {} out of range for rank {}", axis_raw, rank),
            });
        }
        for d in 0..rank {
            if d != axis && dst_shape[d] != src_shape[d] {
                return Err(ExecutionError::KernelFailed {
                    op: OpKind::Scatter, backend: BACKEND_ID,
                    reason: format!("shape mismatch on non-scatter axis {}: {:?} vs {:?}",
                        d, dst_shape, src_shape),
                });
            }
        }
        if offset + src_shape[axis] > dst_shape[axis] {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Scatter, backend: BACKEND_ID,
                reason: format!(
                    "scatter out of bounds: offset {} + src_len {} > dst_len {} on axis {}",
                    offset, src_shape[axis], dst_shape[axis], axis),
            });
        }

        let elem_size = inputs[0].meta.dtype.size_bytes();
        let outer: usize = dst_shape.iter().take(axis).product::<usize>().max(1);
        let dst_axis = dst_shape[axis];
        let src_axis = src_shape[axis];
        let inner_after: usize = dst_shape.iter().skip(axis + 1).product::<usize>().max(1);

        let dst_bytes = inputs[0].bytes;
        let src_bytes = inputs[1].bytes;
        let out_bytes = &mut outputs[0].bytes;

        let dst_row_bytes = dst_axis * inner_after * elem_size;
        let src_row_bytes = src_axis * inner_after * elem_size;
        let scatter_byte_offset = offset * inner_after * elem_size;
        let scatter_byte_len = src_row_bytes;

        for o in 0..outer {
            let dst_off = o * dst_row_bytes;
            let src_off = o * src_row_bytes;
            let out_off = o * dst_row_bytes;

            // Copy dst slab into output.
            out_bytes[out_off..out_off + dst_row_bytes]
                .copy_from_slice(&dst_bytes[dst_off..dst_off + dst_row_bytes]);
            // Overlay src at the scatter offset.
            out_bytes[out_off + scatter_byte_offset
                ..out_off + scatter_byte_offset + scatter_byte_len]
                .copy_from_slice(&src_bytes[src_off..src_off + src_row_bytes]);
        }

        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: (dst_bytes.len() as f64) * 1e-12,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: (dst_bytes.len() + src_bytes.len()) as u64,
            bytes_written: out_bytes.len() as u64,
        })
    }
}

pub struct SliceRef;

impl Kernel for SliceRef {
    fn op_kind(&self) -> OpKind { OpKind::Slice }
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
        let (axis_raw, slice_start, length) = match attrs {
            OpAttrs::Slice { axis, start, length } => (*axis, *start, *length),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::Slice, backend: BACKEND_ID,
                reason: "Slice kernel requires OpAttrs::Slice".into(),
            }),
        };
        if inputs.len() != 1 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Slice, backend: BACKEND_ID,
                reason: format!("Slice expects 1 input, got {}", inputs.len()),
            });
        }
        let in_shape = &inputs[0].meta.shape;
        let rank = in_shape.len();
        let axis = if axis_raw < 0 {
            (rank as i32 + axis_raw) as usize
        } else {
            axis_raw as usize
        };
        if axis >= rank {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Slice, backend: BACKEND_ID,
                reason: format!("axis {} out of range for rank {}", axis_raw, rank),
            });
        }
        if slice_start + length > in_shape[axis] {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Slice, backend: BACKEND_ID,
                reason: format!(
                    "slice out of bounds: start {} + length {} > dim {} on axis {}",
                    slice_start, length, in_shape[axis], axis),
            });
        }

        let elem_size = inputs[0].meta.dtype.size_bytes();
        let outer: usize = in_shape.iter().take(axis).product::<usize>().max(1);
        let in_axis = in_shape[axis];
        let inner: usize = in_shape.iter().skip(axis + 1).product::<usize>().max(1);

        let in_bytes = inputs[0].bytes;
        let out_bytes = &mut outputs[0].bytes;

        let in_row_bytes = in_axis * inner * elem_size;
        let out_row_bytes = length * inner * elem_size;
        let read_off = slice_start * inner * elem_size;

        for o in 0..outer {
            let in_off = o * in_row_bytes + read_off;
            let out_off = o * out_row_bytes;
            out_bytes[out_off..out_off + out_row_bytes]
                .copy_from_slice(&in_bytes[in_off..in_off + out_row_bytes]);
        }

        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: (out_bytes.len() as f64) * 1e-12,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: in_bytes.len() as u64,
            bytes_written: out_bytes.len() as u64,
        })
    }
}
