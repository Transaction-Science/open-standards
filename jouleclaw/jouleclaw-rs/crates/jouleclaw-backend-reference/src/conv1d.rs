//! Reference depthwise causal 1-D convolution.
//!
//! Used by LFM2's `shortconv` recurrent block. Each channel is
//! convolved independently with its own `taps`-long kernel; positions
//! before the start of the sequence are treated as zero, so `y[t, c]`
//! only depends on `x[≤t, c]` (causal). For LFM2-350M `taps == 3`.
//!
//! Math:
//!   y[t, c] = Σ_{k=0..taps-1} w[c, k] · x[t - (taps - 1 - k), c]
//! with x[<0, c] := 0.
//!
//! The weight's logical shape is `[d_model, taps]` — that's the
//! natural layout after the GGUF loader reverses the file's ne-order
//! `[taps, d_model]` storage (channels-outer, taps-inner per byte).

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

pub struct Conv1DDepthwiseCausalRef;

impl Kernel for Conv1DDepthwiseCausalRef {
    fn op_kind(&self) -> OpKind { OpKind::Conv1DDepthwiseCausal }
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
        let taps = match attrs {
            OpAttrs::Conv1DDepthwise { taps } => *taps,
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::Conv1DDepthwiseCausal, backend: BACKEND_ID,
                reason: "expected OpAttrs::Conv1DDepthwise".into(),
            }),
        };
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Conv1DDepthwiseCausal, backend: BACKEND_ID,
                reason: format!("expects 2 inputs / 1 output, got {} / {}",
                    inputs.len(), outputs.len()),
            });
        }
        let x = inputs[0].as_f32_vec();
        let w = inputs[1].as_f32_vec();
        let x_shape = &inputs[0].meta.shape;
        let w_shape = &inputs[1].meta.shape;
        if x_shape.len() != 2 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Conv1DDepthwiseCausal, backend: BACKEND_ID,
                reason: format!("x must be 2-D [seq, d_model], got {:?}", x_shape),
            });
        }
        let seq = x_shape[0];
        let d = x_shape[1];
        if w_shape.len() != 2 || w_shape[0] != d || w_shape[1] != taps {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::Conv1DDepthwiseCausal, backend: BACKEND_ID,
                reason: format!(
                    "weight shape must be [d_model={}, taps={}], got {:?}",
                    d, taps, w_shape),
            });
        }

        let mut y = vec![0f32; seq * d];
        for t in 0..seq {
            for c in 0..d {
                let mut acc = 0f32;
                for k in 0..taps {
                    // k=0 → x[t - (taps-1)]   (oldest)
                    // k=taps-1 → x[t]         (current)
                    let lag = taps - 1 - k;
                    if t >= lag {
                        acc += w[c * taps + k] * x[(t - lag) * d + c];
                    }
                    // else: implicit zero left-pad
                }
                y[t * d + c] = acc;
            }
        }
        outputs[0].write_f32(&y);

        let elapsed = start.elapsed();
        let muls_adds = (seq * d * taps * 2) as f64;
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: muls_adds * 1e-10,
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

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_core::tensor::{Dtype, TensorMeta};

    fn run(x: &[f32], seq: usize, d: usize, w: &[f32], taps: usize) -> Vec<f32> {
        let xm = TensorMeta::new(Dtype::F32, &[seq, d]);
        let wm = TensorMeta::new(Dtype::F32, &[d, taps]);
        let om = TensorMeta::new(Dtype::F32, &[seq, d]);
        let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let wb: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut ob = vec![0u8; seq * d * 4];
        let iv = [
            TensorView { meta: &xm, bytes: &xb },
            TensorView { meta: &wm, bytes: &wb },
        ];
        let mut ov = [TensorViewMut { meta: &om, bytes: &mut ob }];
        let mut scratch = vec![0u8; 64];
        let mut ctx = ExecutionContext {
            backend: BACKEND_ID, deterministic: true, seed: None,
            scratch: &mut scratch,
        };
        Conv1DDepthwiseCausalRef.execute(
            &mut ctx, &OpAttrs::Conv1DDepthwise { taps }, &iv, &mut ov,
        ).unwrap();
        (0..seq * d).map(|i| f32::from_le_bytes(
            [ob[i*4], ob[i*4+1], ob[i*4+2], ob[i*4+3]])).collect()
    }

    #[test]
    fn conv1d_taps3_single_channel_causal_padding() {
        // taps=3, d=1, seq=5. w = [w0, w1, w2]. x = [1, 2, 3, 4, 5].
        // y[0] = w2*1                       (others zero-padded)
        // y[1] = w1*1 + w2*2
        // y[2] = w0*1 + w1*2 + w2*3
        // y[3] = w0*2 + w1*3 + w2*4
        // y[4] = w0*3 + w1*4 + w2*5
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = vec![0.5, 0.25, 1.0]; // w0,w1,w2
        let y = run(&x, 5, 1, &w, 3);
        assert_eq!(y[0], 1.0 * 1.0);
        assert_eq!(y[1], 0.25 * 1.0 + 1.0 * 2.0);
        assert_eq!(y[2], 0.5 * 1.0 + 0.25 * 2.0 + 1.0 * 3.0);
        assert_eq!(y[3], 0.5 * 2.0 + 0.25 * 3.0 + 1.0 * 4.0);
        assert_eq!(y[4], 0.5 * 3.0 + 0.25 * 4.0 + 1.0 * 5.0);
    }

    #[test]
    fn conv1d_independent_per_channel() {
        // 2 channels, distinct kernels. d=2, seq=3, taps=2.
        // Weight layout is [d_model=2, taps=2] (channels outer, taps inner),
        // so flat = [c0_k0, c0_k1, c1_k0, c1_k1] = [0.5, 1.0, 0.1, 0.2].
        // x = [[1,10],[2,20],[3,30]] (row-major [seq, d]).
        // Per channel kernel [w(k=0), w(k=1)] with lag mapping
        // k=0 → x[t-1], k=1 → x[t]:
        //   c=0 kernel [0.5, 1.0]: y[0]=1.0*1=1, y[1]=0.5*1+1.0*2=2.5, y[2]=0.5*2+1.0*3=4
        //   c=1 kernel [0.1, 0.2]: y[0]=0.2*10=2, y[1]=0.1*10+0.2*20=5, y[2]=0.1*20+0.2*30=8
        let x = vec![1.0,10.0,  2.0,20.0,  3.0,30.0];
        let w = vec![0.5, 1.0,   0.1, 0.2];
        let y = run(&x, 3, 2, &w, 2);
        assert_eq!(y[0], 1.0);
        assert_eq!(y[1], 2.0);
        assert_eq!(y[2], 2.5);
        assert_eq!(y[3], 5.0);
        assert_eq!(y[4], 4.0);
        assert_eq!(y[5], 8.0);
    }
}
