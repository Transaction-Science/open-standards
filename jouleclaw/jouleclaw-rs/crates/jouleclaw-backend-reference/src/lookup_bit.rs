//! Reference 1-bit embedding lookup. Indexes a PrismML Q1_0 g128 packed
//! table `[V, d]` with `[n]` I32 ids, decoding only the `n` requested
//! rows to f32 `[n, d]`. Same on-demand semantics as
//! [`crate::lookup_ternary`]; for Bonsai's 151669×2048 table the packed
//! size is ~17 MB vs ~1.2 GB f32.

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind};
use jouleclaw_core::tensor::{Dtype, TensorView, TensorViewMut};
use std::time::Instant;

const QK: usize = 128;
const BLK: usize = 18;

#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let s = (h >> 15) & 1;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;
    let v = if e == 0 {
        (m as f32 / 1024.0) * 2f32.powi(-14)
    } else if e == 31 {
        if m == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15)
    };
    if s == 1 { -v } else { v }
}

#[inline]
fn decode_q1_0_row(row: &[u8], d: usize, dst: &mut [f32]) {
    let n_blocks = (d + QK - 1) / QK;
    for blk in 0..n_blocks {
        let bb = blk * BLK;
        let scale = f16_to_f32(u16::from_le_bytes([row[bb], row[bb + 1]]));
        let codes = &row[bb + 2..bb + BLK];
        let k0 = blk * QK;
        let kn = (k0 + QK).min(d);
        for (local, kk) in (k0..kn).enumerate() {
            let bit = (codes[local / 8] >> (local % 8)) & 1;
            dst[kk] = if bit == 1 { scale } else { -scale };
        }
    }
}

pub struct LookupBitRef;

impl Kernel for LookupBitRef {
    fn op_kind(&self) -> OpKind { OpKind::LookupBit }
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
        let (v, d) = match attrs {
            OpAttrs::LookupBit { v, d } => (*v, *d),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::LookupBit, backend: BACKEND_ID,
                reason: "LookupBit kernel requires OpAttrs::LookupBit".into(),
            }),
        };
        if inputs[0].meta.dtype != Dtype::I32 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::LookupBit, backend: BACKEND_ID,
                reason: format!("idx must be I32, got {:?}", inputs[0].meta.dtype),
            });
        }
        let n = inputs[0].meta.numel();
        let row_bytes = ((d + QK - 1) / QK) * BLK;
        let table = inputs[1].bytes;
        if table.len() < v * row_bytes {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::LookupBit, backend: BACKEND_ID,
                reason: format!("packed table too small: need {} bytes, have {}",
                    v * row_bytes, table.len()),
            });
        }

        let mut y = vec![0f32; n * d];
        for i in 0..n {
            let mut b = [0u8; 4];
            b.copy_from_slice(&inputs[0].bytes[i * 4..i * 4 + 4]);
            let id = i32::from_le_bytes(b);
            if id < 0 || (id as usize) >= v {
                return Err(ExecutionError::KernelFailed {
                    op: OpKind::LookupBit, backend: BACKEND_ID,
                    reason: format!("index {} out of range [0, {})", id, v),
                });
            }
            let src = (id as usize) * row_bytes;
            decode_q1_0_row(&table[src..src + row_bytes], d,
                &mut y[i * d..(i + 1) * d]);
        }

        outputs[0].write_f32(&y);
        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: (n * d) as f64 * 3e-11,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: ((n * 4) + (n * row_bytes)) as u64,
            bytes_written: (y.len() * 4) as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_core::tensor::TensorMeta;

    fn pack_row(bits: &[i8], d_f16: u16) -> Vec<u8> {
        let d = bits.len();
        let nb = (d + QK - 1) / QK;
        let mut out = vec![0u8; nb * BLK];
        for blk in 0..nb {
            let bb = blk * BLK;
            out[bb..bb + 2].copy_from_slice(&d_f16.to_le_bytes());
            for local in 0..QK {
                let idx = blk * QK + local;
                if idx >= d { break; }
                if bits[idx] == 1 {
                    out[bb + 2 + (local >> 3)] |= 1 << (local & 7);
                }
            }
        }
        out
    }

    #[test]
    fn decodes_only_requested_rows() {
        let d = 128;
        let d_f16: u16 = 0x3C00;
        let mut tbl = pack_row(&vec![1i8; d], d_f16);
        tbl.extend(pack_row(&vec![-1i8; d], d_f16));
        tbl.extend(pack_row(&vec![1i8; d], d_f16));

        let idx: Vec<u8> = [1i32, 0]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        let i_meta = TensorMeta::new(Dtype::I32, &[2]);
        let t_meta = TensorMeta::new(Dtype::U8, &[tbl.len()]);
        let o_meta = TensorMeta::new(Dtype::F32, &[2, d]);
        let mut ob = vec![0u8; 2 * d * 4];
        let iv = [
            TensorView { meta: &i_meta, bytes: &idx },
            TensorView { meta: &t_meta, bytes: &tbl },
        ];
        let mut ov = [TensorViewMut { meta: &o_meta, bytes: &mut ob }];
        let mut scratch = vec![0u8; 64];
        let mut ctx = ExecutionContext {
            backend: BACKEND_ID, deterministic: true,
            seed: None, scratch: &mut scratch,
        };
        LookupBitRef.execute(&mut ctx,
            &OpAttrs::LookupBit { v: 3, d }, &iv, &mut ov).unwrap();
        let y: Vec<f32> = (0..2 * d).map(|i| f32::from_le_bytes(
            [ob[i*4], ob[i*4+1], ob[i*4+2], ob[i*4+3]])).collect();
        assert!(y[0..d].iter().all(|&x| x == -1.0), "row1 → all -1");
        assert!(y[d..2*d].iter().all(|&x| x == 1.0), "row0 → all +1");
    }
}
