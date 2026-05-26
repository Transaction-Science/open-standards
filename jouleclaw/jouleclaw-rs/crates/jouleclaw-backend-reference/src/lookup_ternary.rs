//! Reference ternary embedding lookup.
//!
//! Indexes a PrismML Q2_0 g128 packed table `[V, d]` with `[n]` I32
//! ids, decoding **only the `n` requested rows** to f32 `[n, d]`. The
//! full table is never dequantised: a 151669×2048 embedding is ~40 MB
//! packed but ~1.2 GB as f32 — decoding on demand is the difference
//! between "fits on a Pi" and "doesn't". Each row is one Q2_0 stream of
//! `ceil(d/128)` 34-byte blocks (f16 scale + 32 bytes of LSB-first
//! 2-bit codes; value = (q-1)·scale). Deterministic and numerically
//! identical to `lookup` over the dequantised table.

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
const BLK: usize = 34;

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

/// Decode one Q2_0 row (`d` ternary values) into `dst[..d]`.
#[inline]
pub(crate) fn decode_q2_0_row(row: &[u8], d: usize, dst: &mut [f32]) {
    let n_blocks = (d + QK - 1) / QK;
    for blk in 0..n_blocks {
        let bb = blk * BLK;
        let scale = f16_to_f32(u16::from_le_bytes([row[bb], row[bb + 1]]));
        let codes = &row[bb + 2..bb + BLK];
        let k0 = blk * QK;
        let kn = (k0 + QK).min(d);
        for (local, kk) in (k0..kn).enumerate() {
            let q = (codes[local >> 2] >> ((local & 3) * 2)) & 0b11;
            dst[kk] = (q as i32 - 1) as f32 * scale;
        }
    }
}

pub struct LookupTernaryRef;

impl Kernel for LookupTernaryRef {
    fn op_kind(&self) -> OpKind { OpKind::LookupTernary }
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
            OpAttrs::LookupTernary { v, d } => (*v, *d),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::LookupTernary, backend: BACKEND_ID,
                reason: "LookupTernary kernel requires OpAttrs::LookupTernary".into(),
            }),
        };
        if inputs[0].meta.dtype != Dtype::I32 {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::LookupTernary, backend: BACKEND_ID,
                reason: format!("idx must be I32, got {:?}", inputs[0].meta.dtype),
            });
        }
        let n = inputs[0].meta.numel();
        let row_bytes = ((d + QK - 1) / QK) * BLK;
        let table = inputs[1].bytes;
        if table.len() < v * row_bytes {
            return Err(ExecutionError::KernelFailed {
                op: OpKind::LookupTernary, backend: BACKEND_ID,
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
                    op: OpKind::LookupTernary, backend: BACKEND_ID,
                    reason: format!("index {} out of range [0, {})", id, v),
                });
            }
            let src = (id as usize) * row_bytes;
            decode_q2_0_row(&table[src..src + row_bytes], d,
                &mut y[i * d..(i + 1) * d]);
        }

        outputs[0].write_f32(&y);
        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                // Only `n` rows touched, not `v` — that's the whole point.
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

    fn pack_row(vals: &[i8], d_f16: u16) -> Vec<u8> {
        let d = vals.len();
        let nb = (d + QK - 1) / QK;
        let mut out = vec![0u8; nb * BLK];
        for blk in 0..nb {
            let bb = blk * BLK;
            out[bb..bb + 2].copy_from_slice(&d_f16.to_le_bytes());
            for local in 0..QK {
                let idx = blk * QK + local;
                if idx >= d { break; }
                let q: u8 = match vals[idx] { -1 => 0, 0 => 1, 1 => 2, _ => 3 };
                out[bb + 2 + (local >> 2)] |= q << ((local & 3) * 2);
            }
        }
        out
    }

    #[test]
    fn decodes_only_requested_rows() {
        let d = 128;
        let d_f16: u16 = 0x3C00; // 1.0
        // 3-row table: row0 all +1, row1 all -1, row2 all 0.
        let mut tbl = pack_row(&vec![1i8; d], d_f16);
        tbl.extend(pack_row(&vec![-1i8; d], d_f16));
        tbl.extend(pack_row(&vec![0i8; d], d_f16));

        // Ask for rows [2, 0] only.
        let idx: Vec<u8> = [2i32, 0]
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
        LookupTernaryRef.execute(
            &mut ctx, &OpAttrs::LookupTernary { v: 3, d },
            &iv, &mut ov,
        ).unwrap();
        let y: Vec<f32> = (0..2 * d).map(|i| f32::from_le_bytes(
            [ob[i*4], ob[i*4+1], ob[i*4+2], ob[i*4+3]])).collect();
        assert!(y[0..d].iter().all(|&x| x == 0.0), "row2 → all 0");
        assert!(y[d..2*d].iter().all(|&x| x == 1.0), "row0 → all +1");
    }
}
