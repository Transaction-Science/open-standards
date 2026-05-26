//! Reference categorical sampling.
//!
//! `Greedy`: argmax with tie-break on lowest index (deterministic).
//! `TopK { k }`: deterministic SeededStochastic; requires a seed.
//! Phase 1.1 implements Greedy fully; TopK needs a seed and is deterministic
//! given that seed.

use crate::BACKEND_ID;
use jouleclaw_core::backend::BackendId;
use jouleclaw_core::determinism::DeterminismClass;
use jouleclaw_core::energy::{EnergySourceId, JouleMeasurement};
use jouleclaw_core::error::ExecutionError;
use jouleclaw_core::kernel::{ExecutionContext, Kernel, KernelResult};
use jouleclaw_core::op::{OpAttrs, OpKind, SamplerKind};
use jouleclaw_core::tensor::{TensorView, TensorViewMut};
use std::time::Instant;

pub struct SampleRef;

impl Kernel for SampleRef {
    fn op_kind(&self) -> OpKind { OpKind::Sample }
    fn backend(&self) -> BackendId { BACKEND_ID }
    /// SeededStochastic: deterministic given a seed.
    fn determinism(&self) -> DeterminismClass { DeterminismClass::SeededStochastic }

    fn execute(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        attrs: &OpAttrs,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorViewMut<'_>],
    ) -> Result<KernelResult, ExecutionError> {
        let start = Instant::now();
        let (kind, seed) = match attrs {
            OpAttrs::Sample { kind, seed } => (*kind, *seed),
            _ => return Err(ExecutionError::KernelFailed {
                op: OpKind::Sample, backend: BACKEND_ID,
                reason: "Sample kernel requires OpAttrs::Sample".into(),
            }),
        };

        let logits = inputs[0].as_f32_vec();
        let id: i32 = match kind {
            SamplerKind::Greedy => argmax_with_low_index_tiebreak(&logits),
            SamplerKind::TopK { k } => {
                let s = seed.ok_or_else(|| ExecutionError::KernelFailed {
                    op: OpKind::Sample, backend: BACKEND_ID,
                    reason: "TopK sampler requires a seed in deterministic mode".into(),
                })?;
                top_k_seeded(&logits, k, s)
            }
            SamplerKind::TopP { .. } | SamplerKind::Temperature { .. } => {
                return Err(ExecutionError::KernelFailed {
                    op: OpKind::Sample, backend: BACKEND_ID,
                    reason: "TopP / Temperature samplers not implemented in Phase 1.1".into(),
                });
            }
        };

        let bytes = id.to_le_bytes();
        outputs[0].bytes[0..4].copy_from_slice(&bytes);
        let elapsed = start.elapsed();
        Ok(KernelResult {
            joules: JouleMeasurement {
                joules: (logits.len() as f64) * 1e-10,
                energy_source: EnergySourceId(0),
                measurement_window: elapsed,
                attribution_confidence: 0.0,
            },
            wall_clock: elapsed,
            bytes_read: (logits.len() * 4) as u64,
            bytes_written: 4,
        })
    }
}

fn argmax_with_low_index_tiebreak(logits: &[f32]) -> i32 {
    let mut best_i = 0usize;
    let mut best_v = logits[0];
    for i in 1..logits.len() {
        if logits[i] > best_v {
            best_v = logits[i];
            best_i = i;
        }
        // Ties keep the lower index.
    }
    best_i as i32
}

/// Linear-congruential generator with fixed constants. Deterministic given seed.
struct Lcg64 { state: u64 }
impl Lcg64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        // Take top 24 bits for [0, 1).
        let bits = (self.next() >> 40) as u32;
        (bits as f32) * (1.0 / (1u32 << 24) as f32)
    }
}

fn top_k_seeded(logits: &[f32], k: u32, seed: u64) -> i32 {
    let k = (k as usize).min(logits.len()).max(1);
    // Sort indices by logit, descending, with ties broken by lower index.
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        match logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal) {
            std::cmp::Ordering::Equal => a.cmp(&b),
            other => other,
        }
    });
    let top = &idx[..k];

    // Softmax over top-k logits.
    let max = top.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut probs = Vec::with_capacity(k);
    let mut sum = 0f32;
    for &i in top {
        let p = (logits[i] - max).exp();
        probs.push(p);
        sum += p;
    }
    for p in &mut probs { *p /= sum; }

    // Sample with the LCG.
    let mut rng = Lcg64::new(seed);
    let r = rng.next_f32();
    let mut acc = 0f32;
    for j in 0..k {
        acc += probs[j];
        if r < acc { return top[j] as i32; }
    }
    top[k - 1] as i32
}
