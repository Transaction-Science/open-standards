//! Sampling strategies for the speculative-decoding orchestrator.
//!
//! These are the *terminal* samplers used by the orchestrator when
//! turning logits into token ids — both inside the SpS acceptance
//! check (which needs probability-space distributions, not raw
//! argmaxes) and when emitting the target's replacement / bonus token.
//!
//! When the `local-backends` feature is enabled the samplers from
//! `eoc-local::sampling` are re-exported under the same names so the
//! whole codebase shares one implementation. Without the feature, this
//! module ships a small local copy of the four canonical samplers
//! (greedy, temperature, top-k, top-p) so `eoc-spec-decode` is usable
//! without pulling in any inference backend.

#[cfg(feature = "local-backends")]
mod re_export {
    pub use eoc_local::sampling::{
        GreedySampler as LocalGreedySampler, Sampler as LocalSamplerTrait,
        TemperatureSampler as LocalTemperatureSampler, TopKSampler as LocalTopKSampler,
        TopPSampler as LocalTopPSampler,
    };
}

use crate::error::{SpecDecodeError, SpecDecodeResult};

/// Sample a token id from a logit vector.
pub trait Sampler: Send + Sync {
    /// Pick the next token. Implementations are allowed to be stateful
    /// (e.g. they carry a PRNG), which is why this takes `&mut self`.
    fn sample(&mut self, logits: &[f32]) -> SpecDecodeResult<u32>;
}

/// Convert a slice of logits to a probability distribution via softmax.
/// Public because the SpS acceptance check needs it.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return vec![1.0 / (logits.len().max(1) as f32); logits.len()];
    }
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let z: f32 = exps.iter().sum();
    if z > 0.0 {
        exps.iter().map(|x| x / z).collect()
    } else {
        vec![1.0 / (logits.len().max(1) as f32); logits.len()]
    }
}

// ---------------------------------------------------------------------
// Tiny deterministic PRNG — same algorithm as eoc-local's so seeded
// playback matches across the two crates. We keep an independent copy
// so the `default` build of `eoc-spec-decode` doesn't depend on
// `eoc-local`.

/// 64-bit splitmix-style PRNG. Seedable, deterministic, no allocations.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the PRNG. Seeding with 0 is permitted.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Draw the next u64.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Draw the next f32 in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1u32 << 24) as f32)
    }
}

fn weighted_choice(probs: &[f32], rng: &mut SplitMix64) -> SpecDecodeResult<usize> {
    if probs.is_empty() {
        return Err(SpecDecodeError::Sampling(
            "empty probability vector".into(),
        ));
    }
    let r = rng.next_f32();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return Ok(i);
        }
    }
    Ok(probs.len() - 1)
}

// ---------------------------------------------------------------------
// Samplers.

/// Argmax — pick the highest-logit token. Deterministic, no RNG.
#[derive(Debug, Default, Clone, Copy)]
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &[f32]) -> SpecDecodeResult<u32> {
        if logits.is_empty() {
            return Err(SpecDecodeError::Sampling("empty logit vector".into()));
        }
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v.is_nan() {
                return Err(SpecDecodeError::Sampling("NaN in logit vector".into()));
            }
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        Ok(best_idx as u32)
    }
}

/// Temperature scaling — divides every logit by `temperature` before
/// softmax. `temperature == 0.0` collapses to greedy.
#[derive(Debug, Clone)]
pub struct TemperatureSampler {
    /// Sampling temperature.
    pub temperature: f32,
    rng: SplitMix64,
}

impl TemperatureSampler {
    /// Build a temperature sampler.
    pub fn new(temperature: f32, seed: u64) -> Self {
        Self {
            temperature,
            rng: SplitMix64::new(seed),
        }
    }
}

impl Sampler for TemperatureSampler {
    fn sample(&mut self, logits: &[f32]) -> SpecDecodeResult<u32> {
        if logits.is_empty() {
            return Err(SpecDecodeError::Sampling("empty logit vector".into()));
        }
        if self.temperature == 0.0 {
            return GreedySampler.sample(logits);
        }
        let scaled: Vec<f32> = logits.iter().map(|x| x / self.temperature).collect();
        let probs = softmax(&scaled);
        let pick = weighted_choice(&probs, &mut self.rng)?;
        Ok(pick as u32)
    }
}

/// Top-k filter: keep only the `k` highest-logit tokens, softmax over
/// the survivors, sample.
#[derive(Debug, Clone)]
pub struct TopKSampler {
    /// Number of tokens to keep.
    pub k: usize,
    rng: SplitMix64,
}

impl TopKSampler {
    /// Build a top-k sampler.
    pub fn new(k: usize, seed: u64) -> Self {
        Self {
            k: k.max(1),
            rng: SplitMix64::new(seed),
        }
    }
}

impl Sampler for TopKSampler {
    fn sample(&mut self, logits: &[f32]) -> SpecDecodeResult<u32> {
        if logits.is_empty() {
            return Err(SpecDecodeError::Sampling("empty logit vector".into()));
        }
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|a, b| {
            logits[*b]
                .partial_cmp(&logits[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let kept = &idx[..self.k.min(idx.len())];
        let kept_logits: Vec<f32> = kept.iter().map(|&i| logits[i]).collect();
        let probs = softmax(&kept_logits);
        let pick = weighted_choice(&probs, &mut self.rng)?;
        Ok(kept[pick] as u32)
    }
}

/// Top-p (nucleus) filter: keep the smallest set whose cumulative
/// probability mass exceeds `p`, sample from it.
#[derive(Debug, Clone)]
pub struct TopPSampler {
    /// Probability mass cutoff in `(0, 1]`.
    pub p: f32,
    rng: SplitMix64,
}

impl TopPSampler {
    /// Build a top-p sampler.
    pub fn new(p: f32, seed: u64) -> Self {
        Self {
            p: p.clamp(f32::EPSILON, 1.0),
            rng: SplitMix64::new(seed),
        }
    }
}

impl Sampler for TopPSampler {
    fn sample(&mut self, logits: &[f32]) -> SpecDecodeResult<u32> {
        if logits.is_empty() {
            return Err(SpecDecodeError::Sampling("empty logit vector".into()));
        }
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|a, b| {
            logits[*b]
                .partial_cmp(&logits[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sorted: Vec<f32> = idx.iter().map(|&i| logits[i]).collect();
        let probs = softmax(&sorted);

        let mut cum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &q) in probs.iter().enumerate() {
            cum += q;
            if cum >= self.p {
                cutoff = i + 1;
                break;
            }
        }
        let kept_probs = &probs[..cutoff];
        let z: f32 = kept_probs.iter().sum();
        let kept_probs: Vec<f32> = if z > 0.0 {
            kept_probs.iter().map(|p| p / z).collect()
        } else {
            kept_probs.to_vec()
        };
        let pick = weighted_choice(&kept_probs, &mut self.rng)?;
        Ok(idx[pick] as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let logits = vec![0.1, 0.9, 0.3, 0.2];
        let mut s = GreedySampler;
        assert_eq!(s.sample(&logits).expect("non-empty"), 1);
    }

    #[test]
    fn greedy_rejects_empty() {
        let mut s = GreedySampler;
        assert!(s.sample(&[]).is_err());
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits = vec![0.1, 0.9, 0.3];
        let mut s = TemperatureSampler::new(0.0, 0);
        assert_eq!(s.sample(&logits).expect("non-empty"), 1);
    }

    #[test]
    fn top_k_reproducible() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut a = TopKSampler::new(3, 42);
        let mut b = TopKSampler::new(3, 42);
        for _ in 0..16 {
            assert_eq!(
                a.sample(&logits).expect("non-empty"),
                b.sample(&logits).expect("non-empty")
            );
        }
    }

    #[test]
    fn top_p_reproducible() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut a = TopPSampler::new(0.5, 7);
        let mut b = TopPSampler::new(0.5, 7);
        for _ in 0..16 {
            assert_eq!(
                a.sample(&logits).expect("non-empty"),
                b.sample(&logits).expect("non-empty")
            );
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0, 4.0]);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }
}
