//! Sampling strategies for autoregressive generation.
//!
//! Backends produce a vector of unnormalized logits over the vocabulary
//! at each step. A [`Sampler`] turns that vector into a token id. The
//! standard recipes (greedy, top-k, top-p / nucleus, temperature,
//! mirostat) are provided here as composable building blocks.
//!
//! The samplers are deterministic when fed a deterministic RNG seed —
//! `seed(42)` followed by the same logits will always produce the same
//! token. This matters for EOC: deterministic playback is part of how
//! the cascade memoizes responses.

use std::cmp::Ordering;

use crate::error::{LocalError, LocalResult};

/// A pluggable sampler. Implementors consume a slice of logits over
/// the vocabulary and return a token id.
pub trait Sampler: Send + Sync {
    /// Sample a single token from the logit vector.
    fn sample(&mut self, logits: &[f32]) -> LocalResult<u32>;
}

// ---------------------------------------------------------------------
// Tiny deterministic PRNG.  We avoid pulling in `rand` to keep the
// dependency surface lean and to guarantee bit-for-bit reproducibility
// across platforms.

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
        // Top 24 bits as a float in [0, 1).
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1u32 << 24) as f32)
    }
}

// ---------------------------------------------------------------------
// Samplers.

/// Argmax — pick the highest-logit token. Deterministic, no RNG.
#[derive(Debug, Default, Clone, Copy)]
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &[f32]) -> LocalResult<u32> {
        if logits.is_empty() {
            return Err(LocalError::Sampling("empty logit vector".into()));
        }
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v.is_nan() {
                return Err(LocalError::Sampling("NaN in logit vector".into()));
            }
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        Ok(best_idx as u32)
    }
}

/// Top-k filter: keep only the k highest-logit tokens, set the rest to
/// `-inf`, then softmax-sample over the remainder.
#[derive(Debug, Clone)]
pub struct TopKSampler {
    /// Number of tokens to keep.
    pub k: usize,
    rng: SplitMix64,
}

impl TopKSampler {
    /// Build a top-k sampler with the given k and seed.
    pub fn new(k: usize, seed: u64) -> Self {
        Self {
            k: k.max(1),
            rng: SplitMix64::new(seed),
        }
    }
}

impl Sampler for TopKSampler {
    fn sample(&mut self, logits: &[f32]) -> LocalResult<u32> {
        if logits.is_empty() {
            return Err(LocalError::Sampling("empty logit vector".into()));
        }
        // Sort indices by logit descending, keep top k.
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|a, b| {
            logits[*b]
                .partial_cmp(&logits[*a])
                .unwrap_or(Ordering::Equal)
        });
        let kept = &idx[..self.k.min(idx.len())];
        let kept_logits: Vec<f32> = kept.iter().map(|&i| logits[i]).collect();
        let probs = softmax(&kept_logits);
        let pick = weighted_choice(&probs, &mut self.rng)?;
        Ok(kept[pick] as u32)
    }
}

/// Top-p (nucleus) filter: keep the smallest prefix of tokens (sorted
/// by logit descending) whose cumulative probability mass exceeds `p`,
/// then sample from that prefix.
#[derive(Debug, Clone)]
pub struct TopPSampler {
    /// Probability mass cutoff in `(0, 1]`.
    pub p: f32,
    rng: SplitMix64,
}

impl TopPSampler {
    /// Build a top-p sampler with the given threshold and seed.
    pub fn new(p: f32, seed: u64) -> Self {
        Self {
            p: p.clamp(f32::EPSILON, 1.0),
            rng: SplitMix64::new(seed),
        }
    }
}

impl Sampler for TopPSampler {
    fn sample(&mut self, logits: &[f32]) -> LocalResult<u32> {
        if logits.is_empty() {
            return Err(LocalError::Sampling("empty logit vector".into()));
        }
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|a, b| {
            logits[*b]
                .partial_cmp(&logits[*a])
                .unwrap_or(Ordering::Equal)
        });
        let sorted: Vec<f32> = idx.iter().map(|&i| logits[i]).collect();
        let probs = softmax(&sorted);

        // Walk the cumulative mass until we cross p.
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
        // Renormalize.
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

/// Temperature scaling — divides every logit by `temperature` before
/// softmax. `temperature < 1.0` sharpens the distribution; `> 1.0`
/// flattens it. `temperature == 0.0` collapses to greedy.
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
    fn sample(&mut self, logits: &[f32]) -> LocalResult<u32> {
        if logits.is_empty() {
            return Err(LocalError::Sampling("empty logit vector".into()));
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

/// Mirostat sampler (v1) — adaptive perplexity control. Maintains a
/// running estimate of the surprise tax and adjusts the truncation
/// boundary so that observed surprise matches a target.
#[derive(Debug, Clone)]
pub struct MirostatSampler {
    /// Target cross-entropy (in nats). Typical values: 3.0 - 5.0.
    pub target_tau: f32,
    /// Learning rate.
    pub eta: f32,
    /// Running surprise tax.
    mu: f32,
    rng: SplitMix64,
}

impl MirostatSampler {
    /// Build a Mirostat sampler. `mu` is initialised to `2 * target_tau`
    /// per the original paper.
    pub fn new(target_tau: f32, eta: f32, seed: u64) -> Self {
        Self {
            target_tau,
            eta,
            mu: 2.0 * target_tau,
            rng: SplitMix64::new(seed),
        }
    }
}

impl Sampler for MirostatSampler {
    fn sample(&mut self, logits: &[f32]) -> LocalResult<u32> {
        if logits.is_empty() {
            return Err(LocalError::Sampling("empty logit vector".into()));
        }
        // Sort descending by logit.
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|a, b| {
            logits[*b]
                .partial_cmp(&logits[*a])
                .unwrap_or(Ordering::Equal)
        });
        let sorted: Vec<f32> = idx.iter().map(|&i| logits[i]).collect();
        let probs = softmax(&sorted);

        // Number of candidates we'll keep — derived from mu.
        let k_estimate = (self.mu.exp() as usize).clamp(1, probs.len());
        let kept = &probs[..k_estimate];
        // Renormalize.
        let z: f32 = kept.iter().sum();
        let kept: Vec<f32> = if z > 0.0 {
            kept.iter().map(|p| p / z).collect()
        } else {
            kept.to_vec()
        };
        let pick = weighted_choice(&kept, &mut self.rng)?;
        // Update mu using observed surprise.
        let surprise = -kept[pick].ln();
        self.mu -= self.eta * (surprise - self.target_tau);
        Ok(idx[pick] as u32)
    }
}

/// Compose multiple samplers: each transformation is applied in
/// sequence, finishing with a final `terminal` sampler that picks the
/// token. Transformation samplers are used here as logit filters via
/// the `transform` hook below — the terminal is the only one that
/// actually emits a token.
///
/// In practice, callers compose by stacking transformations and then
/// finishing with [`GreedySampler`] or [`TemperatureSampler`].
pub struct ComposeSampler {
    /// Logit transformations (e.g. repetition penalty hook). Applied in
    /// order before the terminal samples.
    pub transforms: Vec<Box<dyn LogitTransform>>,
    /// Terminal sampler.
    pub terminal: Box<dyn Sampler>,
}

/// Stateless transformation of a logit vector — typically a penalty
/// (repetition / presence / frequency) or a hard mask.
pub trait LogitTransform: Send + Sync {
    /// Modify `logits` in place.
    fn transform(&self, logits: &mut [f32]);
}

impl Sampler for ComposeSampler {
    fn sample(&mut self, logits: &[f32]) -> LocalResult<u32> {
        let mut working = logits.to_vec();
        for t in &self.transforms {
            t.transform(&mut working);
        }
        self.terminal.sample(&working)
    }
}

// ---------------------------------------------------------------------
// Helpers.

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let z: f32 = exps.iter().sum();
    if z > 0.0 {
        exps.iter().map(|x| x / z).collect()
    } else {
        // All -inf — fall back to uniform over the slice.
        vec![1.0 / (logits.len() as f32); logits.len()]
    }
}

fn weighted_choice(probs: &[f32], rng: &mut SplitMix64) -> LocalResult<usize> {
    if probs.is_empty() {
        return Err(LocalError::Sampling("empty probability vector".into()));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let logits = vec![0.1, 0.9, 0.3, 0.2];
        let mut s = GreedySampler;
        assert_eq!(s.sample(&logits).unwrap(), 1);
    }

    #[test]
    fn greedy_rejects_empty() {
        let mut s = GreedySampler;
        assert!(s.sample(&[]).is_err());
    }

    #[test]
    fn greedy_rejects_nan() {
        let mut s = GreedySampler;
        assert!(s.sample(&[1.0, f32::NAN, 2.0]).is_err());
    }

    #[test]
    fn top_k_with_seed_is_reproducible() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut a = TopKSampler::new(3, 42);
        let mut b = TopKSampler::new(3, 42);
        for _ in 0..16 {
            assert_eq!(a.sample(&logits).unwrap(), b.sample(&logits).unwrap());
        }
    }

    #[test]
    fn top_k_keeps_only_top_k() {
        // With k=1 top-k degenerates to greedy.
        let logits = vec![0.0, 0.0, 10.0, 0.0];
        let mut s = TopKSampler::new(1, 42);
        for _ in 0..8 {
            assert_eq!(s.sample(&logits).unwrap(), 2);
        }
    }

    #[test]
    fn top_p_with_seed_is_reproducible() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut a = TopPSampler::new(0.5, 7);
        let mut b = TopPSampler::new(0.5, 7);
        for _ in 0..16 {
            assert_eq!(a.sample(&logits).unwrap(), b.sample(&logits).unwrap());
        }
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits = vec![0.1, 0.9, 0.3];
        let mut s = TemperatureSampler::new(0.0, 0);
        assert_eq!(s.sample(&logits).unwrap(), 1);
    }

    #[test]
    fn mirostat_terminates_and_returns_valid_index() {
        let logits = (0..32).map(|i| i as f32 / 8.0).collect::<Vec<_>>();
        let mut s = MirostatSampler::new(3.0, 0.1, 99);
        for _ in 0..64 {
            let idx = s.sample(&logits).unwrap();
            assert!((idx as usize) < logits.len());
        }
    }

    #[test]
    fn splitmix_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..32 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0, 4.0]);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    struct ZeroOut(usize);
    impl LogitTransform for ZeroOut {
        fn transform(&self, logits: &mut [f32]) {
            if self.0 < logits.len() {
                logits[self.0] = f32::NEG_INFINITY;
            }
        }
    }

    #[test]
    fn compose_applies_transforms_before_terminal() {
        let logits = vec![0.1, 5.0, 0.3];
        let mut s = ComposeSampler {
            transforms: vec![Box::new(ZeroOut(1))],
            terminal: Box::new(GreedySampler),
        };
        // After zeroing out index 1, greedy should pick index 2 (0.3).
        assert_eq!(s.sample(&logits).unwrap(), 2);
    }
}
