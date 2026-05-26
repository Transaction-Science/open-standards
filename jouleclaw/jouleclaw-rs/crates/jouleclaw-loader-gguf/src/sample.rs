//! Sampling from logits.
//!
//! Wraps the existing `Sample` op (which only handles the deterministic
//! kernel call inside a graph) with a host-side ergonomic API for
//! generation loops.
//!
//! All samplers are deterministic given a seed. The unseeded forms are
//! intentionally absent: callers that want non-determinism should pass
//! `seed = current_time_ns()` or similar themselves, so randomness is
//! always traceable.

/// Sampling configuration.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature: 0 means greedy (argmax). >0 scales logits by `1/temperature`
    /// before any other transformation.
    pub temperature: f32,
    /// Top-K: keep only the top K logits before softmax. 0 = disabled.
    pub top_k: usize,
    /// Top-P (nucleus): keep the smallest set of tokens whose cumulative
    /// probability exceeds this threshold. 1.0 = disabled.
    pub top_p: f32,
    /// Random seed. Required for any non-greedy sampling.
    pub seed: u64,
    /// Repetition penalty (1.0 = disabled). Applied to logits of tokens
    /// in `recent_tokens`: positive logits are divided by the penalty,
    /// negative logits are multiplied. Standard llama.cpp-style behavior.
    /// 1.1–1.3 is a typical range; >1.5 often makes generation incoherent.
    pub repetition_penalty: f32,
    /// Frequency penalty (0.0 = disabled). Subtracts `frequency_penalty *
    /// count(token)` from the logit of each recent token. Stronger penalty
    /// for tokens that appear many times. OpenAI-style.
    pub frequency_penalty: f32,
    /// Presence penalty (0.0 = disabled). Subtracts `presence_penalty` from
    /// the logit of any token that appears at least once in `recent_tokens`,
    /// regardless of count. OpenAI-style.
    pub presence_penalty: f32,
    /// Logit bias: adds the given value to the logit of each token ID
    /// before sampling. Used for grammar-constrained sampling, banned
    /// words (negative infinity), forced tokens (large positive bias).
    pub logit_bias: Vec<(u32, f32)>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,  // greedy by default
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: Vec::new(),
        }
    }
}

impl SamplingConfig {
    pub fn greedy() -> Self { Self::default() }

    pub fn temperature(t: f32, seed: u64) -> Self {
        Self { temperature: t, seed, ..Self::default() }
    }

    pub fn top_k(k: usize, temperature: f32, seed: u64) -> Self {
        Self { temperature, top_k: k, seed, ..Self::default() }
    }

    pub fn top_p(p: f32, temperature: f32, seed: u64) -> Self {
        Self { temperature, top_p: p, seed, ..Self::default() }
    }

    /// Builder helper: set the repetition penalty (1.0 = no penalty).
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Builder helper: set the frequency penalty (0.0 = no penalty).
    pub fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = penalty;
        self
    }

    /// Builder helper: set the presence penalty (0.0 = no penalty).
    pub fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = penalty;
        self
    }

    /// Builder helper: add a logit bias entry.
    pub fn with_logit_bias(mut self, token_id: u32, bias: f32) -> Self {
        self.logit_bias.push((token_id, bias));
        self
    }

    /// Builder helper: ban a list of token IDs (sets logit to -inf).
    pub fn banning(mut self, token_ids: &[u32]) -> Self {
        for &id in token_ids {
            self.logit_bias.push((id, f32::NEG_INFINITY));
        }
        self
    }
}

/// Sample one token ID from a row of logits.
///
/// Equivalent to `sample_logits_with_history(logits, cfg, &[])`. The
/// penalties (`repetition_penalty`, `frequency_penalty`, `presence_penalty`)
/// have no effect when called this way; only `logit_bias`, `temperature`,
/// `top_k`, and `top_p` apply.
pub fn sample_logits(logits: &[f32], cfg: &SamplingConfig) -> u32 {
    sample_logits_with_history(logits, cfg, &[])
}

/// Sample one token ID from a row of logits, with optional history for
/// applying repetition / frequency / presence penalties.
///
/// `recent_tokens` is the slice of recent token IDs whose presence should
/// be penalized. Typically the last N generated tokens or the full
/// generated sequence.
///
/// Pipeline:
///   1. Apply repetition / frequency / presence penalties from `recent_tokens`.
///   2. Apply `logit_bias` (added to each named token's logit).
///   3. If temperature == 0, return argmax. Otherwise:
///   4. Apply temperature scaling.
///   5. Top-K filter.
///   6. Top-P (nucleus) filter.
///   7. Softmax and sample from the resulting distribution.
pub fn sample_logits_with_history(
    logits: &[f32],
    cfg: &SamplingConfig,
    recent_tokens: &[u32],
) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    // We may need to mutate logits, so clone once up front when any
    // pre-sampling transformation applies.
    let needs_mutation = cfg.repetition_penalty != 1.0
        || cfg.frequency_penalty != 0.0
        || cfg.presence_penalty != 0.0
        || !cfg.logit_bias.is_empty();

    let logits_owned: Vec<f32>;
    let logits_view: &[f32] = if needs_mutation {
        let mut v = logits.to_vec();
        apply_penalties(&mut v, cfg, recent_tokens);
        apply_logit_bias(&mut v, &cfg.logit_bias);
        logits_owned = v;
        &logits_owned
    } else {
        logits
    };

    // Greedy: argmax with low-index tie-break.
    if cfg.temperature == 0.0 {
        return argmax_with_low_index_tiebreak(logits_view) as u32;
    }

    // Temperature scaling.
    let inv_t = 1.0 / cfg.temperature;
    let scaled: Vec<f32> = logits_view.iter().map(|&v| v * inv_t).collect();

    // Sort indices by descending logit (ties broken by lower index).
    let mut idx: Vec<usize> = (0..scaled.len()).collect();
    idx.sort_by(|&a, &b| {
        scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    // Top-K filter: truncate to first K.
    let k = if cfg.top_k > 0 { cfg.top_k.min(idx.len()) } else { idx.len() };
    let kept_idx = &idx[..k];

    // Compute softmax over kept indices.
    let max = kept_idx.iter().map(|&i| scaled[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = kept_idx.iter()
        .map(|&i| (scaled[i] - max).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs { *p /= sum; }

    // Top-P filter: take the smallest prefix whose cumulative prob >= top_p.
    let mut cutoff = probs.len();
    if cfg.top_p < 1.0 {
        let mut cum = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if cum >= cfg.top_p {
                cutoff = i + 1;
                break;
            }
        }
    }
    let final_probs = &probs[..cutoff];
    let final_idx = &kept_idx[..cutoff];

    // Re-normalize after filters.
    let s: f32 = final_probs.iter().sum();
    let normed: Vec<f32> = final_probs.iter().map(|p| p / s).collect();

    // Sample with deterministic LCG seeded from cfg.seed.
    let mut rng = Lcg64::new(cfg.seed);
    let r = rng.next_f32();
    let mut acc = 0f32;
    for (i, &p) in normed.iter().enumerate() {
        acc += p;
        if r < acc { return final_idx[i] as u32; }
    }
    final_idx[final_idx.len() - 1] as u32
}

fn argmax_with_low_index_tiebreak(logits: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = logits[0];
    for i in 1..logits.len() {
        if logits[i] > best_v {
            best_v = logits[i];
            best_i = i;
        }
    }
    best_i
}

struct Lcg64 { state: u64 }
impl Lcg64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next() >> 40) as u32;
        (bits as f32) * (1.0 / (1u32 << 24) as f32)
    }
}

/// Apply repetition / frequency / presence penalties to `logits` in
/// place, based on the token IDs that appear in `recent_tokens`.
///
/// - **Repetition penalty** (multiplicative, llama.cpp-style): positive
///   logits are divided by the penalty, negative logits are multiplied
///   by it. So `repetition_penalty = 1.2` makes a repeated token's logit
///   move toward zero from either side.
/// - **Frequency penalty** (additive, OpenAI-style): subtract
///   `frequency_penalty * count(token)` from each recent token's logit.
/// - **Presence penalty** (additive, OpenAI-style): subtract
///   `presence_penalty` from any token that appears at least once.
fn apply_penalties(logits: &mut [f32], cfg: &SamplingConfig, recent_tokens: &[u32]) {
    if recent_tokens.is_empty() { return; }
    let no_rep = cfg.repetition_penalty == 1.0;
    let no_freq = cfg.frequency_penalty == 0.0;
    let no_pres = cfg.presence_penalty == 0.0;
    if no_rep && no_freq && no_pres { return; }

    // Count token frequencies. Use a small map; for typical decode the
    // history is short enough that a linear pass is fine.
    use std::collections::HashMap;
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &t in recent_tokens {
        *counts.entry(t).or_insert(0) += 1;
    }

    for (&tok, &count) in counts.iter() {
        let idx = tok as usize;
        if idx >= logits.len() { continue; }
        let v = logits[idx];

        // Repetition penalty (multiplicative).
        if !no_rep {
            logits[idx] = if v > 0.0 {
                v / cfg.repetition_penalty
            } else {
                v * cfg.repetition_penalty
            };
        }
        // Frequency penalty (additive, scales with count).
        if !no_freq {
            logits[idx] -= cfg.frequency_penalty * count as f32;
        }
        // Presence penalty (additive, single hit).
        if !no_pres {
            logits[idx] -= cfg.presence_penalty;
        }
    }
}

/// Apply logit biases to `logits` in place.
fn apply_logit_bias(logits: &mut [f32], bias: &[(u32, f32)]) {
    for &(tok, b) in bias {
        let idx = tok as usize;
        if idx < logits.len() {
            if b == f32::NEG_INFINITY {
                logits[idx] = f32::NEG_INFINITY;
            } else {
                logits[idx] += b;
            }
        }
    }
}
