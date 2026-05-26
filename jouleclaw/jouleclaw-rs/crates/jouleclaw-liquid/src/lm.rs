//! `LiquidLanguageModel` — a CfC-recurrent language model built on
//! [`crate::model::LiquidModel`].
//!
//! Where Prism uses attention to mix across positions, Liquid uses pure
//! recurrence: at each step the model receives one token's embedding,
//! the CfC stack updates its state, and the LM head projects that state
//! into vocab logits. No attention, no KV cache, no quadratic blow-up.
//! This is the architectural personality the Liquid AI line bets on
//! for sequence tasks at edge scale.
//!
//! R29.1 caveats (parallel to Prism's R28.1 disclosures):
//!
//! - **Random-weight synthetic models only.** Real LFM checkpoints land
//!   in R29.1.1.
//! - **Byte-level tokenizer.** Same stand-in as Prism uses; vocab=256.
//!   SentencePiece comes with real weights.
//! - **Greedy argmax sampling.** Top-p/temperature later.
//! - **f32 weights everywhere.** No quantization yet — Prism is the
//!   weight-precision story; Liquid's story is the cell dynamics.

use crate::cell::CfcCell;
use crate::model::LiquidModel;

/// CfC-recurrent language model: token embedding → LiquidModel stack →
/// LM head over the vocabulary.
pub struct LiquidLanguageModel {
    pub vocab_size: usize,
    pub d_model: usize,
    /// Per-token f32 embedding, length `vocab_size × d_model` row-major.
    pub embed: Vec<f32>,
    /// The recurrent CfC stack.
    pub model: LiquidModel,
    /// LM head, shape `[vocab_size, d_state]` row-major where `d_state`
    /// is the last layer's output dimension.
    pub head: Vec<f32>,
    /// Cached `d_state` (last cell's state_dim).
    d_state: usize,
}

impl LiquidLanguageModel {
    /// Build with explicit weight tables. Validates dimensions.
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        embed: Vec<f32>,
        model: LiquidModel,
        head: Vec<f32>,
    ) -> Result<Self, String> {
        if vocab_size == 0 || d_model == 0 {
            return Err("vocab_size and d_model must be > 0".into());
        }
        if embed.len() != vocab_size * d_model {
            return Err(format!(
                "embed table {} != vocab_size {} × d_model {}",
                embed.len(),
                vocab_size,
                d_model
            ));
        }
        if model.input_dim() != d_model {
            return Err(format!(
                "first cell input_dim {} != d_model {}",
                model.input_dim(),
                d_model
            ));
        }
        let d_state = model.output_dim();
        if head.len() != vocab_size * d_state {
            return Err(format!(
                "head {} != vocab_size {} × d_state {}",
                head.len(),
                vocab_size,
                d_state
            ));
        }
        Ok(Self { vocab_size, d_model, embed, model, head, d_state })
    }

    /// Look up the embedding row for `token_id`.
    fn embedding(&self, token_id: u32) -> Vec<f32> {
        let row = token_id as usize;
        let off = row * self.d_model;
        self.embed[off..off + self.d_model].to_vec()
    }

    /// Project the last cell's state into vocab logits.
    fn project_logits(&self, state: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0_f32; self.vocab_size];
        for v in 0..self.vocab_size {
            let mut acc = 0.0_f32;
            let row_off = v * self.d_state;
            for d in 0..self.d_state {
                acc += self.head[row_off + d] * state[d];
            }
            logits[v] = acc;
        }
        logits
    }

    /// Reset recurrent state (start a new sequence).
    pub fn reset(&mut self) {
        self.model.reset();
    }

    /// Feed one token through the model, advancing recurrent state.
    /// Returns the LM logits over the vocabulary for that step.
    pub fn step_token(&mut self, token_id: u32) -> Result<Vec<f32>, String> {
        let u = self.embedding(token_id);
        let mut out = vec![0.0_f32; self.d_state];
        self.model
            .step(&u, &mut out)
            .map_err(|e| format!("liquid step: {}", e))?;
        Ok(self.project_logits(&out))
    }

    /// Generate `max_new_tokens` greedy continuations from `prompt`.
    /// Resets recurrent state at the start of the call. Returns the
    /// full token sequence (prompt + continuation).
    pub fn generate_greedy(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, String> {
        self.reset();
        // Burn through the prompt; the last logits are the prediction
        // for the next token.
        let mut last_logits: Option<Vec<f32>> = None;
        for &t in prompt {
            last_logits = Some(self.step_token(t)?);
        }
        let mut tokens = prompt.to_vec();
        let mut next_logits = last_logits.unwrap_or_else(|| vec![0.0_f32; self.vocab_size]);
        for _ in 0..max_new_tokens {
            // Argmax.
            let (next, _) = next_logits.iter().enumerate().fold(
                (0_usize, f32::NEG_INFINITY),
                |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
            );
            tokens.push(next as u32);
            next_logits = self.step_token(next as u32)?;
        }
        Ok(tokens)
    }

    /// Byte-level encode (vocab=256, ascii passthrough).
    pub fn encode_bytes(text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    /// Byte-level decode, lossy-UTF-8 on invalid sequences.
    pub fn decode_bytes(tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|&t| if t < 256 { Some(t as u8) } else { None })
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// First-order joule estimate for one token step. Sums the
    /// CfC stack's per-step joules + embedding lookup + LM head matvec.
    pub fn step_joules(&self) -> f64 {
        let cfc = self.model.step_joules();
        // Embedding lookup: just memcpy, ~1 nJ.
        let emb = 1e-9;
        // LM head: vocab_size * d_state f32 FMAs at ~10 pJ each.
        let head_ops = (self.vocab_size as f64) * (self.d_state as f64);
        let head = head_ops * 10.0 * 1e-12;
        cfc + emb + head
    }
}

/// Synthetic-weight Liquid language model, parallel to Prism's
/// `synthetic_model`. Random weights drawn from a deterministic
/// xorshift PRNG. Outputs are noise until R29.1.1 swaps in trained
/// weights.
#[derive(Debug, Clone)]
pub struct LmConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
}

impl LmConfig {
    /// Tiny byte-level config for tests and demos.
    pub fn tiny_byte() -> Self {
        Self { vocab_size: 256, d_model: 16, n_layers: 2 }
    }
}

fn xs(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn xs_f32(state: &mut u64) -> f32 {
    let x = xs(state);
    let m = (x >> 40) as u32;
    let u = (m as f32) / (1u32 << 24) as f32;
    u - 0.5
}

pub fn synthetic_lm(cfg: LmConfig, seed: u64) -> Result<LiquidLanguageModel, String> {
    if cfg.vocab_size == 0 || cfg.d_model == 0 || cfg.n_layers == 0 {
        return Err("LmConfig dims must be > 0".into());
    }
    let mut state = seed.max(1);

    // Embedding table.
    let mut embed = Vec::with_capacity(cfg.vocab_size * cfg.d_model);
    for _ in 0..cfg.vocab_size * cfg.d_model {
        // Small magnitude so the CfC stack sees inputs in a reasonable range.
        embed.push(xs_f32(&mut state) * 0.5);
    }

    // CfC stack: each cell has state_dim = d_model, input_dim = d_model.
    // (Uniform sizing keeps the LM head's projection simple.)
    let mut cells = Vec::with_capacity(cfg.n_layers);
    for _ in 0..cfg.n_layers {
        let mut c = CfcCell::zeros(cfg.d_model, cfg.d_model)
            .map_err(|e| format!("cell: {}", e))?;
        for w in c.w_f.iter_mut() { *w = xs_f32(&mut state) * 0.3; }
        for w in c.w_g.iter_mut() { *w = xs_f32(&mut state) * 0.3; }
        for w in c.w_h.iter_mut() { *w = xs_f32(&mut state) * 0.3; }
        for b in c.b_f.iter_mut() { *b = xs_f32(&mut state) * 0.1; }
        for b in c.b_g.iter_mut() { *b = xs_f32(&mut state) * 0.1; }
        for b in c.b_h.iter_mut() { *b = xs_f32(&mut state) * 0.1; }
        for t in c.theta_t.iter_mut() { *t = xs_f32(&mut state) * 0.1; }
        cells.push(c);
    }
    let model = LiquidModel::new(cells).map_err(|e| format!("model: {}", e))?;

    // LM head: [vocab_size, d_model].
    let mut head = Vec::with_capacity(cfg.vocab_size * cfg.d_model);
    for _ in 0..cfg.vocab_size * cfg.d_model {
        head.push(xs_f32(&mut state));
    }

    LiquidLanguageModel::new(cfg.vocab_size, cfg.d_model, embed, model, head)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_lm_builds_with_tiny_config() {
        let lm = synthetic_lm(LmConfig::tiny_byte(), 42).unwrap();
        assert_eq!(lm.vocab_size, 256);
        assert_eq!(lm.d_model, 16);
    }

    #[test]
    fn step_token_produces_vocab_sized_logits() {
        let mut lm = synthetic_lm(LmConfig::tiny_byte(), 7).unwrap();
        let logits = lm.step_token(b'a' as u32).unwrap();
        assert_eq!(logits.len(), 256);
        for &v in &logits {
            assert!(v.is_finite(), "non-finite logit: {}", v);
        }
    }

    #[test]
    fn generate_greedy_returns_prompt_plus_continuation() {
        let mut lm = synthetic_lm(LmConfig::tiny_byte(), 0xBEEF).unwrap();
        let prompt = LiquidLanguageModel::encode_bytes("hi");
        let out = lm.generate_greedy(&prompt, 6).unwrap();
        assert_eq!(out.len(), prompt.len() + 6);
        assert_eq!(out[0], prompt[0]);
        assert_eq!(out[1], prompt[1]);
    }

    #[test]
    fn generation_is_deterministic_per_seed() {
        let mut a = synthetic_lm(LmConfig::tiny_byte(), 123).unwrap();
        let mut b = synthetic_lm(LmConfig::tiny_byte(), 123).unwrap();
        let prompt = LiquidLanguageModel::encode_bytes("test");
        let oa = a.generate_greedy(&prompt, 4).unwrap();
        let ob = b.generate_greedy(&prompt, 4).unwrap();
        assert_eq!(oa, ob);
    }

    #[test]
    fn dim_mismatch_in_constructor_is_caught() {
        // d_model=4 but embed table sized for 8.
        let bad_embed = vec![0.0_f32; 256 * 8];
        let model = LiquidModel::new(vec![CfcCell::zeros(4, 4).unwrap()]).unwrap();
        let head = vec![0.0_f32; 256 * 4];
        let r = LiquidLanguageModel::new(256, 4, bad_embed, model, head);
        assert!(r.is_err());
    }
}
