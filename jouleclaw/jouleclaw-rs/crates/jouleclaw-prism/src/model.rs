//! Ternary model + synthetic generator.
//!
//! [`TernaryModel`] bundles a [`TernaryDecoder`] with config so it can
//! be constructed once and reused across queries.
//!
//! [`synthetic_model`] generates a tiny random-weight ternary model for
//! pipeline testing. The forward pass on a synthetic model produces
//! valid (non-NaN, properly-shaped) logits, but the sampled tokens are
//! random — the point is to exercise the kernel + tier integration end
//! to end without requiring real BitNet weights, which is R28.1.1 work.

use crate::forward::{TernaryBlock, TernaryDecoder};
use crate::ternary::TernaryMatrix;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ffn: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_layers: usize,
    pub rms_eps: f32,
}

impl ModelConfig {
    /// A tiny config for tests / demos. Byte-level vocab, 32-dim model,
    /// 2 layers. Fast to construct and run.
    pub fn tiny_byte() -> Self {
        Self {
            vocab_size: 256,
            d_model: 32,
            d_ffn: 128,
            n_heads: 4,
            n_kv_heads: 4,
            n_layers: 2,
            rms_eps: 1e-5,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.d_model == 0 || self.vocab_size == 0 || self.n_heads == 0 {
            return Err("dims must be > 0".into());
        }
        if self.d_model % self.n_heads != 0 {
            return Err(format!(
                "d_model {} not divisible by n_heads {}",
                self.d_model, self.n_heads
            ));
        }
        if self.n_heads % self.n_kv_heads != 0 || self.n_kv_heads == 0 {
            return Err(format!(
                "n_heads {} must be a positive multiple of n_kv_heads {}",
                self.n_heads, self.n_kv_heads
            ));
        }
        Ok(())
    }
}

/// Tiny deterministic xorshift PRNG. We avoid a crate dep so prism stays
/// `jouleclaw-cascade`-only. Same generator joule-l2 uses for synthetic data.
fn xorshift_step(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn xorshift_f32(state: &mut u64) -> f32 {
    // Map to [-0.5, 0.5).
    let x = xorshift_step(state);
    let mantissa = (x >> 40) as u32; // 24-bit mantissa
    let u = (mantissa as f32) / (1u32 << 24) as f32; // [0, 1)
    u - 0.5
}

/// Construct a synthetic ternary model with random weights. Useful for
/// tests, microbenches, and proving the forward pass runs end to end.
/// Output tokens will be random — quality requires real trained weights.
pub fn synthetic_model(cfg: ModelConfig, seed: u64) -> Result<TernaryDecoder, String> {
    cfg.validate()?;
    let mut state = seed.max(1);

    let make_ternary = |rows: usize, cols: usize, state: &mut u64| -> TernaryMatrix {
        let mut w = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            w.push(xorshift_f32(state));
        }
        TernaryMatrix::from_f32(rows, cols, &w).expect("from_f32")
    };

    let d_kv = cfg.d_model * cfg.n_kv_heads / cfg.n_heads;

    let embed = make_ternary(cfg.vocab_size, cfg.d_model, &mut state);
    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for _ in 0..cfg.n_layers {
        let w_q = make_ternary(cfg.d_model, cfg.d_model, &mut state);
        let w_k = make_ternary(d_kv, cfg.d_model, &mut state);
        let w_v = make_ternary(d_kv, cfg.d_model, &mut state);
        let w_o = make_ternary(cfg.d_model, cfg.d_model, &mut state);
        let w_gate = make_ternary(cfg.d_ffn, cfg.d_model, &mut state);
        let w_up = make_ternary(cfg.d_ffn, cfg.d_model, &mut state);
        let w_down = make_ternary(cfg.d_model, cfg.d_ffn, &mut state);
        let norm_attn = vec![1.0_f32; cfg.d_model];
        let norm_ffn = vec![1.0_f32; cfg.d_model];
        blocks.push(TernaryBlock {
            w_q, w_k, w_v, w_o,
            w_gate, w_up, w_down,
            norm_attn, norm_ffn,
        });
    }
    let norm_final = vec![1.0_f32; cfg.d_model];
    let lm_head = make_ternary(cfg.vocab_size, cfg.d_model, &mut state);

    Ok(TernaryDecoder {
        vocab_size: cfg.vocab_size,
        d_model: cfg.d_model,
        d_ffn: cfg.d_ffn,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        n_layers: cfg.n_layers,
        embed,
        blocks,
        norm_final,
        lm_head,
        rms_eps: cfg.rms_eps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_model_builds_with_tiny_config() {
        let m = synthetic_model(ModelConfig::tiny_byte(), 42).unwrap();
        assert_eq!(m.n_layers, 2);
        assert_eq!(m.d_model, 32);
        assert_eq!(m.vocab_size, 256);
        assert_eq!(m.blocks.len(), 2);
    }

    #[test]
    fn config_validates_head_divisibility() {
        let mut bad = ModelConfig::tiny_byte();
        bad.n_heads = 5; // 32 % 5 != 0
        assert!(synthetic_model(bad, 0).is_err());
    }

    #[test]
    fn forward_pass_runs_on_synthetic_model_without_nans() {
        let m = synthetic_model(ModelConfig::tiny_byte(), 123).unwrap();
        let tokens = vec![b'h' as u32, b'i' as u32];
        let logits = m.forward(&tokens);
        assert_eq!(logits.len(), 2);
        assert_eq!(logits[0].len(), 256);
        for row in &logits {
            for &v in row {
                assert!(v.is_finite(), "logit is non-finite: {}", v);
            }
        }
    }

    #[test]
    fn forward_pass_is_deterministic_per_seed() {
        let m1 = synthetic_model(ModelConfig::tiny_byte(), 7).unwrap();
        let m2 = synthetic_model(ModelConfig::tiny_byte(), 7).unwrap();
        let toks = vec![b'a' as u32, b'b' as u32, b'c' as u32];
        let a = m1.forward(&toks);
        let b = m2.forward(&toks);
        for (ar, br) in a.iter().zip(b.iter()) {
            assert_eq!(ar, br, "same seed must give bit-identical logits");
        }
    }

    #[test]
    fn different_seeds_produce_different_outputs() {
        let m1 = synthetic_model(ModelConfig::tiny_byte(), 1).unwrap();
        let m2 = synthetic_model(ModelConfig::tiny_byte(), 999).unwrap();
        let toks = vec![b'x' as u32];
        let a = m1.forward(&toks);
        let b = m2.forward(&toks);
        assert_ne!(a, b, "different seeds must give different logits");
    }

    #[test]
    fn generate_greedy_produces_max_new_tokens() {
        let m = synthetic_model(ModelConfig::tiny_byte(), 0xC0FFEE).unwrap();
        let prompt = TernaryDecoder::encode_bytes("hi");
        let out = m.generate_greedy(&prompt, 8);
        assert_eq!(out.len(), prompt.len() + 8);
        // First two tokens are still the prompt.
        assert_eq!(out[0], prompt[0]);
        assert_eq!(out[1], prompt[1]);
    }
}
