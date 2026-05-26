//! Ternary transformer forward pass.
//!
//! All matmuls dispatch through [`TernaryMatrix::matvec`]. No f32 weight
//! matrices anywhere on the hot path — the only floats are activations,
//! RMSNorm gains, and per-row scales. The architecture is a standard
//! decoder-only transformer: token embedding → N causal-attention +
//! gated-FFN blocks → final RMSNorm → LM head.
//!
//! R28.1.0 caveats (called out so future readers don't mistake the
//! current state for the final story):
//!
//! - **No KV cache.** Generation re-runs the full forward pass each step
//!   (`O(seq²)` instead of `O(seq)` total). R28.1.1 will plug into the
//!   KV cache types already in `jouleclaw-loader-gguf::kv_cache`.
//! - **Greedy argmax sampling.** Temperature/top-p exist in
//!   `jouleclaw-loader-gguf::sample` and will be wired in R28.1.1.
//! - **Byte-level tokenizer.** Vocab is 256; each byte is a token.
//!   Avoids the BPE dependency for the demo. Real BitNet weights use
//!   SentencePiece and need jouleclaw-loader-gguf's `Vocab::from_gguf`.
//! - **No bias terms.** Pure matmuls, in line with BitNet b1.58 / Bonsai.
//! - **Single-precision activations.** f32 throughout. Bit-reproducible
//!   per-platform; cross-platform reproducibility needs a software libm
//!   (same caveat as R29 Liquid).

use crate::ternary::TernaryMatrix;

/// Numerically-stable RMSNorm: `y = (x / sqrt(mean(x²) + eps)) ⊙ gamma`.
pub fn rmsnorm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    debug_assert_eq!(gamma.len(), n);
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / (n as f32);
    let denom = (mean_sq + eps).sqrt();
    x.iter()
        .zip(gamma.iter())
        .map(|(xi, gi)| (xi / denom) * gi)
        .collect()
}

/// Numerically-stable softmax: subtract max before exponentiating.
pub fn softmax(x: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// SiLU activation: `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    if x > 50.0 {
        x
    } else if x < -50.0 {
        0.0
    } else {
        x / (1.0 + (-x).exp())
    }
}

/// One transformer block: causal multi-head attention + gated FFN, each
/// pre-normed with RMSNorm. All projections use ternary weights.
pub struct TernaryBlock {
    /// Causal attention projections. Shape: `[d_model, d_model]` for q/o
    /// and `[d_kv, d_model]` for k/v where `d_kv = d_head * n_kv_heads`.
    pub w_q: TernaryMatrix,
    pub w_k: TernaryMatrix,
    pub w_v: TernaryMatrix,
    pub w_o: TernaryMatrix,
    /// Gated FFN: hidden = SiLU(W_gate · x) ⊙ (W_up · x); out = W_down · hidden.
    pub w_gate: TernaryMatrix,
    pub w_up: TernaryMatrix,
    pub w_down: TernaryMatrix,
    /// Per-channel RMSNorm gains, length `d_model`.
    pub norm_attn: Vec<f32>,
    pub norm_ffn: Vec<f32>,
}

/// Decoder-only transformer with ternary weights.
pub struct TernaryDecoder {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ffn: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_layers: usize,
    /// Token embedding table: `[vocab_size, d_model]`. Stored as ternary
    /// so the embedding lookup is itself a ternary read (sign + scale).
    pub embed: TernaryMatrix,
    pub blocks: Vec<TernaryBlock>,
    /// Final pre-head RMSNorm gains.
    pub norm_final: Vec<f32>,
    /// LM head: `[vocab_size, d_model]`. Ternary.
    pub lm_head: TernaryMatrix,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
}

impl TernaryDecoder {
    /// Head dim derived from `d_model / n_heads`.
    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Lookup the embedding for a single token. The embedding table is
    /// `[vocab_size, d_model]` row-major; we extract row `token_id` and
    /// scale by the row's ternary scale. Equivalent to selecting the
    /// `token_id`-th row of `embed.to_f32()`, but avoids the full
    /// dequantization.
    fn lookup_embedding(&self, token_id: u32) -> Vec<f32> {
        let row = token_id as usize;
        let mut out = Vec::with_capacity(self.d_model);
        for c in 0..self.d_model {
            out.push(self.embed.at(row, c));
        }
        out
    }

    /// Run a forward pass over a sequence of token ids. Returns logits
    /// per position: shape `[seq_len, vocab_size]`. Causal — each
    /// position only attends to itself and prior positions.
    pub fn forward(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        // x: per-position activation vectors, shape [seq_len][d_model].
        let mut x: Vec<Vec<f32>> = tokens.iter()
            .map(|&t| self.lookup_embedding(t))
            .collect();

        for block in &self.blocks {
            x = self.run_block(block, x);
        }

        // Final norm + LM head, applied per position.
        x.into_iter()
            .map(|h| {
                let n = rmsnorm(&h, &self.norm_final, self.rms_eps);
                let mut logits = vec![0.0_f32; self.vocab_size];
                self.lm_head.matvec(&n, &mut logits)
                    .expect("lm_head dim mismatch");
                logits
            })
            .collect()
    }

    /// One block: residual(attn(rmsnorm(x))) + residual(ffn(rmsnorm(x))).
    fn run_block(&self, block: &TernaryBlock, mut x: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let d_model = self.d_model;
        let d_head = self.d_head();
        let n_heads = self.n_heads;
        let n_kv = self.n_kv_heads;
        let d_kv = d_head * n_kv;

        let seq_len = x.len();

        // --- Multi-head causal attention ---
        // Pre-norm.
        let pre: Vec<Vec<f32>> = x.iter()
            .map(|h| rmsnorm(h, &block.norm_attn, self.rms_eps))
            .collect();

        // Project Q, K, V per position.
        let mut q_seq: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut k_seq: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut v_seq: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for p in &pre {
            let mut q = vec![0.0_f32; d_model];
            let mut k = vec![0.0_f32; d_kv];
            let mut v = vec![0.0_f32; d_kv];
            block.w_q.matvec(p, &mut q).expect("w_q dim");
            block.w_k.matvec(p, &mut k).expect("w_k dim");
            block.w_v.matvec(p, &mut v).expect("w_v dim");
            q_seq.push(q);
            k_seq.push(k);
            v_seq.push(v);
        }

        // Per-head causal attention. Heads are contiguous slices of the
        // projection: head h covers [h*d_head, (h+1)*d_head).
        // For GQA (n_kv < n_heads), each KV head is shared by
        // n_heads / n_kv query heads.
        let mut attn_out: Vec<Vec<f32>> = vec![vec![0.0_f32; d_model]; seq_len];
        let scale = 1.0_f32 / (d_head as f32).sqrt();

        for pos in 0..seq_len {
            for h in 0..n_heads {
                let kv_h = h * n_kv / n_heads;
                let q_off = h * d_head;
                let kv_off = kv_h * d_head;

                // Compute attention scores for query at `pos` against
                // all keys at positions [0..=pos] (causal mask).
                let mut scores = vec![0.0_f32; pos + 1];
                for j in 0..=pos {
                    let mut dot = 0.0_f32;
                    for d in 0..d_head {
                        dot += q_seq[pos][q_off + d] * k_seq[j][kv_off + d];
                    }
                    scores[j] = dot * scale;
                }
                softmax(&mut scores);

                // Weighted sum of values.
                for d in 0..d_head {
                    let mut acc = 0.0_f32;
                    for j in 0..=pos {
                        acc += scores[j] * v_seq[j][kv_off + d];
                    }
                    attn_out[pos][q_off + d] = acc;
                }
            }
        }

        // Output projection + residual.
        for pos in 0..seq_len {
            let mut proj = vec![0.0_f32; d_model];
            block.w_o.matvec(&attn_out[pos], &mut proj).expect("w_o dim");
            for d in 0..d_model {
                x[pos][d] += proj[d];
            }
        }

        // --- Gated FFN ---
        for pos in 0..seq_len {
            let n = rmsnorm(&x[pos], &block.norm_ffn, self.rms_eps);
            let mut gate = vec![0.0_f32; self.d_ffn];
            let mut up = vec![0.0_f32; self.d_ffn];
            block.w_gate.matvec(&n, &mut gate).expect("w_gate dim");
            block.w_up.matvec(&n, &mut up).expect("w_up dim");
            // hidden = SiLU(gate) ⊙ up
            let hidden: Vec<f32> = gate.into_iter()
                .zip(up.into_iter())
                .map(|(g, u)| silu(g) * u)
                .collect();
            let mut down = vec![0.0_f32; d_model];
            block.w_down.matvec(&hidden, &mut down).expect("w_down dim");
            for d in 0..d_model {
                x[pos][d] += down[d];
            }
        }

        x
    }

    /// Autoregressive greedy generation. Starts from `prompt` (sequence
    /// of token ids), runs forward, takes argmax of the final-position
    /// logits, appends, repeats up to `max_new_tokens`. Quadratic in
    /// total length because there's no KV cache yet (R28.1.1).
    pub fn generate_greedy(&self, prompt: &[u32], max_new_tokens: usize) -> Vec<u32> {
        let mut tokens = prompt.to_vec();
        for _ in 0..max_new_tokens {
            let logits = self.forward(&tokens);
            let last = logits.last().expect("forward must return ≥ 1 row");
            // Argmax over vocab.
            let (next, _) = last.iter().enumerate().fold(
                (0_usize, f32::NEG_INFINITY),
                |(best_i, best_v), (i, &v)| {
                    if v > best_v { (i, v) } else { (best_i, best_v) }
                },
            );
            tokens.push(next as u32);
        }
        tokens
    }

    /// Convenience: byte-level encode / decode for the R28.1.0 demo.
    /// Real BitNet weights need SentencePiece; this is a stand-in.
    pub fn encode_bytes(text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    pub fn decode_bytes(tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|&t| if t < 256 { Some(t as u8) } else { None })
            .collect();
        // Lossy: random tokens won't form valid utf-8.
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_preserves_zero_input() {
        let g = vec![1.0_f32; 8];
        let x = vec![0.0_f32; 8];
        let y = rmsnorm(&x, &g, 1e-5);
        for v in &y {
            assert!(v.abs() < 1e-3, "rmsnorm(zero) should be near zero, got {}", v);
        }
    }

    #[test]
    fn rmsnorm_scales_to_unit_rms_when_gamma_is_one() {
        let g = vec![1.0_f32; 4];
        let x = vec![2.0_f32, -2.0, 2.0, -2.0]; // mean(x²) = 4
        let y = rmsnorm(&x, &g, 0.0);
        let mean_sq: f32 = y.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((mean_sq - 1.0).abs() < 1e-5,
            "post-rmsnorm mean(x²) should be 1.0, got {}", mean_sq);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        softmax(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        // Monotone preservation: largest input → largest output.
        for i in 1..x.len() {
            assert!(x[i] > x[i - 1]);
        }
    }

    #[test]
    fn softmax_handles_extreme_values() {
        // Without max-subtraction, exp(1000) overflows. The stable
        // softmax we use should produce 1.0 for the max and ~0 for others.
        let mut x = vec![0.0_f32, 1000.0, 0.0];
        softmax(&mut x);
        assert!(x[1] > 0.99);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn silu_matches_expected_values() {
        assert!(silu(0.0).abs() < 1e-7);
        // silu(1) ≈ 0.7311
        assert!((silu(1.0) - 0.7310585786300049_f32).abs() < 1e-5);
    }
}
