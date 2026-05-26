//! DeBERTa-v3 embedding-layer forward pass.
//!
//! v3-specific shape: `position_biased_input = false` and
//! `type_vocab_size = 0` collapse the embedding to:
//!
//! ```text
//! embeddings = LayerNorm(word_embeddings[input_ids])
//! embeddings *= attention_mask  // broadcast along hidden
//! ```
//!
//! The output is the tensor the first encoder layer consumes.

use crate::weights::EmbeddingWeights;

/// Run the embedding layer over a single sequence. Output is a flat
/// row-major buffer of length `seq_len * hidden_size` that the
/// encoder will treat as `[seq_len, hidden_size]`.
pub fn forward_embedding(
    input_ids: &[u32],
    attention_mask: &[u32],
    weights: &EmbeddingWeights,
    layer_norm_eps: f32,
) -> Vec<f32> {
    assert_eq!(
        input_ids.len(),
        attention_mask.len(),
        "input_ids/attention_mask length mismatch"
    );
    let seq_len = input_ids.len();
    let hidden = weights.word_embeddings.cols();
    let mut out = vec![0.0_f32; seq_len * hidden];

    for (i, &id) in input_ids.iter().enumerate() {
        let row_src = weights.word_embeddings.row(id as usize);
        let row_dst = &mut out[i * hidden..(i + 1) * hidden];

        // Compute LayerNorm over the row, then scale by gamma + add beta.
        layer_norm_into(
            row_src,
            row_dst,
            &weights.layer_norm_weight,
            &weights.layer_norm_bias,
            layer_norm_eps,
        );

        // Mask multiply — broadcast across hidden.
        if attention_mask[i] == 0 {
            for v in row_dst.iter_mut() {
                *v = 0.0;
            }
        }
    }

    out
}

/// LayerNorm `(x - mean) / sqrt(var + eps) * gamma + beta` over the
/// last dim. Computes mean and variance in fp32 — same precision
/// path PyTorch uses when the module is materialized in fp32.
fn layer_norm_into(input: &[f32], output: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = input.len();
    debug_assert_eq!(n, output.len());
    debug_assert_eq!(n, gamma.len());
    debug_assert_eq!(n, beta.len());

    // Mean.
    let mut sum = 0.0_f64;
    for &x in input {
        sum += x as f64;
    }
    let mean = (sum / n as f64) as f32;

    // Population variance (PyTorch default since 1.10 for nn.LayerNorm).
    let mut sq = 0.0_f64;
    for &x in input {
        let d = x - mean;
        sq += (d as f64) * (d as f64);
    }
    let var = (sq / n as f64) as f32;
    let inv_std = 1.0 / (var + eps).sqrt();

    for j in 0..n {
        let normed = (input[j] - mean) * inv_std;
        output[j] = normed * gamma[j] + beta[j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::ModelInventory;
    use crate::weights::Weights;
    use serde::Deserialize;
    use std::path::PathBuf;

    #[derive(Debug, Deserialize)]
    struct Fixture {
        seq_len: usize,
        hidden_size: usize,
        input_ids: Vec<u32>,
        attention_mask: Vec<u32>,
        embedding_output_flat: Vec<f32>,
    }

    fn workspace_root() -> PathBuf {
        // CARGO_MANIFEST_DIR points to crates/jouleclaw-deberta at
        // compile time; walking up twice gets the workspace root.
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent() // crates/
            .and_then(|p| p.parent()) // workspace root
            .map(|p| p.to_path_buf())
            .expect("workspace root above CARGO_MANIFEST_DIR")
    }

    fn model_dir() -> Option<PathBuf> {
        let p = workspace_root().join("models/deberta-v3-large-mnli");
        if p.join("model.safetensors").exists() {
            Some(p)
        } else {
            None
        }
    }

    fn load_fixture() -> Option<Fixture> {
        let p = workspace_root()
            .join("crates/jouleclaw-deberta/fixtures/embedding_simple_entail.json");
        let raw = std::fs::read_to_string(p).ok()?;
        serde_json::from_str(&raw).ok()
    }

    /// Headline test: load real weights, run the embedding, compare
    /// numerically to the HF reference tensor. Pass tolerance is
    /// max-abs 1e-3 / mean-abs 1e-5 — driven by fp16 storage of the
    /// weights upcast to fp32. Tighter than tightening would be
    /// false-confidence; looser would mask a real algorithmic bug.
    #[test]
    fn forward_matches_hf_reference_within_fp16_tolerance() {
        let Some(dir) = model_dir() else {
            eprintln!("skip: model not downloaded");
            return;
        };
        let Some(fx) = load_fixture() else {
            eprintln!("skip: fixture missing — run scripts/hf_reference_embedding.py");
            return;
        };

        let inv = ModelInventory::from_dir(&dir).expect("inventory");
        let weights = Weights::load_embeddings_only(&dir, &inv).expect("weights");
        let out = forward_embedding(
            &fx.input_ids,
            &fx.attention_mask,
            &weights.embeddings,
            weights.config.layer_norm_eps,
        );

        assert_eq!(out.len(), fx.seq_len * fx.hidden_size, "output size mismatch");
        assert_eq!(out.len(), fx.embedding_output_flat.len());

        let mut max_abs = 0.0_f32;
        let mut sum_abs = 0.0_f64;
        let mut max_idx = 0_usize;
        for (i, (&ours, &theirs)) in out.iter().zip(fx.embedding_output_flat.iter()).enumerate() {
            let d = (ours - theirs).abs();
            sum_abs += d as f64;
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
        }
        let mean_abs = (sum_abs / out.len() as f64) as f32;

        eprintln!(
            "embedding diff vs HF reference: max_abs={max_abs:.6} mean_abs={mean_abs:.6} \
             (worst at flat idx {max_idx}: ours={} theirs={})",
            out[max_idx], fx.embedding_output_flat[max_idx]
        );

        assert!(
            max_abs < 1e-3,
            "max abs diff {max_abs} exceeds 1e-3 tolerance"
        );
        assert!(
            mean_abs < 1e-5,
            "mean abs diff {mean_abs} exceeds 1e-5 tolerance"
        );
    }

    #[test]
    fn layer_norm_recovers_identity_with_unit_gamma_zero_beta() {
        // For a constant input, LayerNorm produces zeros (because
        // mean equals input; variance is zero so 1/sqrt(eps) scales
        // a zero residual). For a non-constant input, LayerNorm
        // produces a zero-mean unit-variance vector pre-affine.
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        layer_norm_into(&input, &mut out, &gamma, &beta, 1e-7);
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        let var: f32 = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "post-LN mean should be ~0, got {mean}");
        assert!(
            (var - 1.0).abs() < 1e-3,
            "post-LN variance should be ~1, got {var}"
        );
    }

    #[test]
    fn padding_token_position_outputs_zeros() {
        // Construct a synthetic 2-token sequence with the second
        // token padded. After embedding + mask, the padded row must
        // be all zero regardless of LayerNorm output.
        let hidden = 4;
        let word = crate::weights::FloatTensor {
            shape: vec![2, hidden],
            data: vec![
                0.1, 0.2, 0.3, 0.4, // token 0
                0.5, 0.6, 0.7, 0.8, // token 1
            ],
        };
        let weights = EmbeddingWeights {
            word_embeddings: word,
            layer_norm_weight: vec![1.0; hidden],
            layer_norm_bias: vec![0.0; hidden],
        };
        let input_ids = vec![0, 1];
        let attention_mask = vec![1, 0];
        let out = forward_embedding(&input_ids, &attention_mask, &weights, 1e-7);
        let row1 = &out[hidden..];
        assert!(
            row1.iter().all(|&v| v == 0.0),
            "padded row should be zeroed, got {row1:?}"
        );
    }
}
