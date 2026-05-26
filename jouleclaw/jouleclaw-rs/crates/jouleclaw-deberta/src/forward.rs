//! End-to-end forward pass: embedding → 24-layer encoder → pooler →
//! classifier.
//!
//! Encoder layer = attention sub-block (already verified in phase 4e)
//! followed by FFN sub-block:
//!
//! ```text
//! ffn(x) = LayerNorm(output_dense(GELU(intermediate_dense(x))) + x)
//! ```
//!
//! Classifier head = pooler on CLS token + linear → logits:
//!
//! ```text
//! pooled = GELU(pooler_dense(encoder_last[CLS]))
//! logits = classifier(pooled)   // [num_labels]
//! ```

use crate::attention::{
    build_relative_position, forward_attention, layer_norm_rel_embeddings,
};
use crate::config::ModelConfig;
use crate::embedding::forward_embedding;
use crate::tensor_ops::{
    gelu_inplace, layer_norm_rowwise, matmul_at_bias,
};
use crate::weights::{ClassificationHead, FfnWeights, Weights};

#[derive(Debug)]
pub enum ForwardError {
    /// One of the optional weight sections (encoder, head) wasn't
    /// loaded but is required.
    MissingWeights(&'static str),
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingWeights(s) => write!(f, "missing weights: {s}"),
        }
    }
}

impl std::error::Error for ForwardError {}

/// Result of running a single (premise, hypothesis) pair through the
/// full model.
#[derive(Debug, Clone)]
pub struct ForwardResult {
    /// `[L, hidden]` — output of layer 23.
    pub encoder_last_hidden: Vec<f32>,
    pub seq_len: usize,
    /// `[hidden]` — output of pooler.dense + GELU on CLS token.
    pub pooled: Vec<f32>,
    /// `[num_labels]` — raw NLI logits in the model's label order.
    pub logits: Vec<f32>,
}

/// FFN sub-block: intermediate dense → GELU → output dense →
/// residual + LayerNorm.
pub fn forward_ffn(
    hidden: &[f32],
    seq_len: usize,
    layer: &FfnWeights,
    cfg: &ModelConfig,
) -> Vec<f32> {
    let h = cfg.hidden_size;
    let ffn = cfg.intermediate_size;
    assert_eq!(hidden.len(), seq_len * h);

    // intermediate = GELU(hidden @ W.T + b) where W is [ffn, h].
    let mut inter = vec![0.0_f32; seq_len * ffn];
    matmul_at_bias(
        seq_len,
        h,
        ffn,
        hidden,
        &layer.intermediate_w.data,
        &layer.intermediate_b,
        &mut inter,
    );
    gelu_inplace(&mut inter);

    // output_proj = inter @ W.T + b where W is [h, ffn].
    let mut out_proj = vec![0.0_f32; seq_len * h];
    matmul_at_bias(
        seq_len,
        ffn,
        h,
        &inter,
        &layer.output_w.data,
        &layer.output_b,
        &mut out_proj,
    );

    // Residual + LayerNorm.
    for i in 0..(seq_len * h) {
        out_proj[i] += hidden[i];
    }
    let mut final_out = vec![0.0_f32; seq_len * h];
    layer_norm_rowwise(
        seq_len,
        h,
        &out_proj,
        &mut final_out,
        &layer.output_ln_gamma,
        &layer.output_ln_beta,
        cfg.layer_norm_eps,
    );
    final_out
}

/// Run the full encoder (all 24 layers) over the embedding output.
/// Returns the last hidden state `[seq_len, hidden_size]`.
pub fn forward_encoder(
    embedding_output: &[f32],
    seq_len: usize,
    attention_mask: &[u32],
    weights: &Weights,
) -> Result<Vec<f32>, ForwardError> {
    let encoder = weights
        .encoder
        .as_ref()
        .ok_or(ForwardError::MissingWeights("encoder"))?;
    let cfg = &weights.config;

    let relative_pos =
        build_relative_position(seq_len, cfg.position_buckets, cfg.max_relative_positions);
    let rel_embeddings = layer_norm_rel_embeddings(encoder, cfg);

    let mut hidden = embedding_output.to_vec();
    for layer in &encoder.layers {
        let attn_out = forward_attention(
            &hidden,
            seq_len,
            attention_mask,
            &relative_pos,
            &rel_embeddings,
            &layer.attention,
            cfg,
        );
        hidden = forward_ffn(&attn_out, seq_len, &layer.ffn, cfg);
    }
    Ok(hidden)
}

/// Apply the classification head: pooler (CLS → dense → GELU) →
/// classifier (linear → logits).
pub fn forward_head(
    encoder_last_hidden: &[f32],
    seq_len: usize,
    head: &ClassificationHead,
    cfg: &ModelConfig,
) -> (Vec<f32>, Vec<f32>) {
    let h = cfg.hidden_size;
    assert_eq!(encoder_last_hidden.len(), seq_len * h);

    // Take the CLS token's hidden state — row 0.
    let cls = &encoder_last_hidden[..h];

    // pooled = GELU(cls @ pooler_w.T + pooler_b)
    let mut pooled = vec![0.0_f32; h];
    matmul_at_bias(1, h, h, cls, &head.pooler_w.data, &head.pooler_b, &mut pooled);
    gelu_inplace(&mut pooled);

    // logits = pooled @ classifier_w.T + classifier_b
    let num_labels = cfg.num_labels;
    let mut logits = vec![0.0_f32; num_labels];
    matmul_at_bias(
        1,
        h,
        num_labels,
        &pooled,
        &head.classifier_w.data,
        &head.classifier_b,
        &mut logits,
    );

    (pooled, logits)
}

/// End-to-end forward: tokens → embedding → 24-layer encoder →
/// pooler → classifier → logits.
pub fn forward(
    input_ids: &[u32],
    attention_mask: &[u32],
    weights: &Weights,
) -> Result<ForwardResult, ForwardError> {
    let cfg = &weights.config;
    let head = weights
        .head
        .as_ref()
        .ok_or(ForwardError::MissingWeights("head"))?;
    let seq_len = input_ids.len();

    let embedding = forward_embedding(
        input_ids,
        attention_mask,
        &weights.embeddings,
        cfg.layer_norm_eps,
    );
    let encoder_last_hidden =
        forward_encoder(&embedding, seq_len, attention_mask, weights)?;
    let (pooled, logits) = forward_head(&encoder_last_hidden, seq_len, head, cfg);

    Ok(ForwardResult {
        encoder_last_hidden,
        seq_len,
        pooled,
        logits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::path::PathBuf;

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .expect("workspace root")
    }

    #[derive(Debug, Deserialize)]
    struct FullFixture {
        seq_len: usize,
        hidden_size: usize,
        num_labels: usize,
        input_ids: Vec<u32>,
        attention_mask: Vec<u32>,
        encoder_last_hidden_flat: Vec<f32>,
        pooled_output_flat: Vec<f32>,
        logits: Vec<f32>,
        predicted_label_id: usize,
        predicted_label_name: String,
    }

    fn load_full_fixture() -> Option<FullFixture> {
        let p = workspace_root()
            .join("crates/jouleclaw-deberta/fixtures/full_forward_simple_entail.json");
        let raw = std::fs::read_to_string(p).ok()?;
        serde_json::from_str(&raw).ok()
    }

    fn model_dir() -> Option<PathBuf> {
        let p = workspace_root().join("models/deberta-v3-large-mnli");
        if p.join("model.safetensors").exists() {
            Some(p)
        } else {
            None
        }
    }

    #[test]
    fn ffn_with_residual_preserves_input_when_weights_are_zero() {
        let h = 4;
        let ffn = 8;
        let cfg = {
            let mut c = ModelConfig::default();
            c.hidden_size = h;
            c.intermediate_size = ffn;
            c
        };
        let ffn_weights = FfnWeights {
            intermediate_w: crate::weights::FloatTensor {
                shape: vec![ffn, h],
                data: vec![0.0; ffn * h],
            },
            intermediate_b: vec![0.0; ffn],
            output_w: crate::weights::FloatTensor {
                shape: vec![h, ffn],
                data: vec![0.0; h * ffn],
            },
            output_b: vec![0.0; h],
            output_ln_gamma: vec![1.0; h],
            output_ln_beta: vec![0.0; h],
        };
        let hidden = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = forward_ffn(&hidden, 1, &ffn_weights, &cfg);
        // With zero FFN, residual is just `hidden`, so output is
        // LayerNorm(hidden) → zero-mean unit-variance vector.
        let mean: f32 = out.iter().sum::<f32>() / h as f32;
        let var: f32 = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / h as f32;
        assert!(mean.abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-3);
    }

    /// Headline test: load full model, run the canonical NLI pair
    /// end-to-end, verify logits + prediction against HF.
    #[test]
    fn end_to_end_logits_match_hf_reference() {
        let Some(dir) = model_dir() else {
            { eprintln!("[skip] model not downloaded"); return; };
        };
        let Some(fx) = load_full_fixture() else {
            panic!("fixture missing — run scripts/hf_reference_full_forward.py");
        };

        let inv = crate::ModelInventory::from_dir(&dir).expect("inventory");
        let weights = Weights::load_full(&dir, &inv).expect("load_full");

        let result = forward(&fx.input_ids, &fx.attention_mask, &weights).expect("forward");

        // 1. Encoder last hidden state.
        assert_eq!(result.encoder_last_hidden.len(), fx.seq_len * fx.hidden_size);
        let mut max_abs = 0.0_f32;
        let mut sum_abs = 0.0_f64;
        for (a, b) in result
            .encoder_last_hidden
            .iter()
            .zip(fx.encoder_last_hidden_flat.iter())
        {
            let d = (a - b).abs();
            sum_abs += d as f64;
            if d > max_abs {
                max_abs = d;
            }
        }
        let mean_abs = (sum_abs / result.encoder_last_hidden.len() as f64) as f32;
        eprintln!(
            "encoder_last_hidden diff: max_abs={max_abs:.6} mean_abs={mean_abs:.6}"
        );
        assert!(
            max_abs < 1e-2,
            "encoder output max_abs {max_abs} exceeds tolerance"
        );

        // 2. Pooled output.
        assert_eq!(result.pooled.len(), fx.pooled_output_flat.len());
        let mut p_max = 0.0_f32;
        for (a, b) in result.pooled.iter().zip(fx.pooled_output_flat.iter()) {
            let d = (a - b).abs();
            if d > p_max {
                p_max = d;
            }
        }
        eprintln!("pooled diff: max_abs={p_max:.6}");
        assert!(p_max < 1e-2);

        // 3. Logits.
        assert_eq!(result.logits.len(), fx.num_labels);
        eprintln!(
            "logits: ours={:?} theirs={:?}",
            result.logits, fx.logits
        );
        for (a, b) in result.logits.iter().zip(fx.logits.iter()) {
            let d = (a - b).abs();
            assert!(d < 5e-2, "logit diff {d} too large (ours={a} theirs={b})");
        }

        // 4. Prediction.
        let argmax = result
            .logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            argmax, fx.predicted_label_id,
            "Rust picked label {argmax}, HF picked {}",
            fx.predicted_label_id
        );
        eprintln!(
            "prediction: id={argmax} (HF label '{}')",
            fx.predicted_label_name
        );
    }
}
