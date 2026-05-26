//! Disentangled self-attention forward pass (DeBERTa-v3 §6.4).
//!
//! Replicates `DisentangledSelfAttention.forward` +
//! `DebertaV2SelfOutput.forward` from HF
//! `modeling_deberta_v2.py`. For the v3 NLI checkpoint
//! `share_att_key = true` and `pos_att_type = ["p2c", "c2p"]`, so the
//! pipeline is:
//!
//! 1. Q, K, V projections (with biases).
//! 2. Multi-head reshape to `[heads, L, head_dim]`.
//! 3. `scale = sqrt(head_dim * scale_factor)` where
//!    `scale_factor = 1 + |c2p| + |p2c| = 3`.
//! 4. Content-to-content scores: `Q @ K.T / scale`.
//! 5. Disentangled bias:
//!    - With `share_att_key = true`, project `rel_embeddings`
//!      through the same Q and K projections, reshape to
//!      `[heads, 2*att_span, head_dim]`.
//!    - c2p: `Q @ pos_K.T`, gather along axis -1 by
//!      `clamp(rel_pos + att_span, 0, 2*att_span - 1)`,
//!      divide by scale.
//!    - p2c: `K @ pos_Q.T`, gather along axis -1 by
//!      `clamp(-rel_pos + att_span, 0, 2*att_span - 1)`,
//!      transpose to `[L, L]`, divide by scale.
//! 6. Mask: positions where the attention mask is False get
//!    `f32::MIN` so softmax sends them to ~0.
//! 7. Softmax along the last dim, multiply by V, concat heads.
//! 8. Output projection + residual + LayerNorm.

use crate::config::ModelConfig;
use crate::tensor_ops::{
    layer_norm_rowwise, matmul, matmul_at, matmul_at_bias, softmax_rowwise,
};
use crate::weights::{AttentionWeights, EncoderWeights};

/// Build the bucketed relative-position matrix for a sequence of
/// length `L`. Output is row-major `[L, L]`: `rel[i*L + j]` is the
/// bucketed relative position from query `i` to key `j`.
pub fn build_relative_position(
    seq_len: usize,
    bucket_size: usize,
    max_position: usize,
) -> Vec<i32> {
    let mut out = vec![0_i32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let rel = (i as i32) - (j as i32);
            out[i * seq_len + j] = if bucket_size > 0 && max_position > 0 {
                make_log_bucket_position(rel, bucket_size, max_position)
            } else {
                rel
            };
        }
    }
    out
}

/// Log-spaced bucketing of a single relative-position offset.
/// Mirrors HF's `make_log_bucket_position`:
///
/// ```text
/// mid = bucket_size // 2
/// if |rel_pos| < mid:
///     return rel_pos
/// else:
///     return sign(rel_pos) * (ceil(log(|rel_pos|/mid) /
///                                  log((max_position-1)/mid) *
///                                  (mid-1)) + mid)
/// ```
pub fn make_log_bucket_position(
    rel_pos: i32,
    bucket_size: usize,
    max_position: usize,
) -> i32 {
    let mid = (bucket_size / 2) as i32;
    let sign = rel_pos.signum();
    let abs_rel = rel_pos.abs();
    if abs_rel < mid {
        return rel_pos;
    }
    // abs_pos is set to mid-1 for the "inside" case, but that branch
    // already returned. For the outside case it's abs(rel_pos).
    let abs_pos = abs_rel as f64;
    let log_pos = ((abs_pos / mid as f64).ln()
        / ((max_position - 1) as f64 / mid as f64).ln()
        * (mid - 1) as f64)
        .ceil() as i32
        + mid;
    sign * log_pos
}

/// Run layer-0 (or any layer's) attention sub-block.
///
/// - `hidden`: `[L, H]` row-major
/// - `attention_mask`: `[L]` (1 = real token, 0 = pad)
/// - `relative_pos`: `[L, L]` bucketed
/// - `rel_embeddings_normed`: `[2*att_span, H]` — already
///   LayerNorm'd via the encoder-level LayerNorm.
///
/// Returns `[L, H]` — the input to the FFN sub-block.
pub fn forward_attention(
    hidden: &[f32],
    seq_len: usize,
    attention_mask: &[u32],
    relative_pos: &[i32],
    rel_embeddings_normed: &[f32],
    layer: &AttentionWeights,
    cfg: &ModelConfig,
) -> Vec<f32> {
    let h = cfg.hidden_size;
    let heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim();
    assert_eq!(hidden.len(), seq_len * h);
    assert_eq!(relative_pos.len(), seq_len * seq_len);
    let att_span = cfg.position_buckets;
    assert_eq!(rel_embeddings_normed.len(), 2 * att_span * h);

    // 1. Q, K, V projections — all [L, H] @ W.T + b, weight stored
    // as [out_features=H, in_features=H].
    let mut q_all = vec![0.0_f32; seq_len * h];
    let mut k_all = vec![0.0_f32; seq_len * h];
    let mut v_all = vec![0.0_f32; seq_len * h];
    matmul_at_bias(
        seq_len,
        h,
        h,
        hidden,
        &layer.query_proj_w.data,
        &layer.query_proj_b,
        &mut q_all,
    );
    matmul_at_bias(
        seq_len,
        h,
        h,
        hidden,
        &layer.key_proj_w.data,
        &layer.key_proj_b,
        &mut k_all,
    );
    matmul_at_bias(
        seq_len,
        h,
        h,
        hidden,
        &layer.value_proj_w.data,
        &layer.value_proj_b,
        &mut v_all,
    );

    // 2. share_att_key = true: project rel_embeddings through the
    // same Q and K weights.
    let pos_ebd = 2 * att_span;
    let mut pos_q_all = vec![0.0_f32; pos_ebd * h];
    let mut pos_k_all = vec![0.0_f32; pos_ebd * h];
    matmul_at_bias(
        pos_ebd,
        h,
        h,
        rel_embeddings_normed,
        &layer.query_proj_w.data,
        &layer.query_proj_b,
        &mut pos_q_all,
    );
    matmul_at_bias(
        pos_ebd,
        h,
        h,
        rel_embeddings_normed,
        &layer.key_proj_w.data,
        &layer.key_proj_b,
        &mut pos_k_all,
    );

    // 3. Per-head split. The HF reshape goes
    // `[L, num_heads, head_dim]` then permutes to
    // `[num_heads, L, head_dim]`. In our row-major-flat storage
    // that means: for head `h_idx`, position `i`, the slice is
    // `q_all[i*H + h_idx*head_dim : i*H + (h_idx+1)*head_dim]`.
    // We physically rearrange so each head's matrix is contiguous.
    let q_per_head = split_heads(&q_all, seq_len, heads, head_dim);
    let k_per_head = split_heads(&k_all, seq_len, heads, head_dim);
    let v_per_head = split_heads(&v_all, seq_len, heads, head_dim);
    let pos_q_per_head = split_heads(&pos_q_all, pos_ebd, heads, head_dim);
    let pos_k_per_head = split_heads(&pos_k_all, pos_ebd, heads, head_dim);

    let scale_factor = (1 + cfg.pos_att_type.len()) as f32; // 3 for v3
    let scale = (head_dim as f32 * scale_factor).sqrt();

    // 4-5. Per-head attention scores: c2c + c2p + p2c, all divided
    // by `scale`.
    let mut scores = vec![0.0_f32; heads * seq_len * seq_len];
    for h_idx in 0..heads {
        let q = head_slice(&q_per_head, h_idx, seq_len, head_dim);
        let k = head_slice(&k_per_head, h_idx, seq_len, head_dim);
        let pos_q = head_slice(&pos_q_per_head, h_idx, pos_ebd, head_dim);
        let pos_k = head_slice(&pos_k_per_head, h_idx, pos_ebd, head_dim);

        // c2c: Q[L, d] @ K[L, d].T = [L, L]; divided by scale.
        // We split scaling between K and the per-bias additions to
        // match HF: their c2c divides K by scale (one location);
        // c2p and p2c also divide their results by scale.
        let mut head_scores = vec![0.0_f32; seq_len * seq_len];
        matmul_at(seq_len, head_dim, seq_len, q, k, &mut head_scores);
        for v in head_scores.iter_mut() {
            *v /= scale;
        }

        // c2p: Q[L, d] @ pos_K[pos_ebd, d].T = [L, pos_ebd], gather
        // along axis -1 by clamp(rel_pos + att_span, 0, pos_ebd - 1).
        let mut c2p_full = vec![0.0_f32; seq_len * pos_ebd];
        matmul_at(seq_len, head_dim, pos_ebd, q, pos_k, &mut c2p_full);
        let att_span_i = att_span as i32;
        for i in 0..seq_len {
            for j in 0..seq_len {
                let rel = relative_pos[i * seq_len + j];
                let idx = (rel + att_span_i).clamp(0, (pos_ebd - 1) as i32) as usize;
                head_scores[i * seq_len + j] += c2p_full[i * pos_ebd + idx] / scale;
            }
        }

        // p2c: K[L, d] @ pos_Q[pos_ebd, d].T = [L, pos_ebd].
        //
        // Tracing HF's gather + transpose: the final score
        // contribution at (q_idx, k_idx) is
        //     p2c_att[k_idx, clamp(-rel_pos[k_idx, q_idx] + att_span, ...)]
        // and since rel_pos[i, j] = i - j is antisymmetric,
        //     -rel_pos[k_idx, q_idx] = q_idx - k_idx = rel_pos[q_idx, k_idx],
        // so the gather index reduces to `clamp(rel_pos[q_idx, k_idx]
        // + att_span, ...)` — i.e. the **same** index as c2p. (c2p
        // and p2c differ only in which matrix they gather from.)
        let mut p2c_full = vec![0.0_f32; seq_len * pos_ebd];
        matmul_at(seq_len, head_dim, pos_ebd, k, pos_q, &mut p2c_full);
        for q_idx in 0..seq_len {
            for k_idx in 0..seq_len {
                let rel = relative_pos[q_idx * seq_len + k_idx];
                let idx = (rel + att_span_i).clamp(0, (pos_ebd - 1) as i32) as usize;
                head_scores[q_idx * seq_len + k_idx] += p2c_full[k_idx * pos_ebd + idx] / scale;
            }
        }

        // 6. Mask: if either attention_mask[i] or attention_mask[j]
        // is 0, set score to f32::MIN. (HF's get_attention_mask
        // builds an [L, L] outer product of the input mask.)
        for i in 0..seq_len {
            for j in 0..seq_len {
                if attention_mask[i] == 0 || attention_mask[j] == 0 {
                    head_scores[i * seq_len + j] = f32::MIN;
                }
            }
        }

        // 7. Softmax along last dim.
        softmax_rowwise(seq_len, seq_len, &mut head_scores);

        // Write head_scores into the bigger scores buffer so the
        // softmax stays accessible per-head. We immediately consume
        // it via context = probs @ V below; no need to keep it
        // around, but we do copy for clarity.
        let head_off = h_idx * seq_len * seq_len;
        scores[head_off..head_off + seq_len * seq_len].copy_from_slice(&head_scores);
    }

    // 8. Context = probs @ V per head, then concat back to [L, H].
    let mut context_heads = vec![0.0_f32; heads * seq_len * head_dim];
    for h_idx in 0..heads {
        let probs_off = h_idx * seq_len * seq_len;
        let probs = &scores[probs_off..probs_off + seq_len * seq_len];
        let v = head_slice(&v_per_head, h_idx, seq_len, head_dim);
        let ctx_off = h_idx * seq_len * head_dim;
        let ctx = &mut context_heads[ctx_off..ctx_off + seq_len * head_dim];
        matmul(seq_len, seq_len, head_dim, probs, v, ctx);
    }
    let context_concat = concat_heads(&context_heads, seq_len, heads, head_dim);

    // 9. Output projection: [L, H] @ W.T + b.
    let mut output_proj = vec![0.0_f32; seq_len * h];
    matmul_at_bias(
        seq_len,
        h,
        h,
        &context_concat,
        &layer.output_dense_w.data,
        &layer.output_dense_b,
        &mut output_proj,
    );

    // 10. Residual + LayerNorm.
    for i in 0..(seq_len * h) {
        output_proj[i] += hidden[i];
    }
    let mut final_out = vec![0.0_f32; seq_len * h];
    layer_norm_rowwise(
        seq_len,
        h,
        &output_proj,
        &mut final_out,
        &layer.output_ln_gamma,
        &layer.output_ln_beta,
        cfg.layer_norm_eps,
    );
    final_out
}

/// Apply the encoder-level LayerNorm to the raw rel_embeddings
/// table. This matches HF's `get_rel_embedding()` when
/// `norm_rel_ebd = "layer_norm"`.
pub fn layer_norm_rel_embeddings(encoder: &EncoderWeights, cfg: &ModelConfig) -> Vec<f32> {
    let rows = encoder.rel_embeddings.rows();
    let cols = encoder.rel_embeddings.cols();
    let mut out = vec![0.0_f32; rows * cols];
    layer_norm_rowwise(
        rows,
        cols,
        &encoder.rel_embeddings.data,
        &mut out,
        &encoder.rel_ln_gamma,
        &encoder.rel_ln_beta,
        cfg.layer_norm_eps,
    );
    out
}

/// Convert a `[rows, heads*head_dim]` row-major buffer to a
/// per-head layout where heads are contiguous: `head h, row i,
/// dim d` lives at `head_buffer[h * rows * head_dim + i * head_dim + d]`.
fn split_heads(input: &[f32], rows: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let h = heads * head_dim;
    let mut out = vec![0.0_f32; heads * rows * head_dim];
    for r in 0..rows {
        for h_idx in 0..heads {
            let src = &input[r * h + h_idx * head_dim..r * h + (h_idx + 1) * head_dim];
            let dst_off = h_idx * rows * head_dim + r * head_dim;
            out[dst_off..dst_off + head_dim].copy_from_slice(src);
        }
    }
    out
}

/// Inverse of `split_heads`. Concatenates per-head `[rows, head_dim]`
/// slabs back into a `[rows, heads*head_dim]` row-major buffer.
fn concat_heads(input: &[f32], rows: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let h = heads * head_dim;
    let mut out = vec![0.0_f32; rows * h];
    for h_idx in 0..heads {
        for r in 0..rows {
            let src_off = h_idx * rows * head_dim + r * head_dim;
            let dst_off = r * h + h_idx * head_dim;
            out[dst_off..dst_off + head_dim]
                .copy_from_slice(&input[src_off..src_off + head_dim]);
        }
    }
    out
}

fn head_slice<'a>(
    per_head: &'a [f32],
    h_idx: usize,
    rows: usize,
    head_dim: usize,
) -> &'a [f32] {
    &per_head[h_idx * rows * head_dim..(h_idx + 1) * rows * head_dim]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_identity_when_within_mid() {
        // For seq_len=17, bucket_size=256, max_position=256,
        // mid=128, all |rel| <= 16 < 128 → identity.
        for rel in -16..=16i32 {
            assert_eq!(make_log_bucket_position(rel, 256, 256), rel);
        }
    }

    #[test]
    fn bucket_compresses_far_relations() {
        // mid = 128; at rel = 200, abs is > mid, so log bucket fires.
        let bucketed = make_log_bucket_position(200, 256, 256);
        // sign positive, |bucketed| > mid (= 128)
        assert!(bucketed > 128, "got {bucketed}");
        // Same magnitude flips sign for negative input.
        let neg = make_log_bucket_position(-200, 256, 256);
        assert_eq!(-bucketed, neg);
    }

    #[test]
    fn build_relative_position_is_q_minus_k() {
        // bucket_size = 0 → no bucketing, just identity.
        let r = build_relative_position(3, 0, 0);
        // Expected:
        //   i=0 j=0,1,2 → 0,-1,-2
        //   i=1 j=0,1,2 → 1, 0,-1
        //   i=2 j=0,1,2 → 2, 1, 0
        assert_eq!(r, vec![0, -1, -2, 1, 0, -1, 2, 1, 0]);
    }

    fn workspace_root() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .expect("workspace root")
    }

    #[derive(Debug, serde::Deserialize)]
    struct AttnFixture {
        seq_len: usize,
        attention_mask: Vec<u32>,
        relative_pos_flat: Vec<i32>,
        relative_pos_shape: Vec<usize>,
        rel_embeddings_normed_flat: Vec<f32>,
        rel_embeddings_shape: Vec<usize>,
        embedding_output_flat: Vec<f32>,
        layer0_attn_output_flat: Vec<f32>,
    }

    fn load_attn_fixture() -> Option<AttnFixture> {
        let p = workspace_root()
            .join("crates/jouleclaw-deberta/fixtures/layer0_attention.json");
        if !p.exists() {
            return None;
        }
        let raw = std::fs::read_to_string(&p).ok()?;
        serde_json::from_str(&raw).ok()
    }

    fn model_dir() -> Option<std::path::PathBuf> {
        let p = workspace_root().join("models/deberta-v3-large-mnli");
        if p.join("model.safetensors").exists() {
            Some(p)
        } else {
            None
        }
    }

    /// Confirm our Rust-built relative-position matrix matches the
    /// one HF produced for the same (L, bucket_size, max_position).
    /// Bucketing is identity for L<=128 so this is mostly a
    /// build_relative_position correctness check, but it's the
    /// foundation the disentangled bias rests on.
    #[test]
    fn relative_position_matches_hf_reference() {
        let Some(fx) = load_attn_fixture() else {
            panic!("fixture missing — run scripts/hf_reference_layer0_attention.py");
        };
        assert_eq!(fx.relative_pos_shape, vec![fx.seq_len, fx.seq_len]);
        let mine = build_relative_position(fx.seq_len, 256, 256);
        assert_eq!(mine, fx.relative_pos_flat);
    }

    /// Confirm our LayerNorm of the rel_embeddings table matches
    /// HF's. This is what gets fed into the per-layer attention's
    /// disentangled bias computation.
    #[test]
    fn rel_embeddings_normed_matches_hf_reference() {
        let Some(dir) = model_dir() else {
            { eprintln!("[skip] model not downloaded"); return; };
        };
        let Some(fx) = load_attn_fixture() else {
            { eprintln!("[skip] fixture missing"); return; };
        };
        let inv = crate::ModelInventory::from_dir(&dir).expect("inventory");
        let weights = crate::Weights::load_embeddings_and_layer0_attention(&dir, &inv)
            .expect("weights");
        let enc = weights.encoder.as_ref().expect("encoder loaded");
        let normed = layer_norm_rel_embeddings(enc, &weights.config);

        assert_eq!(fx.rel_embeddings_shape, vec![512, 1024]);
        assert_eq!(normed.len(), fx.rel_embeddings_normed_flat.len());

        let mut max_abs = 0.0_f32;
        let mut sum_abs = 0.0_f64;
        for (a, b) in normed.iter().zip(fx.rel_embeddings_normed_flat.iter()) {
            let d = (a - b).abs();
            sum_abs += d as f64;
            if d > max_abs {
                max_abs = d;
            }
        }
        let mean_abs = (sum_abs / normed.len() as f64) as f32;
        eprintln!("rel_embeddings normed diff: max_abs={max_abs:.6} mean_abs={mean_abs:.6}");
        assert!(max_abs < 1e-3, "rel_embeddings LN diverges: max_abs={max_abs}");
    }

    /// Headline test: load weights, run the full layer-0 attention
    /// sub-block on the verified embedding output, compare against
    /// HF's reference attention output.
    #[test]
    fn forward_attention_matches_hf_reference_within_fp16_tolerance() {
        let Some(dir) = model_dir() else {
            { eprintln!("[skip] model not downloaded"); return; };
        };
        let Some(fx) = load_attn_fixture() else {
            { eprintln!("[skip] fixture missing"); return; };
        };

        let inv = crate::ModelInventory::from_dir(&dir).expect("inventory");
        let weights = crate::Weights::load_embeddings_and_layer0_attention(&dir, &inv)
            .expect("weights");
        let enc = weights.encoder.as_ref().expect("encoder loaded");
        let rel_embeddings = layer_norm_rel_embeddings(enc, &weights.config);

        let relative_pos = build_relative_position(fx.seq_len, 256, 256);
        let layer0 = &enc.layers[0].attention;

        let out = forward_attention(
            &fx.embedding_output_flat,
            fx.seq_len,
            &fx.attention_mask,
            &relative_pos,
            &rel_embeddings,
            layer0,
            &weights.config,
        );

        assert_eq!(out.len(), fx.layer0_attn_output_flat.len());

        let mut max_abs = 0.0_f32;
        let mut sum_abs = 0.0_f64;
        let mut max_idx = 0_usize;
        for (i, (a, b)) in out.iter().zip(fx.layer0_attn_output_flat.iter()).enumerate() {
            let d = (a - b).abs();
            sum_abs += d as f64;
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
        }
        let mean_abs = (sum_abs / out.len() as f64) as f32;
        eprintln!(
            "layer-0 attention diff vs HF reference: max_abs={max_abs:.6} \
             mean_abs={mean_abs:.6} worst at idx {max_idx} \
             (ours={} theirs={})",
            out[max_idx], fx.layer0_attn_output_flat[max_idx]
        );

        assert!(
            max_abs < 5e-3,
            "max abs diff {max_abs} exceeds 5e-3 tolerance"
        );
        assert!(
            mean_abs < 1e-4,
            "mean abs diff {mean_abs} exceeds 1e-4 tolerance"
        );
    }

    #[test]
    fn split_concat_heads_round_trip() {
        // [rows=2, heads*head_dim=4] with heads=2, head_dim=2.
        let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let split = split_heads(&input, 2, 2, 2);
        // head 0 row 0: (1,2); head 0 row 1: (5,6)
        // head 1 row 0: (3,4); head 1 row 1: (7,8)
        assert_eq!(split, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
        let back = concat_heads(&split, 2, 2, 2);
        assert_eq!(back, input);
    }
}
