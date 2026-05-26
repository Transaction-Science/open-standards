//! Batched forward pass — runs B sequences of length up to
//! L_max through the encoder as a single set of matmul calls.
//!
//! When DeBERTa is asked to entail N (premise, hypothesis) pairs,
//! the prior implementation runs N sequential forwards. With
//! `jouleclaw-diagnose::entail_batch` parallelism that's already
//! 3-4× faster than serial, but each forward still pays the per-
//! call BLAS dispatch overhead. Stacking all N pairs into one
//! padded `[B, L_max, hidden]` forward lets Accelerate amortize
//! dispatch over one large matmul per projection.
//!
//! ## When batching helps
//!
//! Best case: B sequences of similar length (homogeneous batch).
//! The padded work is close to the sum-of-actual-work.
//!
//! Mixed-length batches (e.g., short Wikidata claims + long
//! Wikipedia summaries) lose some of the batching benefit
//! because short sequences pay padded-L_max cost. For our
//! canonical "what is the capital of France?" workload the win
//! is modest; for workloads with N similar-length premises (e.g.,
//! 20 search results all in the ~200-token range) the win is
//! significant.
//!
//! ## What's reused from the single-sequence path
//!
//! - [`crate::embedding::forward_embedding`] — operates on a flat
//!   `[seq_len, hidden]` buffer. Treating `seq_len = B*L_max` lets
//!   us reuse the single-sequence implementation verbatim.
//! - [`crate::forward::forward_ffn`] — same, pure per-row math.
//! - [`crate::tensor_ops::layer_norm_rowwise`] — same.
//!
//! ## What's new
//!
//! - [`forward_attention_batch`] — the inter-token interaction must
//!   stay per-batch-element. Sequence i attends only to its own
//!   L_max tokens; we never cross batches. Loop body iterates
//!   `(b, h)` instead of just `h`.
//! - Padding logic — caller-supplied sequences get padded to
//!   L_max with PAD tokens; the attention_mask zeroes them out.
//! - Pooler extracts row `[b, 0]` (CLS) per batch.

use crate::attention::{build_relative_position, layer_norm_rel_embeddings};
use crate::config::ModelConfig;
use crate::embedding::forward_embedding;
use crate::forward::{forward_ffn, ForwardError, ForwardResult};
use crate::tensor_ops::{gelu_inplace, layer_norm_rowwise, matmul_at, matmul_at_bias, matmul, softmax_rowwise};
use crate::weights::{AttentionWeights, Weights};

/// Run a batch of B sequences through the full pipeline. Returns
/// one [`ForwardResult`] per input sequence, indexed positionally.
///
/// Pads all sequences to the maximum length in the batch. PAD
/// tokens use `pad_id = 0` (matches DeBERTa-v3's tokenizer
/// config). Attention mask zeros out padded positions so they
/// can't influence real tokens.
pub fn forward_batch(
    input_ids_batch: &[Vec<u32>],
    attention_mask_batch: &[Vec<u32>],
    weights: &Weights,
) -> Result<Vec<ForwardResult>, ForwardError> {
    assert_eq!(
        input_ids_batch.len(),
        attention_mask_batch.len(),
        "batch size mismatch between input_ids and attention_mask"
    );
    let batch_size = input_ids_batch.len();
    if batch_size == 0 {
        return Ok(Vec::new());
    }
    let l_max = input_ids_batch.iter().map(|s| s.len()).max().unwrap_or(0);
    if l_max == 0 {
        return Ok(vec![
            ForwardResult {
                encoder_last_hidden: Vec::new(),
                seq_len: 0,
                pooled: Vec::new(),
                logits: Vec::new(),
            };
            batch_size
        ]);
    }

    let cfg = &weights.config;
    let h = cfg.hidden_size;
    let encoder = weights
        .encoder
        .as_ref()
        .ok_or(ForwardError::MissingWeights("encoder"))?;
    let head = weights
        .head
        .as_ref()
        .ok_or(ForwardError::MissingWeights("head"))?;

    // 1. Pad inputs to [B, L_max], flatten to [B*L_max].
    let mut flat_input_ids = vec![0u32; batch_size * l_max];
    let mut flat_attention_mask = vec![0u32; batch_size * l_max];
    let mut per_seq_len = Vec::with_capacity(batch_size);
    for (b, (ids, mask)) in input_ids_batch
        .iter()
        .zip(attention_mask_batch.iter())
        .enumerate()
    {
        assert_eq!(
            ids.len(),
            mask.len(),
            "input_ids/attention_mask length mismatch at batch {b}"
        );
        per_seq_len.push(ids.len());
        for (i, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
            flat_input_ids[b * l_max + i] = id;
            flat_attention_mask[b * l_max + i] = m;
        }
    }

    // 2. Embedding. Reuses the single-sequence implementation —
    // [B*L_max, H] is just "lots of rows" to it.
    let mut hidden = forward_embedding(
        &flat_input_ids,
        &flat_attention_mask,
        &weights.embeddings,
        cfg.layer_norm_eps,
    );

    // 3. Pre-compute encoder-shared resources.
    let relative_pos =
        build_relative_position(l_max, cfg.position_buckets, cfg.max_relative_positions);
    let rel_embeddings = layer_norm_rel_embeddings(encoder, cfg);

    // 4. Encoder loop. Each layer = attention sub-block + FFN
    // sub-block. The attention sub-block is the only piece that
    // sees the batch structure; FFN treats the buffer as flat.
    for layer in &encoder.layers {
        let attn_out = forward_attention_batch(
            &hidden,
            batch_size,
            l_max,
            &flat_attention_mask,
            &relative_pos,
            &rel_embeddings,
            &layer.attention,
            cfg,
        );
        hidden = forward_ffn(&attn_out, batch_size * l_max, &layer.ffn, cfg);
    }

    // 5. Pooler + classifier per batch.
    let mut results = Vec::with_capacity(batch_size);
    let mut pooled_input = vec![0.0_f32; h];
    let mut pooled = vec![0.0_f32; h];
    let mut logits = vec![0.0_f32; cfg.num_labels];
    for b in 0..batch_size {
        // CLS token of batch b is at row [b, 0] of the encoder
        // output — i.e. the slice starting at b*l_max*h, length h.
        let cls = &hidden[b * l_max * h..b * l_max * h + h];
        pooled_input.copy_from_slice(cls);

        matmul_at_bias(
            1,
            h,
            h,
            &pooled_input,
            &head.pooler_w.data,
            &head.pooler_b,
            &mut pooled,
        );
        gelu_inplace(&mut pooled);

        matmul_at_bias(
            1,
            h,
            cfg.num_labels,
            &pooled,
            &head.classifier_w.data,
            &head.classifier_b,
            &mut logits,
        );

        // Extract this batch's encoder hidden state too (only the
        // un-padded portion — callers expect [L_actual, hidden]).
        let l_actual = per_seq_len[b];
        let mut enc_last = Vec::with_capacity(l_actual * h);
        let base = b * l_max * h;
        for i in 0..l_actual {
            enc_last.extend_from_slice(&hidden[base + i * h..base + (i + 1) * h]);
        }

        results.push(ForwardResult {
            encoder_last_hidden: enc_last,
            seq_len: l_actual,
            pooled: pooled.clone(),
            logits: logits.clone(),
        });
    }
    Ok(results)
}

/// Per-layer attention sub-block, batched. Mirrors
/// [`crate::attention::forward_attention`] but the inner loop
/// iterates `(b, h)` so each batch element attends only to its
/// own L_max tokens.
fn forward_attention_batch(
    hidden: &[f32],
    batch_size: usize,
    l_max: usize,
    attention_mask_flat: &[u32], // [B*L_max]
    relative_pos: &[i32],         // [L_max, L_max]
    rel_embeddings_normed: &[f32],
    layer: &AttentionWeights,
    cfg: &ModelConfig,
) -> Vec<f32> {
    let h = cfg.hidden_size;
    let heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim();
    let att_span = cfg.position_buckets;
    let pos_ebd = 2 * att_span;
    let bl = batch_size * l_max;
    assert_eq!(hidden.len(), bl * h);
    assert_eq!(attention_mask_flat.len(), bl);

    // 1. Q/K/V projections — one big matmul each over [B*L_max, H].
    let mut q_all = vec![0.0_f32; bl * h];
    let mut k_all = vec![0.0_f32; bl * h];
    let mut v_all = vec![0.0_f32; bl * h];
    matmul_at_bias(
        bl,
        h,
        h,
        hidden,
        &layer.query_proj_w.data,
        &layer.query_proj_b,
        &mut q_all,
    );
    matmul_at_bias(
        bl,
        h,
        h,
        hidden,
        &layer.key_proj_w.data,
        &layer.key_proj_b,
        &mut k_all,
    );
    matmul_at_bias(
        bl,
        h,
        h,
        hidden,
        &layer.value_proj_w.data,
        &layer.value_proj_b,
        &mut v_all,
    );

    // 2. Per-head reshape. Source layout: [B, L_max, heads, head_dim]
    // (which is just [B*L_max, H] with H = heads*head_dim). Target:
    // [B, heads, L_max, head_dim], flattened to [B*heads, L_max, head_dim].
    let q_bh = split_batch_heads(&q_all, batch_size, l_max, heads, head_dim);
    let k_bh = split_batch_heads(&k_all, batch_size, l_max, heads, head_dim);
    let v_bh = split_batch_heads(&v_all, batch_size, l_max, heads, head_dim);

    // 3. Disentangled position projections — share_att_key=true on
    // DeBERTa-v3, so pos_Q and pos_K both project rel_embeddings
    // through the same Q and K weights respectively.
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
    let pos_q_per_head = split_heads(&pos_q_all, pos_ebd, heads, head_dim);
    let pos_k_per_head = split_heads(&pos_k_all, pos_ebd, heads, head_dim);

    let scale_factor = (1 + cfg.pos_att_type.len()) as f32;
    let scale = (head_dim as f32 * scale_factor).sqrt();

    // 4. Per (b, h): compute attention scores, softmax, mix V.
    let mut context_heads = vec![0.0_f32; batch_size * heads * l_max * head_dim];
    let mut head_scores = vec![0.0_f32; l_max * l_max];
    let mut c2p_full = vec![0.0_f32; l_max * pos_ebd];
    let mut p2c_full = vec![0.0_f32; l_max * pos_ebd];
    let att_span_i = att_span as i32;
    for b in 0..batch_size {
        let mask_b = &attention_mask_flat[b * l_max..(b + 1) * l_max];
        for h_idx in 0..heads {
            let q = batch_head_slice(&q_bh, b, h_idx, batch_size, heads, l_max, head_dim);
            let k = batch_head_slice(&k_bh, b, h_idx, batch_size, heads, l_max, head_dim);
            let v = batch_head_slice(&v_bh, b, h_idx, batch_size, heads, l_max, head_dim);
            let pos_q = head_slice(&pos_q_per_head, h_idx, pos_ebd, head_dim);
            let pos_k = head_slice(&pos_k_per_head, h_idx, pos_ebd, head_dim);

            // c2c: Q @ K.T / scale → [L_max, L_max]
            matmul_at(l_max, head_dim, l_max, q, k, &mut head_scores);
            for v in head_scores.iter_mut() {
                *v /= scale;
            }

            // c2p: Q @ pos_K.T → gather by relative_pos
            matmul_at(l_max, head_dim, pos_ebd, q, pos_k, &mut c2p_full);
            for i in 0..l_max {
                for j in 0..l_max {
                    let rel = relative_pos[i * l_max + j];
                    let idx = (rel + att_span_i).clamp(0, (pos_ebd - 1) as i32) as usize;
                    head_scores[i * l_max + j] += c2p_full[i * pos_ebd + idx] / scale;
                }
            }

            // p2c: K @ pos_Q.T → gather by relative_pos (antisymmetric)
            matmul_at(l_max, head_dim, pos_ebd, k, pos_q, &mut p2c_full);
            for q_idx in 0..l_max {
                for k_idx in 0..l_max {
                    let rel = relative_pos[q_idx * l_max + k_idx];
                    let idx = (rel + att_span_i).clamp(0, (pos_ebd - 1) as i32) as usize;
                    head_scores[q_idx * l_max + k_idx] += p2c_full[k_idx * pos_ebd + idx] / scale;
                }
            }

            // Mask: per-batch attention_mask zeros out padded positions.
            for i in 0..l_max {
                for j in 0..l_max {
                    if mask_b[i] == 0 || mask_b[j] == 0 {
                        head_scores[i * l_max + j] = f32::MIN;
                    }
                }
            }
            softmax_rowwise(l_max, l_max, &mut head_scores);

            // context = probs @ V → [L_max, head_dim]
            let ctx_off = (b * heads + h_idx) * l_max * head_dim;
            let ctx = &mut context_heads[ctx_off..ctx_off + l_max * head_dim];
            matmul(l_max, l_max, head_dim, &head_scores, v, ctx);
        }
    }

    // 5. Concat heads → [B*L_max, H]
    let context_concat = concat_batch_heads(&context_heads, batch_size, l_max, heads, head_dim);

    // 6. Output projection + residual + LayerNorm (per-row, batch-blind).
    let mut output_proj = vec![0.0_f32; bl * h];
    matmul_at_bias(
        bl,
        h,
        h,
        &context_concat,
        &layer.output_dense_w.data,
        &layer.output_dense_b,
        &mut output_proj,
    );
    for i in 0..(bl * h) {
        output_proj[i] += hidden[i];
    }
    let mut final_out = vec![0.0_f32; bl * h];
    layer_norm_rowwise(
        bl,
        h,
        &output_proj,
        &mut final_out,
        &layer.output_ln_gamma,
        &layer.output_ln_beta,
        cfg.layer_norm_eps,
    );
    final_out
}

/// `[B*L_max, H]` row-major to per-(batch, head) layout
/// `[B*heads, L_max, head_dim]`, concatenated.
fn split_batch_heads(
    input: &[f32],
    batch_size: usize,
    l_max: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let h = heads * head_dim;
    let mut out = vec![0.0_f32; batch_size * heads * l_max * head_dim];
    for b in 0..batch_size {
        for h_idx in 0..heads {
            for i in 0..l_max {
                let src_off = (b * l_max + i) * h + h_idx * head_dim;
                let dst_off = (b * heads + h_idx) * l_max * head_dim + i * head_dim;
                out[dst_off..dst_off + head_dim]
                    .copy_from_slice(&input[src_off..src_off + head_dim]);
            }
        }
    }
    out
}

fn concat_batch_heads(
    input: &[f32],
    batch_size: usize,
    l_max: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let h = heads * head_dim;
    let mut out = vec![0.0_f32; batch_size * l_max * h];
    for b in 0..batch_size {
        for h_idx in 0..heads {
            for i in 0..l_max {
                let src_off = (b * heads + h_idx) * l_max * head_dim + i * head_dim;
                let dst_off = (b * l_max + i) * h + h_idx * head_dim;
                out[dst_off..dst_off + head_dim]
                    .copy_from_slice(&input[src_off..src_off + head_dim]);
            }
        }
    }
    out
}

fn batch_head_slice<'a>(
    per_bh: &'a [f32],
    b: usize,
    h_idx: usize,
    _batch_size: usize,
    heads: usize,
    rows: usize,
    head_dim: usize,
) -> &'a [f32] {
    let off = (b * heads + h_idx) * rows * head_dim;
    &per_bh[off..off + rows * head_dim]
}

/// Helpers copy-pasted from `attention::split_heads` for
/// position embeddings (these aren't batched).
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

fn head_slice<'a>(per_head: &'a [f32], h_idx: usize, rows: usize, head_dim: usize) -> &'a [f32] {
    &per_head[h_idx * rows * head_dim..(h_idx + 1) * rows * head_dim]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::ModelInventory;
    use crate::weights::Weights;
    use std::path::PathBuf;

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .expect("workspace root")
    }

    fn model_dir() -> Option<PathBuf> {
        let p = workspace_root().join("models/deberta-v3-large-mnli");
        if p.join("model.safetensors").exists() {
            Some(p)
        } else {
            None
        }
    }

    /// Batched forward on B=1 should match the single-sequence
    /// `forward` byte-for-byte (or to within fp32 noise). This is
    /// the key correctness anchor — if it fails, the batched path
    /// is doing something different from the verified single path.
    #[test]
    fn batch_size_one_matches_single_forward() {
        let Some(dir) = model_dir() else {
            { eprintln!("[skip] model not downloaded"); return; };
        };
        let inv = ModelInventory::from_dir(&dir).expect("inventory");
        let weights = Weights::load_full(&dir, &inv).expect("weights");

        // A representative short NLI input.
        let tokenizer = crate::tokenizer::DebertaTokenizer::from_dir(&dir).unwrap();
        let enc = tokenizer
            .encode_pair("Paris is the capital of France.", "Paris is the capital of France.")
            .unwrap();
        let input_ids = enc.token_ids;
        let attention_mask = enc.attention_mask;

        // Single-sequence path (ground truth).
        let single = crate::forward::forward(&input_ids, &attention_mask, &weights)
            .expect("single forward");

        // Batched path with B=1.
        let batched = forward_batch(
            &[input_ids.clone()],
            &[attention_mask.clone()],
            &weights,
        )
        .expect("batch forward");
        assert_eq!(batched.len(), 1);
        let b = &batched[0];

        // Logits must match within fp32 noise (each path does
        // independent allocs but should produce identical results).
        let mut max_abs = 0.0_f32;
        for (a, c) in single.logits.iter().zip(b.logits.iter()) {
            let d = (a - c).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!(
            "B=1 logits diff vs single: max_abs={max_abs:.6} (single={:?} batched={:?})",
            single.logits, b.logits
        );
        assert!(
            max_abs < 1e-4,
            "B=1 batched logits diverged from single: max_abs={max_abs}"
        );

        // Pooled output should also match.
        let mut p_max = 0.0_f32;
        for (a, c) in single.pooled.iter().zip(b.pooled.iter()) {
            let d = (a - c).abs();
            if d > p_max {
                p_max = d;
            }
        }
        assert!(p_max < 1e-4, "pooled diff {p_max}");
    }

    /// Batched forward of N identical pairs should produce N
    /// identical results, each matching the single-sequence forward.
    #[test]
    fn batched_n_identical_pairs_match_single_n_times() {
        let Some(dir) = model_dir() else {
            { eprintln!("[skip] model not downloaded"); return; };
        };
        let inv = ModelInventory::from_dir(&dir).expect("inventory");
        let weights = Weights::load_full(&dir, &inv).expect("weights");

        let tokenizer = crate::tokenizer::DebertaTokenizer::from_dir(&dir).unwrap();
        let enc = tokenizer
            .encode_pair("Paris is the capital of France.", "Paris is the capital of France.")
            .unwrap();
        let input_ids = enc.token_ids;
        let attention_mask = enc.attention_mask;

        let single = crate::forward::forward(&input_ids, &attention_mask, &weights).unwrap();

        let n = 4;
        let ids_batch = vec![input_ids.clone(); n];
        let mask_batch = vec![attention_mask.clone(); n];
        let batched = forward_batch(&ids_batch, &mask_batch, &weights).unwrap();
        assert_eq!(batched.len(), n);
        for (i, b) in batched.iter().enumerate() {
            let mut max_abs = 0.0_f32;
            for (a, c) in single.logits.iter().zip(b.logits.iter()) {
                let d = (a - c).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            assert!(
                max_abs < 1e-4,
                "batch[{i}] logits diverged from single: max_abs={max_abs}"
            );
        }
    }

    /// Mixed-length batch — padding shouldn't influence the
    /// shorter sequence's output (attention mask zeros out the
    /// padded positions).
    #[test]
    fn mixed_length_batch_does_not_corrupt_short_sequence() {
        let Some(dir) = model_dir() else {
            { eprintln!("[skip] model not downloaded"); return; };
        };
        let inv = ModelInventory::from_dir(&dir).expect("inventory");
        let weights = Weights::load_full(&dir, &inv).expect("weights");
        let tokenizer = crate::tokenizer::DebertaTokenizer::from_dir(&dir).unwrap();

        let short = tokenizer
            .encode_pair("Paris.", "Paris.")
            .unwrap();
        let long = tokenizer
            .encode_pair(
                "Paris is the capital and most populous city of France, with an estimated population of over 2.1 million.",
                "Paris is in France.",
            )
            .unwrap();

        // Run the short pair alone (ground truth).
        let single_short = crate::forward::forward(
            &short.token_ids,
            &short.attention_mask,
            &weights,
        )
        .unwrap();

        // Now put it in a batch with the long pair.
        let batched = forward_batch(
            &[short.token_ids.clone(), long.token_ids.clone()],
            &[short.attention_mask.clone(), long.attention_mask.clone()],
            &weights,
        )
        .unwrap();
        assert_eq!(batched.len(), 2);

        // The short batch[0] should match the single-short output.
        // Padding tolerance is slightly looser because BLAS
        // accumulation order across larger matmuls differs.
        let mut max_abs = 0.0_f32;
        for (a, c) in single_short.logits.iter().zip(batched[0].logits.iter()) {
            let d = (a - c).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!(
            "mixed-length: short[batch] vs short[single] logits max_abs={max_abs:.6}"
        );
        assert!(
            max_abs < 5e-3,
            "padded batch corrupted short sequence: max_abs={max_abs} \
             single={:?} batched={:?}",
            single_short.logits,
            batched[0].logits
        );
    }
}
