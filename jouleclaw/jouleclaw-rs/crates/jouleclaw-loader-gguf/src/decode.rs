//! Build a forward-pass graph for one decode step using the KV cache.
//!
//! Unlike `build_llama_graph`, which builds a fresh-start prefill graph,
//! `build_decode_step_graph` builds a graph that:
//! 1. Takes new tokens of shape `[new_seq]`
//! 2. Takes per-layer cached K and V tensors (each `[n_heads,
//!    cached_seq, d_head]`) as inputs
//! 3. Computes new Q/K/V projections for just the new tokens
//! 4. Concats new K and V onto cached K and V to get full-context K/V
//! 5. Runs offset-causal attention against the full-context K and V
//! 6. Outputs logits for only the new tokens, plus the updated K and V
//!    for each layer (so the host can stash them in the cache for the
//!    next step)
//!
//! The graph is rebuilt for each step (since shapes change as the cached
//! seq grows). A future engine optimization will memoize unchanged
//! subgraphs, but for now correctness is the priority.

use crate::kv_cache::KvCache;
use crate::llama::{LlamaConfig, LoadError};
use crate::{tensor_from_gguf, GgmlType, GgufModel};
use jouleclaw_core::graph::{Graph, GraphBuilder, NodeId};
use jouleclaw_core::op::{ActivationKind, NormKind};
use jouleclaw_core::tensor::{Dtype, TensorMeta, TensorRef};

/// A graph that performs one decode step using the KV cache, plus
/// metadata so the host knows which output names to read.
pub struct DecodeStepGraph {
    pub graph: Graph,
    pub config: LlamaConfig,
    /// Sequence length of the new tokens (i.e., the input slice).
    pub new_seq: usize,
    /// Sequence length of the cache *before* this step. New full seq is
    /// `cached_seq + new_seq`.
    pub cached_seq: usize,
    /// Output name for the layer-`i` updated K tensor.
    pub k_output_names: Vec<String>,
    /// Output name for the layer-`i` updated V tensor.
    pub v_output_names: Vec<String>,
    /// Output name for the final logits.
    pub logits_output_name: String,
    /// Input names for the layer-`i` previous K (when `cached_seq > 0`).
    pub k_input_names: Vec<String>,
    /// Input names for the layer-`i` previous V.
    pub v_input_names: Vec<String>,
}

/// Build a decode-step graph.
///
/// `new_seq` is the number of new tokens this step is processing. For
/// pure incremental decoding it's typically 1; for prefill in chunks it
/// can be larger.
///
/// `cache` provides the current cached_seq and (when `cached_seq > 0`)
/// the K/V tensors that the host will bind at execute time.
pub fn build_decode_step_graph(
    model: &GgufModel,
    cache: &KvCache,
    new_seq: usize,
) -> Result<DecodeStepGraph, LoadError> {
    let config = LlamaConfig::from_metadata(model)?;
    let cached_seq = cache.current_seq;
    let total_seq = cached_seq + new_seq;

    let mut g = GraphBuilder::new();

    // Input: new token IDs.
    let token_ids = g.input("token_ids",
        TensorMeta::new(Dtype::I32, &[new_seq]));

    // Per-layer cached K/V inputs (when present).
    let mut k_input_names = Vec::with_capacity(config.block_count);
    let mut v_input_names = Vec::with_capacity(config.block_count);
    let mut k_input_nodes: Vec<Option<NodeId>> = Vec::with_capacity(config.block_count);
    let mut v_input_nodes: Vec<Option<NodeId>> = Vec::with_capacity(config.block_count);

    let n_heads = config.head_count.max(1);
    let n_heads_kv = config.head_count_kv.max(1);
    let d_head = config.embedding_length / n_heads;
    let group_size = n_heads / n_heads_kv;
    if group_size * n_heads_kv != n_heads {
        return Err(LoadError::UnsupportedArchitecture(format!(
            "head_count ({}) not divisible by head_count_kv ({})",
            n_heads, n_heads_kv)));
    }

    for layer in 0..config.block_count {
        let k_name = format!("kv_in_k_{}", layer);
        let v_name = format!("kv_in_v_{}", layer);
        if cached_seq > 0 {
            // Cache stores K/V at the *KV head count* shape (smaller under GQA).
            let k_node = g.input(&k_name,
                TensorMeta::new(Dtype::F32, &[n_heads_kv, cached_seq, d_head]));
            let v_node = g.input(&v_name,
                TensorMeta::new(Dtype::F32, &[n_heads_kv, cached_seq, d_head]));
            k_input_nodes.push(Some(k_node));
            v_input_nodes.push(Some(v_node));
        } else {
            k_input_nodes.push(None);
            v_input_nodes.push(None);
        }
        k_input_names.push(k_name);
        v_input_names.push(v_name);
    }

    // Token embedding + (tied) LM head — Q2_0-aware. For Bonsai the
    // 151669×2048 table stays packed; the lookup decodes only the
    // requested rows and the tied LM head reuses the same packed
    // constant. Non-Q2_0 (TinyLlama) falls through to the prior
    // dense path bit-identically.
    use crate::llama::{embed_weight, lookup_w, wmm, Weight as LWeight};
    let te_info = model.tensor_by_name("token_embd.weight")
        .ok_or_else(|| LoadError::MissingTensor("token_embd.weight".into()))?;
    let token_embd_w: Option<LWeight> = if matches!(
        te_info.dtype, crate::GgmlType::Q2_0 | crate::GgmlType::Q1_0,
    ) {
        Some(embed_weight(&mut g, model, "token_embd.weight")?)
    } else {
        None
    };
    let mut x = match &token_embd_w {
        Some(w) => lookup_w(&mut g, token_ids, w),
        None => {
            let te = embed_constant(&mut g, model, "token_embd.weight")?;
            g.lookup(token_ids, te)
        }
    };  // [new_seq, d_model]

    // Per-layer K/V outputs we'll need to collect.
    let mut k_output_nodes = Vec::with_capacity(config.block_count);
    let mut v_output_nodes = Vec::with_capacity(config.block_count);
    let mut k_output_names = Vec::with_capacity(config.block_count);
    let mut v_output_names = Vec::with_capacity(config.block_count);

    // Transformer blocks.
    for layer in 0..config.block_count {
        let result = build_decode_block(
            &mut g, model, &config, layer, x,
            new_seq, cached_seq, total_seq,
            n_heads, n_heads_kv, d_head,
            k_input_nodes[layer], v_input_nodes[layer],
        )?;
        x = result.x_out;
        k_output_nodes.push(result.k_full);
        v_output_nodes.push(result.v_full);

        let k_out_name = format!("kv_out_k_{}", layer);
        let v_out_name = format!("kv_out_v_{}", layer);
        g.output(&k_out_name, result.k_full);
        g.output(&v_out_name, result.v_full);
        k_output_names.push(k_out_name);
        v_output_names.push(v_out_name);
    }

    // Final norm + lm_head.
    let output_norm = embed_constant(&mut g, model, "output_norm.weight")?;
    let xn = g.norm(x, output_norm, NormKind::Rms, config.rms_eps);

    let logits = if model.tensor_by_name("output.weight").is_some() {
        let w = embed_weight(&mut g, model, "output.weight")?;
        wmm(&mut g, xn, &w)
    } else {
        match &token_embd_w {
            Some(w) => wmm(&mut g, xn, w),
            None => {
                let te = embed_constant(&mut g, model, "token_embd.weight")?;
                g.matmul_bt(xn, te)
            }
        }
    };
    g.output("logits", logits);

    Ok(DecodeStepGraph {
        graph: g.build(),
        config,
        new_seq,
        cached_seq,
        k_output_names,
        v_output_names,
        logits_output_name: "logits".into(),
        k_input_names,
        v_input_names,
    })
}

/// Result of building one decode-block subgraph.
struct DecodeBlockResult {
    /// The block's residual output (input to next block).
    x_out: NodeId,
    /// Layer's full-context K (cached + new), shape `[n_heads, total_seq, d_head]`.
    k_full: NodeId,
    /// Layer's full-context V, same shape.
    v_full: NodeId,
}

#[allow(clippy::too_many_arguments)]
fn build_decode_block(
    g: &mut GraphBuilder,
    model: &GgufModel,
    config: &LlamaConfig,
    layer: usize,
    x: NodeId,
    new_seq: usize,
    cached_seq: usize,
    total_seq: usize,
    n_heads: usize,
    n_heads_kv: usize,
    d_head: usize,
    k_prev: Option<NodeId>,
    v_prev: Option<NodeId>,
) -> Result<DecodeBlockResult, LoadError> {
    let _ = total_seq;
    let group_size = n_heads / n_heads_kv;

    // Attention norm + projections.
    let attn_norm = embed_constant(g, model, &format!("blk.{}.attn_norm.weight", layer))?;
    let xn = g.norm(x, attn_norm, NormKind::Rms, config.rms_eps);

    use crate::llama::{embed_weight, wmm};
    let w_q = embed_weight(g, model, &format!("blk.{}.attn_q.weight", layer))?;
    let w_k = embed_weight(g, model, &format!("blk.{}.attn_k.weight", layer))?;
    let w_v = embed_weight(g, model, &format!("blk.{}.attn_v.weight", layer))?;
    let w_o = embed_weight(g, model, &format!("blk.{}.attn_output.weight", layer))?;

    // Project the new tokens. Note Q gets full n_heads*d_head columns,
    // K and V get only n_heads_kv*d_head columns. `wmm` routes Q2_0
    // weights through the ternary kernel; dense weights stay on the
    // existing f32 `matmul_bt` path bit-identically.
    let q_new = wmm(g, xn, &w_q);  // [new_seq, n_heads * d_head]
    let k_new = wmm(g, xn, &w_k);  // [new_seq, n_heads_kv * d_head]
    let v_new = wmm(g, xn, &w_v);  // [new_seq, n_heads_kv * d_head]

    // Multi-head split. Q uses n_heads; K, V use n_heads_kv.
    let q_h = {
        let r = g.reshape(q_new, &[new_seq, n_heads, d_head]);
        g.transpose(r, &[1, 0, 2])  // [n_heads, new_seq, d_head]
    };
    let k_h = {
        let r = g.reshape(k_new, &[new_seq, n_heads_kv, d_head]);
        g.transpose(r, &[1, 0, 2])  // [n_heads_kv, new_seq, d_head]
    };
    let v_h = {
        let r = g.reshape(v_new, &[new_seq, n_heads_kv, d_head]);
        g.transpose(r, &[1, 0, 2])  // [n_heads_kv, new_seq, d_head]
    };

    // qwen3 QK-RMSNorm (per-head, pre-RoPE). See the in-place block for
    // the rationale; no-op for llama (no attn_q_norm tensor).
    let (q_h, k_h) = if config.qk_norm {
        let q_norm = embed_constant(g, model, &format!("blk.{}.attn_q_norm.weight", layer))?;
        let k_norm = embed_constant(g, model, &format!("blk.{}.attn_k_norm.weight", layer))?;
        let q_n = g.norm(q_h, q_norm, NormKind::Rms, config.rms_eps);
        let k_n = g.norm(k_h, k_norm, NormKind::Rms, config.rms_eps);
        (q_n, k_n)
    } else {
        (q_h, k_h)
    };

    // RoPE on the new Q and K, with position_offset = cached_seq.
    let q_rot = g.rope(q_h, config.rope_base, cached_seq as u32);
    let k_rot = g.rope(k_h, config.rope_base, cached_seq as u32);

    // Concat new K/V onto cached K/V along the seq axis (axis 1).
    // Cache is at n_heads_kv head count — same as new K/V projections.
    let k_full_kv = match k_prev {
        Some(prev) => g.concat(prev, k_rot, 1),
        None => k_rot,
    };
    let v_full_kv = match v_prev {
        Some(prev) => g.concat(prev, v_h, 1),
        None => v_h,
    };

    // The cache STAYS at n_heads_kv shape — that's the GQA benefit.
    // We return these as the cache outputs.
    let k_cache_out = k_full_kv;
    let v_cache_out = v_full_kv;

    // For attention, broadcast K and V to n_heads via repeat-interleave.
    let total_kv_seq = cached_seq + new_seq;
    let k_full = if group_size == 1 { k_full_kv } else {
        let r = g.reshape(k_full_kv, &[n_heads_kv, 1, total_kv_seq, d_head]);
        let rep = g.repeat(r, 1, group_size);
        g.reshape(rep, &[n_heads, total_kv_seq, d_head])
    };
    let v_full = if group_size == 1 { v_full_kv } else {
        let r = g.reshape(v_full_kv, &[n_heads_kv, 1, total_kv_seq, d_head]);
        let rep = g.repeat(r, 1, group_size);
        g.reshape(rep, &[n_heads, total_kv_seq, d_head])
    };

    // Scaled QK^T: q_rot is [n_heads, new_seq, d_head], k_full is
    // [n_heads, total_seq, d_head]. matmul_bt gives [n_heads, new_seq, total_seq].
    let inv_sqrt_d = 1.0 / (d_head as f32).sqrt();
    let scores = g.matmul_bt_scaled(q_rot, k_full, inv_sqrt_d);
    let probs = g.softmax_causal_offset(scores, -1, cached_seq as i32);
    let ctx_h = g.matmul(probs, v_full);

    let ctx = {
        let t = g.transpose(ctx_h, &[1, 0, 2]);  // [new_seq, n_heads, d_head]
        g.reshape(t, &[new_seq, n_heads * d_head])
    };
    let y = wmm(g, ctx, &w_o);
    let after_attn = g.add(x, y);

    // FFN (gated SiLU).
    let ffn_norm = embed_constant(g, model, &format!("blk.{}.ffn_norm.weight", layer))?;
    let xn2 = g.norm(after_attn, ffn_norm, NormKind::Rms, config.rms_eps);

    let w_gate = embed_weight(g, model, &format!("blk.{}.ffn_gate.weight", layer))?;
    let w_up = embed_weight(g, model, &format!("blk.{}.ffn_up.weight", layer))?;
    let w_down = embed_weight(g, model, &format!("blk.{}.ffn_down.weight", layer))?;

    let gate = wmm(g, xn2, &w_gate);
    let gate = g.activation(gate, ActivationKind::SiLU);
    let up = wmm(g, xn2, &w_up);
    let hidden = g.mul(gate, up);
    let ffn_out = wmm(g, hidden, &w_down);
    let x_out = g.add(after_attn, ffn_out);

    Ok(DecodeBlockResult { x_out, k_full: k_cache_out, v_full: v_cache_out })
}

fn embed_constant(
    g: &mut GraphBuilder,
    model: &GgufModel,
    name: &str,
) -> Result<NodeId, LoadError> {
    let info = model.tensor_by_name(name)
        .ok_or_else(|| LoadError::MissingTensor(name.to_string()))?;
    if !matches!(info.dtype,
        GgmlType::F32 | GgmlType::F16 | GgmlType::Q8_0 | GgmlType::Q4_K
        | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::I2_S | GgmlType::Q1_0
        | GgmlType::Q2_0 | GgmlType::STQ1_0)
    {
        return Err(LoadError::UnsupportedTensorType { name: name.into(), dtype: info.dtype });
    }
    let tensor = tensor_from_gguf(model, info)
        .map_err(|e| LoadError::TensorExtraction { name: name.into(), source: e })?;
    let tref = TensorRef { meta: tensor.meta, storage: tensor.storage };
    Ok(g.constant(tref))
}

/// Public re-export of `embed_constant` for use by the in-place cache module.
pub fn embed_constant_pub(
    g: &mut GraphBuilder,
    model: &GgufModel,
    name: &str,
) -> Result<NodeId, LoadError> {
    embed_constant(g, model, name)
}

/// The LM-head projection weight. Models with an untied head store it
/// as `output.weight`; tied-embedding models (qwen3 Bonsai, many small
/// llamas) omit it and reuse `token_embd.weight` — both are
/// `[vocab, d_model]`, so `matmul_bt` against either is correct.
pub fn lm_head_constant(
    g: &mut GraphBuilder,
    model: &GgufModel,
) -> Result<NodeId, LoadError> {
    let name = if model.tensor_by_name("output.weight").is_some() {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    embed_constant(g, model, name)
}

/// One in-place decode block's output. The K and V tensors here are the
/// preallocated buffers AFTER scattering the new step's K/V at the current
/// position.
pub struct InPlaceDecodeBlockResult {
    pub x_out: NodeId,
    pub k_buf_new: NodeId,
    pub v_buf_new: NodeId,
    /// For LFM2 recurrent (shortconv) layers: the updated rolling-
    /// window state (`[taps - 1, d]` f32). `None` for attention
    /// layers. When `Some`, the caller is expected to wire this as
    /// a graph output and feed it back as `state_in` next step.
    pub shortconv_state_out: Option<NodeId>,
}

/// Build one transformer block of the in-place decode-step graph.
///
/// Differences from `build_decode_block`:
/// - Takes the layer's preallocated K and V buffer nodes (shape
///   `[n_heads_kv, max_seq, d_head]`) instead of optional prev tensors.
/// - Scatters the new K and V into the buffer at the current `cached_seq`
///   position instead of concatenating.
/// - Attention runs over the full buffer with offset-causal softmax doing
///   the masking for positions beyond `cached_seq + new_seq`.
#[allow(clippy::too_many_arguments)]
pub fn build_decode_step_graph_inplace_block(
    g: &mut GraphBuilder,
    model: &GgufModel,
    config: &LlamaConfig,
    layer: usize,
    x: NodeId,
    new_seq: usize,
    cached_seq: usize,
    _max_seq: usize,
    n_heads: usize,
    n_heads_kv: usize,
    d_head: usize,
    group_size: usize,
    k_buf: NodeId,
    v_buf: NodeId,
) -> Result<InPlaceDecodeBlockResult, LoadError> {
    use crate::llama::{embed_weight, wmm};
    let attn_norm = embed_constant(g, model, &format!("blk.{}.attn_norm.weight", layer))?;
    let xn = g.norm(x, attn_norm, NormKind::Rms, config.rms_eps);

    // Weight matmuls route through `wmm` — ternary kernel for Q2_0
    // (Bonsai), bit-identical dense `matmul_bt` for everything else
    // (TinyLlama). This is the change that brings the *generation*
    // path up to par with the prefill path; previously these went
    // through the full f32 dequant.
    let w_q = embed_weight(g, model, &format!("blk.{}.attn_q.weight", layer))?;
    let w_k = embed_weight(g, model, &format!("blk.{}.attn_k.weight", layer))?;
    let w_v = embed_weight(g, model, &format!("blk.{}.attn_v.weight", layer))?;
    let w_o = embed_weight(g, model, &format!("blk.{}.attn_output.weight", layer))?;

    let q_new = wmm(g, xn, &w_q);
    let k_new = wmm(g, xn, &w_k);
    let v_new = wmm(g, xn, &w_v);

    let q_h = {
        let r = g.reshape(q_new, &[new_seq, n_heads, d_head]);
        g.transpose(r, &[1, 0, 2])
    };
    let k_h = {
        let r = g.reshape(k_new, &[new_seq, n_heads_kv, d_head]);
        g.transpose(r, &[1, 0, 2])
    };
    let v_h = {
        let r = g.reshape(v_new, &[new_seq, n_heads_kv, d_head]);
        g.transpose(r, &[1, 0, 2])
    };

    // qwen3 QK-RMSNorm (per-head, pre-RoPE). Llama (no q_norm tensor)
    // skips this and behaves as before.
    let (q_h, k_h) = if config.qk_norm {
        let q_norm = embed_constant(g, model, &format!("blk.{}.attn_q_norm.weight", layer))?;
        let k_norm = embed_constant(g, model, &format!("blk.{}.attn_k_norm.weight", layer))?;
        let q_n = g.norm(q_h, q_norm, NormKind::Rms, config.rms_eps);
        let k_n = g.norm(k_h, k_norm, NormKind::Rms, config.rms_eps);
        (q_n, k_n)
    } else {
        (q_h, k_h)
    };

    let q_rot = g.rope(q_h, config.rope_base, cached_seq as u32);
    let k_rot = g.rope(k_h, config.rope_base, cached_seq as u32);

    // Scatter the new K and V into the preallocated buffers at cached_seq.
    // `scatter_inplace` hints to the executor that it may reuse the buffer's
    // storage as the output's storage, avoiding the O(max_seq * d_head)
    // dst→output memcpy. Safe here because k_buf and v_buf are only consumed
    // by these scatters (no other graph node references them after this).
    let k_buf_new = g.scatter_inplace(k_buf, k_rot, 1, cached_seq);
    let v_buf_new = g.scatter_inplace(v_buf, v_h, 1, cached_seq);

    // Slice K and V to the live region [0..valid_seq] along the seq axis.
    // This is the perf win: subsequent attention work scales with valid_seq
    // (= cached_seq + new_seq), not max_seq. The Slice copy itself is
    // O(valid_seq * d_head) per layer; for typical decode (valid_seq «
    // max_seq) this is much smaller than running attention over the full
    // buffer.
    let valid_seq = cached_seq + new_seq;
    let k_live = g.slice(k_buf_new, 1, 0, valid_seq);
    let v_live = g.slice(v_buf_new, 1, 0, valid_seq);

    // For attention, broadcast K and V to n_heads via repeat-interleave.
    let k_full = if group_size == 1 { k_live } else {
        let r = g.reshape(k_live, &[n_heads_kv, 1, valid_seq, d_head]);
        let rep = g.repeat(r, 1, group_size);
        g.reshape(rep, &[n_heads, valid_seq, d_head])
    };
    let v_full = if group_size == 1 { v_live } else {
        let r = g.reshape(v_live, &[n_heads_kv, 1, valid_seq, d_head]);
        let rep = g.repeat(r, 1, group_size);
        g.reshape(rep, &[n_heads, valid_seq, d_head])
    };

    // Scaled QK^T over the live region. The offset-causal mask still
    // applies for in-step causality (query i can only see keys 0..=i + cached_seq);
    // positions beyond valid_seq don't exist in K_full anymore.
    let inv_sqrt_d = 1.0 / (d_head as f32).sqrt();
    let scores = g.matmul_bt_scaled(q_rot, k_full, inv_sqrt_d);
    let probs = g.softmax_causal_offset(scores, -1, cached_seq as i32);
    let ctx_h = g.matmul(probs, v_full);

    let ctx = {
        let t = g.transpose(ctx_h, &[1, 0, 2]);
        g.reshape(t, &[new_seq, n_heads * d_head])
    };
    let y = wmm(g, ctx, &w_o);
    let after_attn = g.add(x, y);

    let ffn_norm = embed_constant(g, model, &format!("blk.{}.ffn_norm.weight", layer))?;
    let xn2 = g.norm(after_attn, ffn_norm, NormKind::Rms, config.rms_eps);

    let w_gate = embed_weight(g, model, &format!("blk.{}.ffn_gate.weight", layer))?;
    let w_up = embed_weight(g, model, &format!("blk.{}.ffn_up.weight", layer))?;
    let w_down = embed_weight(g, model, &format!("blk.{}.ffn_down.weight", layer))?;

    let gate = wmm(g, xn2, &w_gate);
    let gate = g.activation(gate, ActivationKind::SiLU);
    let up = wmm(g, xn2, &w_up);
    let hidden = g.mul(gate, up);
    let ffn_out = wmm(g, hidden, &w_down);
    let x_out = g.add(after_attn, ffn_out);

    Ok(InPlaceDecodeBlockResult { x_out, k_buf_new, v_buf_new, shortconv_state_out: None })
}

/// **Constant-topology** in-place decode block. Identical math to
/// [`build_decode_step_graph_inplace_block`] but the only thing that
/// varies token-to-token — `cached_seq` — enters as a runtime input
/// `kv_pos` (I32 `[1]`) instead of a build-time constant, and there is
/// **no `valid_seq` slice**: attention runs over the full
/// `max_seq`-sized K/V buffer with the dynamic causal mask zeroing the
/// not-yet-written tail (those positions are `> q + cached_seq`, so
/// causality already excludes them — the result is bit-identical to
/// the sliced version).
///
/// Because nothing shape-dependent on `cached_seq` remains, the whole
/// graph has fixed topology for a given `(model, new_seq, max_seq)` —
/// so it can be compiled **once** and reused for every streaming
/// decode step. Trade-off: attention is `O(max_seq)` per step instead
/// of `O(valid_seq)`; the win is eliminating the per-token
/// build+compile, which dominates once generation is long enough that
/// the same compiled graph is reused many times.
#[allow(clippy::too_many_arguments)]
pub fn build_decode_step_graph_inplace_const_block(
    g: &mut GraphBuilder,
    model: &GgufModel,
    config: &LlamaConfig,
    layer: usize,
    x: NodeId,
    new_seq: usize,
    kv_pos: NodeId,
    max_seq: usize,
    n_heads: usize,
    n_heads_kv: usize,
    d_head: usize,
    group_size: usize,
    k_buf: NodeId,
    v_buf: NodeId,
    shortconv_state_in: Option<NodeId>,
) -> Result<InPlaceDecodeBlockResult, LoadError> {
    use crate::llama::{embed_weight, wmm};
    let attn_norm = embed_constant(g, model, &format!("blk.{}.attn_norm.weight", layer))?;

    // LFM2 hybrid arch: some layers are shortconv (recurrent) rather
    // than attention. The shortconv path doesn't touch K/V buffers, so
    // we pass them through unchanged — keeping the graph's
    // input/output names constant across layer types (a const-topology
    // requirement). Both attention and shortconv branches fall through
    // into the shared FFN, matching the legacy prefill `build_block`.
    let is_lfm2_recurrent = config.arch == "lfm2"
        && config.per_layer_head_count_kv.get(layer).copied() == Some(0);
    let (after_attn, k_buf_new, v_buf_new, shortconv_state_out) = if is_lfm2_recurrent {
        let state_in = shortconv_state_in.ok_or_else(|| {
            LoadError::UnsupportedArchitecture(format!(
                "LFM2 layer {layer} is shortconv but no state_in node was \
                 supplied to the block builder — the parent must allocate \
                 a per-layer state input"))
        })?;
        let (x_out, state_out) = build_lfm2_shortconv_block_streaming(
            g, model, config, layer, x, attn_norm, new_seq, state_in)?;
        (x_out, k_buf, v_buf, Some(state_out))
    } else {
        let (after_attn, k_buf_new, v_buf_new) =
            build_decode_attn_block(
                g, model, config, layer, x, attn_norm,
                new_seq, kv_pos, max_seq,
                n_heads, n_heads_kv, d_head, group_size,
                k_buf, v_buf,
            )?;
        (after_attn, k_buf_new, v_buf_new, None)
    };

    let ffn_norm = embed_constant(g, model, &format!("blk.{}.ffn_norm.weight", layer))?;
    let xn2 = g.norm(after_attn, ffn_norm, NormKind::Rms, config.rms_eps);

    let w_gate = embed_weight(g, model, &format!("blk.{}.ffn_gate.weight", layer))?;
    let w_up = embed_weight(g, model, &format!("blk.{}.ffn_up.weight", layer))?;
    let w_down = embed_weight(g, model, &format!("blk.{}.ffn_down.weight", layer))?;

    let gate = wmm(g, xn2, &w_gate);
    let gate = g.activation(gate, ActivationKind::SiLU);
    let up = wmm(g, xn2, &w_up);
    let hidden = g.mul(gate, up);
    let ffn_out = wmm(g, hidden, &w_down);
    let x_out = g.add(after_attn, ffn_out);

    Ok(InPlaceDecodeBlockResult { x_out, k_buf_new, v_buf_new, shortconv_state_out })
}

#[allow(clippy::too_many_arguments)]
fn build_decode_attn_block(
    g: &mut GraphBuilder,
    model: &GgufModel,
    config: &LlamaConfig,
    layer: usize,
    x: NodeId,
    attn_norm: NodeId,
    new_seq: usize,
    kv_pos: NodeId,
    max_seq: usize,
    n_heads: usize,
    n_heads_kv: usize,
    d_head: usize,
    group_size: usize,
    k_buf: NodeId,
    v_buf: NodeId,
) -> Result<(NodeId, NodeId, NodeId), LoadError> {
    use crate::llama::{embed_weight, wmm};
    let xn = g.norm(x, attn_norm, NormKind::Rms, config.rms_eps);

    let w_q = embed_weight(g, model, &format!("blk.{}.attn_q.weight", layer))?;
    let w_k = embed_weight(g, model, &format!("blk.{}.attn_k.weight", layer))?;
    let w_v = embed_weight(g, model, &format!("blk.{}.attn_v.weight", layer))?;
    let w_o = embed_weight(g, model, &format!("blk.{}.attn_output.weight", layer))?;

    let q_new = wmm(g, xn, &w_q);
    let k_new = wmm(g, xn, &w_k);
    let v_new = wmm(g, xn, &w_v);

    let q_h = { let r = g.reshape(q_new, &[new_seq, n_heads, d_head]); g.transpose(r, &[1, 0, 2]) };
    let k_h = { let r = g.reshape(k_new, &[new_seq, n_heads_kv, d_head]); g.transpose(r, &[1, 0, 2]) };
    let v_h = { let r = g.reshape(v_new, &[new_seq, n_heads_kv, d_head]); g.transpose(r, &[1, 0, 2]) };

    let (q_h, k_h) = if config.qk_norm {
        let q_norm = embed_constant(g, model, &format!("blk.{}.attn_q_norm.weight", layer))?;
        let k_norm = embed_constant(g, model, &format!("blk.{}.attn_k_norm.weight", layer))?;
        let q_n = g.norm(q_h, q_norm, NormKind::Rms, config.rms_eps);
        let k_n = g.norm(k_h, k_norm, NormKind::Rms, config.rms_eps);
        (q_n, k_n)
    } else {
        (q_h, k_h)
    };

    // Dynamic RoPE position + scatter offset (both = cached_seq, from
    // the `kv_pos` runtime input).
    let q_rot = g.rope_dyn(q_h, config.rope_base, kv_pos);
    let k_rot = g.rope_dyn(k_h, config.rope_base, kv_pos);

    let k_buf_new = g.scatter_inplace_dyn(k_buf, k_rot, 1, kv_pos);
    let v_buf_new = g.scatter_inplace_dyn(v_buf, v_h, 1, kv_pos);

    // FULL-buffer attention (no slice): K/V are the entire
    // `[n_heads_kv, max_seq, d_head]` buffer → constant shape.
    let k_full = if group_size == 1 { k_buf_new } else {
        let r = g.reshape(k_buf_new, &[n_heads_kv, 1, max_seq, d_head]);
        let rep = g.repeat(r, 1, group_size);
        g.reshape(rep, &[n_heads, max_seq, d_head])
    };
    let v_full = if group_size == 1 { v_buf_new } else {
        let r = g.reshape(v_buf_new, &[n_heads_kv, 1, max_seq, d_head]);
        let rep = g.repeat(r, 1, group_size);
        g.reshape(rep, &[n_heads, max_seq, d_head])
    };

    let inv_sqrt_d = 1.0 / (d_head as f32).sqrt();
    let scores = g.matmul_bt_scaled(q_rot, k_full, inv_sqrt_d);
    // Dynamic causal offset: query rel-pos q attends keys
    // 0..=q + cached_seq. Tail positions (≥ valid_seq) are
    // > q + cached_seq → masked to -inf, so the zero-filled buffer
    // tail never leaks. Bit-identical to the sliced path.
    let probs = g.softmax_causal_offset_dyn(scores, -1, kv_pos);
    let ctx_h = g.matmul(probs, v_full);

    let ctx = { let t = g.transpose(ctx_h, &[1, 0, 2]); g.reshape(t, &[new_seq, n_heads * d_head]) };
    let y = wmm(g, ctx, &w_o);
    let after_attn = g.add(x, y);

    Ok((after_attn, k_buf_new, v_buf_new))
}

/// Streaming variant of LFM2's shortconv attention sub-block.
///
/// `state_in` carries the previous step's rolling window of length
/// `taps - 1` (per-channel). The returned `state_out` is the
/// updated window. Numerically equivalent to a fresh prefill over
/// `[state_history..new_tokens]` — the conv1d_depthwise_causal
/// kernel's left-zero-pad is replaced by the real state we kept
/// across decode steps.
///
/// Returns `(x_out, state_out)`. The state output should be wired
/// as a graph output and fed back as `state_in` next step (the
/// runtime's `ShortConvStateCache` does that bookkeeping, mirror-
/// imaging `InPlaceKvCache` for the attention layers).
#[allow(clippy::too_many_arguments)]
pub fn build_lfm2_shortconv_block_streaming(
    g: &mut GraphBuilder,
    model: &GgufModel,
    config: &crate::llama::LlamaConfig,
    layer: usize,
    x: NodeId,
    attn_norm: NodeId,
    new_seq: usize,
    state_in: NodeId,
) -> Result<(NodeId, NodeId), crate::llama::LoadError> {
    let d = config.embedding_length;
    let taps = config.shortconv_l_cache.max(3);
    let window = taps.saturating_sub(1);

    let xn = g.norm(x, attn_norm, NormKind::Rms, config.rms_eps);

    // Input projection: [seq, d] @ [3d, d]^T → [seq, 3d].
    let w_in_proj = embed_constant(g, model,
        &format!("blk.{}.shortconv.in_proj.weight", layer))?;
    let bcx = g.matmul_bt(xn, w_in_proj);

    let b       = g.slice(bcx, 1, 0,       d);
    let c_chunk = g.slice(bcx, 1, d,       d);
    let x_inner = g.slice(bcx, 1, 2 * d,   d);
    let bx = g.mul(b, x_inner); // [new_seq, d]

    // Concatenate the prior rolling window with the new gated input
    // along the sequence axis. effective shape: [window + new_seq, d].
    // The conv below sees the real prior tokens (no zero-padding
    // artefact at the streaming boundary).
    let effective = g.concat(state_in, bx, 0);

    let w_conv = embed_constant(g, model,
        &format!("blk.{}.shortconv.conv.weight", layer))?;
    // The op assumes left-zero-padding for the start of the
    // sequence; we've replaced those leading zeros with `state_in`,
    // so positions [window..window+new_seq] in the conv output are
    // the correct streamed outputs (no zero-pad contribution).
    let conv_full = g.conv1d_depthwise_causal(effective, w_conv, taps);
    let conv_out = g.slice(conv_full, 0, window, new_seq); // [new_seq, d]

    let y = g.mul(c_chunk, conv_out);

    let w_out_proj = embed_constant(g, model,
        &format!("blk.{}.shortconv.out_proj.weight", layer))?;
    let y_out = g.matmul_bt(y, w_out_proj);
    let x_out = g.add(x, y_out);

    // Updated rolling window: the last `window` positions of
    // `effective`. Slicing at `[new_seq..new_seq + window]` covers
    // both cases (new_seq ≥ window and new_seq < window).
    let state_out = g.slice(effective, 0, new_seq, window);

    Ok((x_out, state_out))
}
