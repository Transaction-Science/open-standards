//! Llama-style architecture loader.
//!
//! Takes a parsed `GgufModel` and builds a complete inference graph using the
//! `jouleclaw_core::blocks` block constructors. Weights are extracted into
//! `Tensor`s and embedded as constants, so the only input the user needs to
//! bind at execute time is the token-id sequence.
//!
//! Phase 1.4 supports:
//! - F32 and F16 (widened to F32) tensors
//! - Llama-style architecture: token embedding + N transformer blocks +
//!   final norm + lm_head
//! - Single-head attention only (multi-head requires reshape/transpose ops)
//!
//! Phase 1.5+ will add:
//! - Multi-head and grouped-query attention (after reshape/transpose primitives)
//! - Quantized weight types
//! - Tied embeddings (when `output.weight` is missing)
//! - RoPE positional encoding (currently absent — toy graphs work but real
//!   inference requires it)

use crate::{tensor_from_gguf, GgmlType, GgufModel, ParseError};
use jouleclaw_core::graph::{Graph, GraphBuilder, NodeId};
use jouleclaw_core::op::{ActivationKind, NormKind};
use jouleclaw_core::tensor::{Dtype, TensorMeta, TensorRef};

/// Parsed transformer configuration extracted from GGUF metadata.
///
/// Despite the name (kept for API stability), this now covers the
/// `llama` *and* `qwen3` architectures. Qwen3 differs by: per-head
/// QK-RMSNorm (`attn_q_norm`/`attn_k_norm`), a decoupled `head_dim`
/// (`<arch>.attention.key_length`, not `embedding/head_count`), a
/// higher RoPE base, and tied input/output embeddings (no
/// `output.weight`). Metadata keys are read under the arch prefix.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub arch: String,
    pub vocab_size: usize,
    pub embedding_length: usize,
    pub block_count: usize,
    pub feed_forward_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    /// Per-head dimension. For llama this is `embedding/head_count`;
    /// for qwen3 it is the explicit `key_length` (may differ).
    pub head_dim: usize,
    /// True when the model carries per-head QK-RMSNorm weights
    /// (`blk.*.attn_q_norm.weight`) — the qwen3 signature. Detected
    /// by tensor presence, not assumed from the arch string.
    pub qk_norm: bool,
    pub rms_eps: f32,
    pub context_length: usize,
    pub rope_base: f32,
    /// LFM2 hybrid arch: per-layer kv-head count. A `0` entry means
    /// that layer is **recurrent** (`shortconv` block, see
    /// [`Self::shortconv_l_cache`]), a non-zero entry means it's a
    /// standard attention layer. Empty for non-LFM2 archs.
    pub per_layer_head_count_kv: Vec<usize>,
    /// LFM2 shortconv cache length (kernel taps). `lfm2.shortconv.l_cache`,
    /// typically 3. `0` for non-LFM2 archs.
    pub shortconv_l_cache: usize,
    /// Encoder/embedding models (eurobert, jina-v5) use bidirectional
    /// (non-causal) attention. Set from `<arch>.attention.causal =
    /// False` in metadata; default `false` (causal) for autoregressive
    /// decoders.
    pub bidirectional: bool,
    /// Pooling strategy for sentence-embedding output. Maps llama.cpp
    /// `LLAMA_POOLING_TYPE_*`: 0=None, 1=Mean, 2=CLS, 3=Last, 4=Rank.
    /// Drives `MrlEmbedder` in `crates/mrl`. `0` for non-encoder models.
    pub pooling_type: u32,
}

const SUPPORTED_ARCHS: &[&str] = &[
    "llama", "qwen3", "hunyuan-dense", "lfm2", "eurobert",
];

impl LlamaConfig {
    pub fn from_metadata(model: &GgufModel) -> Result<Self, LoadError> {
        let arch = model.metadata_string("general.architecture")
            .ok_or_else(|| LoadError::MissingMetadata("general.architecture"))?
            .to_string();
        if !SUPPORTED_ARCHS.contains(&arch.as_str()) {
            return Err(LoadError::UnsupportedArchitecture(arch));
        }
        // All metadata keys are namespaced under the arch string.
        let k = |suffix: &str| format!("{}.{}", arch, suffix);
        let mu = |suffix: &str| model.metadata_u64(&k(suffix));

        let embedding_length = mu("embedding_length")
            .ok_or(LoadError::MissingMetadata("embedding_length"))? as usize;
        let head_count = mu("attention.head_count")
            .ok_or(LoadError::MissingMetadata("attention.head_count"))? as usize;
        // Most archs store head_count_kv as a scalar; LFM2 stores it as
        // a per-layer ARRAY (one entry per block, 0 = recurrent/conv,
        // non-zero = attention). Try scalar first, then fall back to
        // the first non-zero array entry as the "representative" kv.
        let per_layer_head_count_kv = model
            .metadata_u_array(&k("attention.head_count_kv"))
            .unwrap_or_default();
        let head_count_kv = mu("attention.head_count_kv")
            .map(|v| v as usize)
            .or_else(|| per_layer_head_count_kv.iter().copied().find(|&v| v > 0))
            .or_else(|| Some(head_count))
            .unwrap_or(1);
        // qwen3 decouples head_dim via key_length; llama does not set it.
        let head_dim = mu("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or_else(|| embedding_length / head_count.max(1));
        // QK-RMSNorm: detect by the actual presence of the weight, so
        // a future llama variant with q_norm also works and a qwen3
        // export lacking it degrades gracefully.
        let qk_norm = model.tensor_by_name("blk.0.attn_q_norm.weight").is_some();

        Ok(Self {
            vocab_size: mu("vocab_size")
                .or_else(|| count_tokens(model).ok())
                .unwrap_or(0) as usize,
            embedding_length,
            block_count: mu("block_count")
                .ok_or(LoadError::MissingMetadata("block_count"))? as usize,
            feed_forward_length: mu("feed_forward_length")
                .ok_or(LoadError::MissingMetadata("feed_forward_length"))? as usize,
            head_count,
            head_count_kv,
            head_dim,
            qk_norm,
            rms_eps: model.metadata_f32(&k("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-6),
            shortconv_l_cache: mu("shortconv.l_cache").unwrap_or(0) as usize,
            per_layer_head_count_kv,
            bidirectional: model.metadata.get(&k("attention.causal"))
                .and_then(|v| v.as_bool())
                .map(|c| !c).unwrap_or(false),
            pooling_type: mu("pooling_type").unwrap_or(0) as u32,
            context_length: mu("context_length").unwrap_or(2048) as usize,
            // llama 1/2: 1e4; llama 3: 5e5; qwen3 Bonsai: 1e6.
            rope_base: model.metadata_f32(&k("rope.freq_base")).unwrap_or(10000.0),
            arch,
        })
    }
}

fn count_tokens(model: &GgufModel) -> Result<u64, LoadError> {
    let arr = model.metadata.get("tokenizer.ggml.tokens")
        .and_then(|v| match v {
            crate::GgufValue::Array(a) => Some(a.len()),
            _ => None,
        })
        .ok_or(LoadError::MissingMetadata("tokenizer.ggml.tokens"))?;
    Ok(arr as u64)
}

/// A built Llama graph: structure plus the bound config.
pub struct LlamaGraph {
    pub graph: Graph,
    pub config: LlamaConfig,
}

/// Build a Llama inference graph from a parsed GGUF model.
///
/// The graph takes one input — `token_ids` of shape `[seq_len]` (I32) — and
/// produces one output — `logits` of shape `[seq_len, vocab_size]` (F32).
///
/// All weights are embedded as constants in the graph; nothing needs to be
/// bound at execution time besides the token IDs.
///
/// `seq_len` is fixed at graph-build time; rebuild the graph for a different
/// sequence length. Phase 2+ adds dynamic shapes.
///
/// Phase 1.4 limitation: this implements **single-head attention** because
/// multi-head requires reshape/transpose ops not yet in the primitive set.
/// On a multi-head model the graph is structurally a head_count=1 variant
/// of the architecture. It runs and produces shape-correct logits, but is
/// not a faithful reproduction of multi-head attention semantics.
pub fn build_llama_graph(model: &GgufModel, seq_len: usize) -> Result<LlamaGraph, LoadError> {
    let config = LlamaConfig::from_metadata(model)?;
    let mut g = GraphBuilder::new();

    // Input: token IDs.
    let token_ids = g.input(
        "token_ids",
        TensorMeta::new(Dtype::I32, &[seq_len]),
    );

    // Token embedding + (tied) LM head. For Q2_0 the table is decoded
    // on demand by the lookup and reused *packed* for the head — the
    // full f32 table (≈1.2 GB for Bonsai's 151669×2048) is never
    // materialised. Non-Q2_0 models keep the exact prior dense path.
    let te_info = model.tensor_by_name("token_embd.weight")
        .ok_or_else(|| LoadError::MissingTensor("token_embd.weight".into()))?;
    let token_embd_w = if matches!(te_info.dtype, GgmlType::Q2_0 | GgmlType::Q1_0) {
        Some(embed_weight(&mut g, model, "token_embd.weight")?)
    } else {
        None
    };

    let mut x = match &token_embd_w {
        Some(Weight::Ternary { node, out, k }) =>
            g.lookup_ternary(token_ids, *node, *out, *k),
        _ => {
            let te = embed_constant(&mut g, model, "token_embd.weight")?;
            g.lookup(token_ids, te)
        }
    };

    // Transformer blocks.
    for layer in 0..config.block_count {
        x = build_block(&mut g, model, &config, layer, x, seq_len)?;
    }

    // Final norm + lm_head. LFM2 stores the final norm as
    // `token_embd_norm.weight` (a naming quirk inherited from
    // Liquid's Transformers config); every other arch uses
    // `output_norm.weight`. Pick whichever exists.
    let final_norm_name = if config.arch == "lfm2"
        && model.tensor_by_name("token_embd_norm.weight").is_some()
    {
        "token_embd_norm.weight"
    } else {
        "output_norm.weight"
    };
    let output_norm = embed_constant(&mut g, model, final_norm_name)?;
    let xn = g.norm(x, output_norm, NormKind::Rms, config.rms_eps);

    // LM head → `[seq, vocab]`. Untied: its own `output.weight`
    // (ternary if Q2_0). Tied (qwen3 Bonsai): reuse the already-packed
    // `token_embd` constant — no second copy, no f32 dequant.
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
    Ok(LlamaGraph { graph: g.build(), config })
}

/// Like [`build_llama_graph`] but the sequence enters as a precomputed
/// `input_embeds` tensor (`[seq_len, d_model]` F32) instead of
/// `token_ids → lookup`. This is the multimodal entry point: callers
/// (LFM2.5-VL) build the embedding stream themselves by splicing
/// projected image tokens into the text token embeddings, then run the
/// LFM2 backbone over the combined sequence. Output is still
/// `logits [seq_len, vocab]`.
pub fn build_llama_graph_from_embeds(
    model: &GgufModel,
    seq_len: usize,
) -> Result<LlamaGraph, LoadError> {
    let config = LlamaConfig::from_metadata(model)?;
    let mut g = GraphBuilder::new();

    // The caller binds "input_embeds" at execute time.
    let mut x = g.input(
        "input_embeds",
        TensorMeta::new(Dtype::F32, &[seq_len, config.embedding_length]),
    );

    // token_embd is still needed for the (tied) LM head.
    let te_info = model.tensor_by_name("token_embd.weight")
        .ok_or_else(|| LoadError::MissingTensor("token_embd.weight".into()))?;
    let token_embd_w = if matches!(
        te_info.dtype, GgmlType::Q2_0 | GgmlType::Q1_0 | GgmlType::STQ1_0,
    ) {
        Some(embed_weight(&mut g, model, "token_embd.weight")?)
    } else {
        None
    };

    for layer in 0..config.block_count {
        x = build_block(&mut g, model, &config, layer, x, seq_len)?;
    }

    let final_norm_name = if config.arch == "lfm2"
        && model.tensor_by_name("token_embd_norm.weight").is_some()
    {
        "token_embd_norm.weight"
    } else {
        "output_norm.weight"
    };
    let output_norm = embed_constant(&mut g, model, final_norm_name)?;
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
    Ok(LlamaGraph { graph: g.build(), config })
}

/// Built Llama "encoder" graph — terminates at the per-token hidden
/// states (post-norm, pre-lm-head). Used for embedding/representation
/// tasks rather than text generation.
pub struct LlamaEncoderGraph {
    pub graph: Graph,
    pub config: LlamaConfig,
}

/// Build a Llama graph that outputs `hidden_states` with shape
/// `[seq, d_model]` instead of logits. This is the standard
/// representation used for sentence embeddings, retrieval, clustering,
/// and other tasks where you want a fixed-dimensional summary of the
/// input rather than a probability distribution over tokens.
///
/// The graph runs through all transformer blocks and the final RMS norm,
/// then exposes the result as the output. It does NOT run the lm_head
/// projection; the saved compute is roughly `seq_len * vocab_size *
/// d_model` flops per call.
pub fn build_llama_encoder_graph(
    model: &GgufModel,
    seq_len: usize,
) -> Result<LlamaEncoderGraph, LoadError> {
    let config = LlamaConfig::from_metadata(model)?;
    let mut g = GraphBuilder::new();

    let token_ids = g.input(
        "token_ids",
        TensorMeta::new(Dtype::I32, &[seq_len]),
    );

    let token_embd = embed_constant(&mut g, model, "token_embd.weight")?;
    let mut x = g.lookup(token_ids, token_embd);

    for layer in 0..config.block_count {
        x = build_block(&mut g, model, &config, layer, x, seq_len)?;
    }

    // Final norm, but no lm_head.
    let output_norm = embed_constant(&mut g, model, "output_norm.weight")?;
    let hidden_states = g.norm(x, output_norm, NormKind::Rms, config.rms_eps);
    g.output("hidden_states", hidden_states);

    Ok(LlamaEncoderGraph { graph: g.build(), config })
}

fn build_block(
    g: &mut GraphBuilder,
    model: &GgufModel,
    config: &LlamaConfig,
    layer: usize,
    x: NodeId,
    seq_len: usize,
) -> Result<NodeId, LoadError> {
    // Attention.
    let attn_norm = embed_constant(g, model, &format!("blk.{}.attn_norm.weight", layer))?;

    // LFM2 hybrid: per-layer decide between standard attention and the
    // `shortconv` recurrent block. `head_count_kv[layer] == 0` ⇒ this
    // layer is recurrent. Empty `per_layer_head_count_kv` ⇒ not LFM2.
    let is_lfm2_recurrent = config.arch == "lfm2"
        && config.per_layer_head_count_kv.get(layer).copied() == Some(0);

    let after_attn = if is_lfm2_recurrent {
        build_lfm2_shortconv_block(g, model, config, layer, x, attn_norm, seq_len)?
    } else if config.bidirectional {
        // Encoder / sentence-embedding model (eurobert, jina-v5). Same
        // GQA/MHA shape as the qwen3 path but: (a) no QK-RMSNorm,
        // (b) `softmax` instead of `softmax_causal` — attention attends
        // over the entire sequence, no future mask. Weight matmuls
        // still route through `wmm`, so quantised paths still light up.
        let nq = config.head_count;
        let nkv = config.head_count_kv;
        let dh = config.head_dim;
        let group = nq / nkv;
        let w_q = embed_weight(g, model, &format!("blk.{}.attn_q.weight", layer))?;
        let w_k = embed_weight(g, model, &format!("blk.{}.attn_k.weight", layer))?;
        let w_v = embed_weight(g, model, &format!("blk.{}.attn_v.weight", layer))?;
        let w_o = embed_weight(g, model, &format!("blk.{}.attn_output.weight", layer))?;
        let xn = g.norm(x, attn_norm, NormKind::Rms, config.rms_eps);
        let q = wmm(g, xn, &w_q);
        let k = wmm(g, xn, &w_k);
        let v = wmm(g, xn, &w_v);
        let q_h = { let r = g.reshape(q, &[seq_len, nq, dh]); g.transpose(r, &[1, 0, 2]) };
        let k_h = { let r = g.reshape(k, &[seq_len, nkv, dh]); g.transpose(r, &[1, 0, 2]) };
        let v_h = { let r = g.reshape(v, &[seq_len, nkv, dh]); g.transpose(r, &[1, 0, 2]) };
        let q_rot = g.rope(q_h, config.rope_base, 0);
        let k_rot = g.rope(k_h, config.rope_base, 0);
        let k_broad = if group == 1 { k_rot } else {
            let r = g.reshape(k_rot, &[nkv, 1, seq_len, dh]);
            let rep = g.repeat(r, 1, group);
            g.reshape(rep, &[nq, seq_len, dh])
        };
        let v_broad = if group == 1 { v_h } else {
            let r = g.reshape(v_h, &[nkv, 1, seq_len, dh]);
            let rep = g.repeat(r, 1, group);
            g.reshape(rep, &[nq, seq_len, dh])
        };
        let inv_sqrt_d = 1.0 / (dh as f32).sqrt();
        let scores = g.matmul_bt_scaled(q_rot, k_broad, inv_sqrt_d);
        let probs = g.softmax(scores, -1);          // non-causal
        let ctx_h = g.matmul(probs, v_broad);
        let ctx = { let t = g.transpose(ctx_h, &[1, 0, 2]); g.reshape(t, &[seq_len, nq * dh]) };
        let y = wmm(g, ctx, &w_o);
        g.add(x, y)
    } else if config.qk_norm && config.head_count > 1 {
        // qwen3 attention, inlined so the q/k/v/o projections can route
        // through the ternary kernel (Q2_0 weights stay packed). This
        // mirrors `jouleclaw_core::blocks::multi_head_attention_gqa_qknorm`
        // exactly except the four projections use `wmm`.
        if config.head_count % config.head_count_kv != 0 {
            return Err(LoadError::UnsupportedArchitecture(format!(
                "GQA: head_count ({}) must be divisible by head_count_kv ({})",
                config.head_count, config.head_count_kv)));
        }
        let w_q = embed_weight(g, model, &format!("blk.{}.attn_q.weight", layer))?;
        let w_k = embed_weight(g, model, &format!("blk.{}.attn_k.weight", layer))?;
        let w_v = embed_weight(g, model, &format!("blk.{}.attn_v.weight", layer))?;
        let w_o = embed_weight(g, model, &format!("blk.{}.attn_output.weight", layer))?;
        let w_q_norm = embed_constant(g, model, &format!("blk.{}.attn_q_norm.weight", layer))?;
        let w_k_norm = embed_constant(g, model, &format!("blk.{}.attn_k_norm.weight", layer))?;

        let nq = config.head_count;
        let nkv = config.head_count_kv;
        let dh = config.head_dim;
        let group = nq / nkv;

        let xn = g.norm(x, attn_norm, NormKind::Rms, config.rms_eps);
        let q = wmm(g, xn, &w_q);
        let k = wmm(g, xn, &w_k);
        let v = wmm(g, xn, &w_v);

        let q_h = { let r = g.reshape(q, &[seq_len, nq, dh]); g.transpose(r, &[1, 0, 2]) };
        let k_h = { let r = g.reshape(k, &[seq_len, nkv, dh]); g.transpose(r, &[1, 0, 2]) };
        let v_h = { let r = g.reshape(v, &[seq_len, nkv, dh]); g.transpose(r, &[1, 0, 2]) };

        // Per-head QK-RMSNorm (pre-RoPE), weight broadcast over [heads, seq].
        let q_h = g.norm(q_h, w_q_norm, NormKind::Rms, config.rms_eps);
        let k_h = g.norm(k_h, w_k_norm, NormKind::Rms, config.rms_eps);

        let q_rot = g.rope(q_h, config.rope_base, 0);
        let k_rot = g.rope(k_h, config.rope_base, 0);

        let k_broad = if group == 1 { k_rot } else {
            let r = g.reshape(k_rot, &[nkv, 1, seq_len, dh]);
            let rep = g.repeat(r, 1, group);
            g.reshape(rep, &[nq, seq_len, dh])
        };
        let v_broad = if group == 1 { v_h } else {
            let r = g.reshape(v_h, &[nkv, 1, seq_len, dh]);
            let rep = g.repeat(r, 1, group);
            g.reshape(rep, &[nq, seq_len, dh])
        };

        let inv_sqrt_d = 1.0 / (dh as f32).sqrt();
        let scores = g.matmul_bt_scaled(q_rot, k_broad, inv_sqrt_d);
        let probs = g.softmax_causal(scores, -1);
        let ctx_h = g.matmul(probs, v_broad);
        let ctx = { let t = g.transpose(ctx_h, &[1, 0, 2]); g.reshape(t, &[seq_len, nq * dh]) };
        let y = wmm(g, ctx, &w_o);
        g.add(x, y)
    } else if config.head_count > 1 {
        let w_q = embed_constant(g, model, &format!("blk.{}.attn_q.weight", layer))?;
        let w_k = embed_constant(g, model, &format!("blk.{}.attn_k.weight", layer))?;
        let w_v = embed_constant(g, model, &format!("blk.{}.attn_v.weight", layer))?;
        let w_o = embed_constant(g, model, &format!("blk.{}.attn_output.weight", layer))?;
        let d_head = config.embedding_length / config.head_count;
        if d_head * config.head_count != config.embedding_length {
            return Err(LoadError::UnsupportedArchitecture(format!(
                "embedding_length {} not divisible by head_count {}",
                config.embedding_length, config.head_count)));
        }
        if config.head_count_kv != config.head_count {
            // Grouped-Query Attention: K/V heads < Q heads.
            if config.head_count % config.head_count_kv != 0 {
                return Err(LoadError::UnsupportedArchitecture(format!(
                    "GQA: head_count ({}) must be divisible by head_count_kv ({})",
                    config.head_count, config.head_count_kv)));
            }
            jouleclaw_core::blocks::multi_head_attention_gqa(
                g, x, attn_norm, w_q, w_k, w_v, w_o,
                seq_len, config.head_count, config.head_count_kv,
                d_head, config.rope_base,
            )
        } else {
            // Standard MHA: n_heads_q == n_heads_kv.
            jouleclaw_core::blocks::multi_head_attention_with_rope(
                g, x, attn_norm, w_q, w_k, w_v, w_o,
                seq_len, config.head_count, d_head, config.rope_base,
            )
        }
    } else {
        // Single-head fast path.
        let w_q = embed_constant(g, model, &format!("blk.{}.attn_q.weight", layer))?;
        let w_k = embed_constant(g, model, &format!("blk.{}.attn_k.weight", layer))?;
        let w_v = embed_constant(g, model, &format!("blk.{}.attn_v.weight", layer))?;
        let w_o = embed_constant(g, model, &format!("blk.{}.attn_output.weight", layer))?;
        let xn = g.norm(x, attn_norm, NormKind::Rms, config.rms_eps);
        let q = g.matmul_bt(xn, w_q);
        let k = g.matmul_bt(xn, w_k);
        let v = g.matmul_bt(xn, w_v);
        let scores = g.matmul_bt(q, k);
        let probs = g.softmax(scores, -1);
        let ctx = g.matmul(probs, v);
        let attn_out = g.matmul_bt(ctx, w_o);
        g.add(x, attn_out)
    };

    // FFN (gated SiLU; same regardless of head count).
    let ffn_norm = embed_constant(g, model, &format!("blk.{}.ffn_norm.weight", layer))?;
    let xn2 = g.norm(after_attn, ffn_norm, NormKind::Rms, config.rms_eps);

    // FFN projections route through `wmm`: ternary for Q2_0 (Bonsai),
    // bit-identical dense `matmul_bt` for everything else.
    let w_gate = embed_weight(g, model, &format!("blk.{}.ffn_gate.weight", layer))?;
    let w_up = embed_weight(g, model, &format!("blk.{}.ffn_up.weight", layer))?;
    let w_down = embed_weight(g, model, &format!("blk.{}.ffn_down.weight", layer))?;

    let gate = wmm(g, xn2, &w_gate);
    let gate = g.activation(gate, ActivationKind::SiLU);
    let up = wmm(g, xn2, &w_up);
    let hidden = g.mul(gate, up);
    let ffn_out = wmm(g, hidden, &w_down);
    Ok(g.add(after_attn, ffn_out))
}

/// LFM2 `shortconv` recurrent block — the depthwise-causal-conv
/// counterpart to attention in Liquid's hybrid model. Mirrors the
/// upstream `build_shortconv_block` in `ggml-org/llama.cpp@master`
/// (`src/models/lfm2.cpp`) for the prefill case (no cached state):
///
///   xn   = RMSNorm(x, attn_norm)
///   bcx  = xn @ in_proj^T                       // [seq, 3*d]
///   b    = bcx[:,   0..d   ]                    // [seq, d]
///   c    = bcx[:,   d..2d  ]
///   x'   = bcx[:, 2d..3d   ]
///   bx   = b * x'                               // gated input
///   y    = c * Conv1DDepthwiseCausal(bx, conv)  // [seq, d]
///   out  = y @ out_proj^T                       // [seq, d]
///   return x + out                              // residual
///
/// Decode/streaming with a 2-element conv-state cache (l_cache - 1)
/// is a follow-on; this prefill path is the correctness floor.
fn build_lfm2_shortconv_block(
    g: &mut GraphBuilder,
    model: &GgufModel,
    config: &LlamaConfig,
    layer: usize,
    x: NodeId,
    attn_norm: NodeId,
    _seq_len: usize,
) -> Result<NodeId, LoadError> {
    let d = config.embedding_length;
    // l_cache in the metadata is the *cache* length (kernel taps = l_cache).
    // For LFM2 it's 3.
    let taps = config.shortconv_l_cache.max(3);

    let xn = g.norm(x, attn_norm, NormKind::Rms, config.rms_eps);

    let w_in_proj = embed_constant(g, model,
        &format!("blk.{}.shortconv.in_proj.weight", layer))?;
    let bcx = g.matmul_bt(xn, w_in_proj);     // [seq, 3*d]

    let b       = g.slice(bcx, 1, 0,       d); // [seq, d]
    let c_chunk = g.slice(bcx, 1, d,       d);
    let x_inner = g.slice(bcx, 1, 2 * d,   d);

    let bx = g.mul(b, x_inner);

    let w_conv = embed_constant(g, model,
        &format!("blk.{}.shortconv.conv.weight", layer))?;
    let conv_out = g.conv1d_depthwise_causal(bx, w_conv, taps);

    let y = g.mul(c_chunk, conv_out);

    let w_out_proj = embed_constant(g, model,
        &format!("blk.{}.shortconv.out_proj.weight", layer))?;
    let y_out = g.matmul_bt(y, w_out_proj);   // [seq, d]

    Ok(g.add(x, y_out))
}

/// Pull a tensor out of the GGUF model and embed it as a constant in the graph.
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

/// A weight operand for a `Y = X @ W^T` projection. `Ternary` keeps the
/// PrismML Q2_0 bytes packed so the ternary kernel runs without ever
/// materialising the f32 weight (the energy/bandwidth win); `Dense` is
/// the existing dequantised-f32 constant path (TinyLlama, F16/Q8_0/…),
/// bit-identical to before.
pub(crate) enum Weight {
    Dense(NodeId),
    Ternary { node: NodeId, out: usize, k: usize },
    Bit { node: NodeId, out: usize, k: usize },
    /// Tencent STQ1_0 sparse-ternary 3:4 (1.3125 bpw, g256).
    SparseTernary { node: NodeId, out: usize, k: usize },
    /// Q8_0 packed weight (32-element blocks, 1 fp16 scale per block,
    /// 8.5 bpw). Stays packed through the graph; the kernel reads
    /// the bytes directly and fuses dequant + matmul into one int8
    /// dot-product pass.
    Q80 { node: NodeId, out: usize, k: usize },
}

/// Embed a projection weight. Q2_0 → packed U8 constant + logical dims;
/// everything else → the usual dequantised-f32 constant.
pub(crate) fn embed_weight(
    g: &mut GraphBuilder,
    model: &GgufModel,
    name: &str,
) -> Result<Weight, LoadError> {
    let info = model.tensor_by_name(name)
        .ok_or_else(|| LoadError::MissingTensor(name.to_string()))?;
    // Q8_0 platform routing: on Apple Silicon, the existing
    // `Weight::Dense` path (dequant Q8_0 → fp32 once at graph compile,
    // then cblas_sgemm via AMX) is genuinely faster than the packed
    // NEON int8 path for single-token decode shapes (m≈17). AMX's
    // dispatch is near-zero overhead and its fp32 throughput is hard
    // to beat at small sizes. On non-AMX edge targets
    // (aarch64-linux-gnu, Android), the packed path wins because
    // sgemm there falls back to scalar/NEON fp32 — no coprocessor to
    // amortize against. So Q8_0 dispatches differently per platform:
    //   macOS:      Dense (dequant + AMX sgemm via AccelerateMatMul)
    //   else:       Q80   (packed bytes + NEON int8 via MatMulQ80Ref)
    let q8_0_packed_path = !cfg!(target_os = "macos");

    match info.dtype {
        GgmlType::Q2_0 | GgmlType::Q1_0 | GgmlType::STQ1_0 => {
            // GGUF ne-order is [in, out]; logical weight is [out, k=in]
            // (reverse), matching `matmul_bt(x[..,in], W[out,in])`.
            let dims: Vec<usize> = info.shape.iter().rev().map(|&d| d as usize).collect();
            let (out, k) = (dims[0], dims[1]);
            let tref = mapped_or_owned_constant(model, info)?;
            let node = g.constant(tref);
            Ok(match info.dtype {
                GgmlType::Q2_0 => Weight::Ternary { node, out, k },
                GgmlType::Q1_0 => Weight::Bit { node, out, k },
                GgmlType::STQ1_0 => Weight::SparseTernary { node, out, k },
                _ => unreachable!(),
            })
        }
        GgmlType::Q8_0 if q8_0_packed_path => {
            // Non-Mac path: keep Q8_0 packed, route through MatMulQ80Ref.
            let dims: Vec<usize> = info.shape.iter().rev().map(|&d| d as usize).collect();
            let (out, k) = (dims[0], dims[1]);
            let tref = mapped_or_owned_constant(model, info)?;
            let node = g.constant(tref);
            Ok(Weight::Q80 { node, out, k })
        }
        _ => Ok(Weight::Dense(embed_constant(g, model, name)?)),
    }
}

/// Build a packed-weight `TensorRef` (Dtype::U8 flat byte buffer) for a
/// tensor that should NOT be dequantised. Uses zero-copy mmap when the
/// model came from a file; falls back to an owned `Vec<u8>` copy for
/// in-memory / synthetic loads (no backing to reference). The mmap
/// path saves the ~hundreds-of-MB copy that previously hit RAM on
/// every model load.
fn mapped_or_owned_constant(
    model: &GgufModel,
    info: &crate::TensorInfo,
) -> Result<TensorRef, LoadError> {
    let bytes = model.tensor_bytes(info);
    let len = bytes.len();
    let meta = TensorMeta::new(Dtype::U8, &[len]);
    let storage = if let Some(backing) = model.mmap_backing() {
        // Zero-copy: refer into the mmap directly. `info.offset` is
        // relative to the data section, which is exactly what
        // `GgufMmapBacking::bytes()` exposes.
        let backing: std::sync::Arc<dyn jouleclaw_core::tensor::ByteBacking> = backing;
        std::sync::Arc::new(jouleclaw_core::tensor::TensorStorage::from_mapped(
            backing, info.offset as usize, len,
        ))
    } else {
        std::sync::Arc::new(jouleclaw_core::tensor::TensorStorage::from_bytes(
            bytes.to_vec(),
        ))
    };
    Ok(TensorRef { meta, storage })
}

/// `X @ W^T`, dispatching to the right kernel by weight format.
pub(crate) fn wmm(g: &mut GraphBuilder, x: NodeId, w: &Weight) -> NodeId {
    match *w {
        Weight::Dense(n) => g.matmul_bt(x, n),
        Weight::Ternary { node, out, k } => g.matmul_bt_ternary(x, node, out, k),
        Weight::Bit { node, out, k } => g.matmul_bt_bit(x, node, out, k),
        Weight::SparseTernary { node, out, k } => g.matmul_bt_stq1_0(x, node, out, k),
        Weight::Q80 { node, out, k } => g.matmul_bt_q8_0(x, node, out, k),
    }
}

/// `table[idx]`, dispatching by table format. The full f32 table is
/// never materialised for `Ternary`/`Bit` — only the requested rows
/// are decoded on demand.
pub(crate) fn lookup_w(g: &mut GraphBuilder, idx: NodeId, table: &Weight) -> NodeId {
    match *table {
        Weight::Dense(n) => g.lookup(idx, n),
        Weight::Ternary { node, out, k } => g.lookup_ternary(idx, node, out, k),
        Weight::Bit { node, out, k } => g.lookup_bit(idx, node, out, k),
        // No STQ1_0 lookup kernel exists — Tencent's only STQ1_0 model
        // (Hy-MT1.5) keeps `token_embd` at Q6_K, so this path is never
        // hit in practice. If a future STQ1_0 model puts the embedding
        // in STQ1_0 too, dequant-and-lookup is the right fallback.
        Weight::SparseTernary { .. } => panic!(
            "lookup_w: STQ1_0-packed lookup tables are not yet supported; \
             the embedding should be stored at higher precision (Q6_K, F16)"),
        // No Q8_0 lookup kernel; models that put `token_embd` at Q8_0
        // need dequant-and-lookup. The existing `embed_constant` path
        // handles this — `embed_weight` returning Q80 here is only
        // expected for projection weights (matmul_bt callers), not
        // for embedding tables.
        Weight::Q80 { .. } => panic!(
            "lookup_w: Q8_0-packed lookup tables aren't supported in this \
             path; load the embedding via embed_constant_pub for dequant \
             + standard lookup instead"),
    }
}

#[derive(Debug)]
pub enum LoadError {
    MissingMetadata(&'static str),
    UnsupportedArchitecture(String),
    MissingTensor(String),
    UnsupportedTensorType { name: String, dtype: GgmlType },
    TensorExtraction { name: String, source: ParseError },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingMetadata(k) => write!(f, "missing metadata key: {}", k),
            Self::UnsupportedArchitecture(a) => write!(f, "unsupported architecture: {}", a),
            Self::MissingTensor(n) => write!(f, "missing tensor: {}", n),
            Self::UnsupportedTensorType { name, dtype } =>
                write!(f, "unsupported tensor type for {}: {:?}", name, dtype),
            Self::TensorExtraction { name, source } =>
                write!(f, "failed to extract {}: {:?}", name, source),
        }
    }
}

impl std::error::Error for LoadError {}
