//! High-level embedding API.
//!
//! Given a prompt, runs a model up through the final norm and pools the
//! per-token hidden states into a single fixed-dimensional vector. The
//! most common use case for representation models: similarity search,
//! retrieval, clustering, classification.
//!
//! The model is run as an encoder — the lm_head projection is skipped,
//! saving `seq * vocab * d_model` flops per call.
//!
//! Three pooling strategies:
//! - `Pooling::Mean`: average over all token hidden states. Most common
//!   for sentence embeddings.
//! - `Pooling::LastToken`: take the final position's hidden state.
//!   Standard for causal decoder-only models used as feature extractors
//!   (Llama, Mistral, Qwen at inference for embeddings).
//! - `Pooling::None`: return per-token hidden states as a flat vector
//!   `[seq * d_model]`. Caller pools externally.

use crate::generate::{resolve_tokenizer_kind, tokens_to_tensor, GenerateError, TokenizerKind};
use crate::Runtime;
use jouleclaw_loader_gguf::llama::build_llama_encoder_graph;
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_loader_gguf::GgufModel;
use crate::{compile, execute, ExecutionOptions};
use std::collections::HashMap;

/// How to pool per-token hidden states into a fixed-dim embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Average over all positions. Output shape: `[d_model]`.
    Mean,
    /// Take the last position's hidden state. Output shape: `[d_model]`.
    LastToken,
    /// No pooling. Output shape: `[seq * d_model]` — caller pools.
    None,
}

/// Configuration for an embedding call.
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    pub pooling: Pooling,
    pub add_bos: bool,
    pub tokenizer_kind: TokenizerKind,
    /// L2-normalize the output vector. Standard for cosine-similarity use.
    /// Only meaningful when pooling produces a single vector; ignored
    /// when `pooling == Pooling::None`.
    pub l2_normalize: bool,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            pooling: Pooling::Mean,
            add_bos: true,
            tokenizer_kind: TokenizerKind::Auto,
            l2_normalize: true,
        }
    }
}

/// Result of an embedding call.
#[derive(Debug, Clone)]
pub struct EmbedResult {
    /// The pooled (and optionally L2-normalized) embedding vector. Shape
    /// is `[d_model]` for Mean/LastToken pooling, `[seq * d_model]` for
    /// None pooling.
    pub vector: Vec<f32>,
    /// Dimensionality of the embedding (the model's `d_model`).
    pub d_model: usize,
    /// Number of tokens after tokenization (including BOS if applicable).
    pub token_count: usize,
}

/// Run a model on `text`, pool the hidden states, and return the
/// resulting embedding.
pub fn embed(
    model: &GgufModel,
    vocab: &Vocab,
    text: &str,
    cfg: &EmbedConfig,
) -> Result<EmbedResult, GenerateError> {
    let tokenizer_kind = resolve_tokenizer_kind(vocab, cfg.tokenizer_kind);
    let tokens = match tokenizer_kind {
        TokenizerKind::Spm => vocab.encode_spm(text, cfg.add_bos),
        TokenizerKind::Bpe => vocab.encode_bpe_regex(text, cfg.add_bos),
        TokenizerKind::Auto => unreachable!(),
    };
    if tokens.is_empty() {
        return Err(GenerateError::EmptyPromptTokens);
    }

    let runtime = Runtime::boot();
    let encoder = build_llama_encoder_graph(model, tokens.len())?;
    let compiled = compile(encoder.graph, &runtime.kernels)?;

    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), tokens_to_tensor(&tokens));
    let res = execute(&compiled, inputs, ExecutionOptions::default())?;

    let hidden = res.outputs.get("hidden_states").expect("hidden_states");
    let d_model = encoder.config.embedding_length;
    let seq = tokens.len();
    let flat = hidden.as_f32_vec();
    debug_assert_eq!(flat.len(), seq * d_model);

    let mut vector = match cfg.pooling {
        Pooling::Mean => {
            let mut v = vec![0f32; d_model];
            for s in 0..seq {
                for d in 0..d_model {
                    v[d] += flat[s * d_model + d];
                }
            }
            let inv_seq = 1.0 / seq as f32;
            for d in 0..d_model { v[d] *= inv_seq; }
            v
        }
        Pooling::LastToken => {
            let off = (seq - 1) * d_model;
            flat[off..off + d_model].to_vec()
        }
        Pooling::None => flat,
    };

    if cfg.l2_normalize && cfg.pooling != Pooling::None {
        let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
        let inv_norm = if norm_sq > 0.0 { 1.0 / norm_sq.sqrt() } else { 0.0 };
        for x in vector.iter_mut() { *x *= inv_norm; }
    }

    Ok(EmbedResult {
        vector, d_model, token_count: seq,
    })
}

/// Cosine similarity between two embedding vectors. Both must have the
/// same length; if both are L2-normalized this reduces to a dot product.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(),
        "cosine_similarity: length mismatch {} vs {}", a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / denom
}
