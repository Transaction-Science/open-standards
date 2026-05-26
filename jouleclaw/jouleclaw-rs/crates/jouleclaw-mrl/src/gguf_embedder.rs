//! `GgufTextEmbedder` — encode text into a dense embedding using a
//! real GGUF encoder model (e.g. `jinaai/jina-embeddings-v5-omni-nano`).
//!
//! Pairs with the rest of [`crate`]: the resulting vectors feed
//! [`MatryoshkaEmbedder`](crate::matryoshka) for truncatable retrieval
//! and [`MrlTier`](crate::tier) for cascade dispatch.
//!
//! Pipeline:
//!   text → BPE tokenize → encoder forward (non-causal attn) → pool
//!         (per `pooling_type`) → optional Matryoshka truncate → out
//!
//! The encoder forward goes through the same `build_llama_encoder_graph`
//! path the rest of the substrate uses (qwen3 inline / bidirectional
//! branch / ternary kernels — whichever applies for the loaded model).

use std::path::Path;

use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};
use jouleclaw_loader_gguf::llama::{build_llama_encoder_graph, LlamaConfig, LoadError};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_loader_gguf::{read_gguf_file, GgufModel};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};

/// Pooling strategy. Mirrors llama.cpp's `LLAMA_POOLING_TYPE_*`.
/// Unknown / 0 / unsupported values fall back to `Mean`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingKind {
    Mean,
    Cls,
    Last,
}

impl PoolingKind {
    pub fn from_metadata(v: u32) -> Self {
        match v {
            2 => Self::Cls,
            3 => Self::Last,
            _ => Self::Mean,
        }
    }
}

/// Errors building or running a `GgufTextEmbedder`.
#[derive(Debug)]
pub enum EmbedError {
    Parse(jouleclaw_loader_gguf::ParseError),
    Config(LoadError),
    Vocab,
    Compile(String),
    Execute(String),
    EmptyInput,
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "GGUF parse error: {:?}", e),
            Self::Config(e) => write!(f, "GGUF config error: {}", e),
            Self::Vocab => write!(f, "vocab load failed"),
            Self::Compile(s) => write!(f, "graph compile: {}", s),
            Self::Execute(s) => write!(f, "graph execute: {}", s),
            Self::EmptyInput => write!(f, "empty input after tokenisation"),
        }
    }
}

impl std::error::Error for EmbedError {}

pub struct GgufTextEmbedder {
    model: GgufModel,
    vocab: Vocab,
    config: LlamaConfig,
    pooling: PoolingKind,
}

impl GgufTextEmbedder {
    /// Load an encoder GGUF (e.g. Jina v5 omni nano) and prepare for
    /// per-call encoding.
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> Result<Self, EmbedError> {
        let model = read_gguf_file(path.as_ref()).map_err(EmbedError::Parse)?;
        let vocab = Vocab::from_gguf(&model).map_err(|_| EmbedError::Vocab)?;
        let config = LlamaConfig::from_metadata(&model).map_err(EmbedError::Config)?;
        let pooling = PoolingKind::from_metadata(config.pooling_type);
        Ok(Self { model, vocab, config, pooling })
    }

    pub fn full_dim(&self) -> usize { self.config.embedding_length }
    pub fn arch(&self) -> &str { &self.config.arch }
    pub fn pooling(&self) -> PoolingKind { self.pooling }

    /// Encode a single text into a `full_dim`-vector. Optionally
    /// truncate to a Matryoshka-valid prefix dim (caller passes
    /// `truncate_to`; pass `None` for full).
    pub fn encode(
        &self,
        text: &str,
        truncate_to: Option<usize>,
    ) -> Result<Vec<f32>, EmbedError> {
        let tokens = self.vocab.encode_bpe_regex(text, false);
        if tokens.is_empty() {
            return Err(EmbedError::EmptyInput);
        }
        let seq_len = tokens.len();

        let graph = build_llama_encoder_graph(&self.model, seq_len)
            .map_err(EmbedError::Config)?
            .graph;

        let runtime = Runtime::reference_only();
        let compiled = compile(graph, &runtime.kernels)
            .map_err(|e| EmbedError::Compile(format!("{:?}", e)))?;

        let bytes: Vec<u8> = tokens.iter()
            .flat_map(|&id| (id as i32).to_le_bytes()).collect();
        let token_tensor = Tensor {
            meta: TensorMeta::new(Dtype::I32, &[seq_len]),
            storage: std::sync::Arc::new(TensorStorage { bytes, mapped: None }),
        };
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("token_ids".to_string(), token_tensor);

        let res = execute(&compiled, inputs, ExecutionOptions::default())
            .map_err(|e| EmbedError::Execute(format!("{:?}", e)))?;
        let hidden = res.outputs.get("hidden_states")
            .ok_or_else(|| EmbedError::Execute("no hidden_states output".into()))?;
        let h = hidden.as_f32_vec();
        let d = self.full_dim();
        assert_eq!(h.len(), seq_len * d);

        // Pool [seq, d] → [d].
        let mut pooled = vec![0f32; d];
        match self.pooling {
            PoolingKind::Mean => {
                for t in 0..seq_len {
                    for c in 0..d {
                        pooled[c] += h[t * d + c];
                    }
                }
                let inv = 1.0 / seq_len as f32;
                for v in &mut pooled { *v *= inv; }
            }
            PoolingKind::Cls => {
                pooled.copy_from_slice(&h[0..d]);
            }
            PoolingKind::Last => {
                pooled.copy_from_slice(&h[(seq_len - 1) * d..seq_len * d]);
            }
        }

        // Matryoshka truncation. Per-spec the model is trained such
        // that any prefix is a valid (lower-fidelity) embedding.
        if let Some(td) = truncate_to {
            assert!(td > 0 && td <= d, "truncate_to {} out of range 1..={}", td, d);
            pooled.truncate(td);
        }

        // L2-normalise — standard for cosine retrieval.
        let norm2: f32 = pooled.iter().map(|v| v * v).sum();
        let inv = 1.0 / (norm2.sqrt().max(1e-12));
        for v in &mut pooled { *v *= inv; }

        Ok(pooled)
    }
}
