//! DeBERTa-v3 transformer family for the Edge-First Architecture
//! (spec §6.4: entailment via DeBERTa-v3 NLI).
//!
//! The model the spec names is
//! `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, an
//! NLI head on top of `microsoft/deberta-v3-large`. We port the
//! architecture in stages and verify each stage against
//! HuggingFace's reference logits (see `scripts/hf_reference_*.py`).
//!
//! Stages (committed incrementally):
//!
//! 1. [`config`] + [`loader`] — safetensors weight inventory matches
//!    the documented architecture; load weights into typed
//!    [`weights::Weights`].
//! 2. [`tokenizer`] — DeBERTaV2 SentencePiece tokenizer behind a
//!    `Tokenizer` wrapper.
//! 3. Embedding layer + per-layer forward pass with disentangled
//!    attention (`attention`, `forward`).
//! 4. NLI head + the public [`nli::NliInference`] trait that
//!    jouleclaw-diagnose consumes.

pub mod attention;
pub mod config;
pub mod embedding;
pub mod engine;
pub mod forward;
pub mod forward_batch;
pub mod int8;
pub mod loader;
pub mod nli;
pub mod tensor_ops;
pub mod tokenizer;
pub mod weights;

pub use attention::{
    build_relative_position, forward_attention, layer_norm_rel_embeddings,
    make_log_bucket_position,
};
pub use config::{ModelConfig, NliLabel, NliLabelLayout};
pub use engine::{EngineError, NliEngine};
pub use embedding::forward_embedding;
pub use forward::{forward, forward_encoder, forward_ffn, forward_head, ForwardError, ForwardResult};
pub use forward_batch::forward_batch;
pub use int8::{matmul_q8, quantize_activation_per_row, Int8Linear};
pub use loader::{LoaderError, ModelFiles, ModelInventory};
pub use nli::{NliInference, NliInferenceError, NliPrediction};
pub use tokenizer::{DebertaTokenizer, Encoded, TokenizerError};
pub use weights::{
    AttentionWeights, ClassificationHead, EmbeddingWeights, EncoderWeights, FfnWeights,
    FloatTensor, LayerWeights, Weights,
};
