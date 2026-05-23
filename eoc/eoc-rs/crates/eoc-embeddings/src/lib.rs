//! EOC embedding-model backends.
//!
//! This crate ships a uniform [`Embedder`] trait and concrete backends for
//! the major commercial and open-source embedding model families:
//!
//! * **OpenAI v3** — `text-embedding-3-small` / `text-embedding-3-large`
//!   (plus legacy `text-embedding-ada-002`).
//! * **Cohere v3** — `embed-english-v3.0`, `embed-multilingual-v3.0`,
//!   `embed-english-light-v3.0`.
//! * **Voyage** — `voyage-3`, `voyage-3-lite`, `voyage-code-3`,
//!   `voyage-finance-2`, `voyage-law-2`.
//! * **Mistral** — `mistral-embed`.
//! * **Jina** — `jina-embeddings-v3`, `jina-embeddings-v2-base-code`.
//! * **Local ONNX** — BGE family (`bge-large-en-v1.5`, `bge-m3`,
//!   `bge-small-en-v1.5`), `nomic-embed-text-v1.5`, `mxbai-embed-large-v1`,
//!   `GTE`, `E5`. Behind the `local` feature.
//!
//! The KV stage in [`eoc_kv`] uses cosine similarity on embeddings to hit
//! cached responses; this crate makes that work out-of-the-box without the
//! caller bringing their own vectors.
//!
//! ## Dimension mismatch
//!
//! Mixing embedders across the storage/query boundary produces vectors of
//! incompatible dimensions. [`dimension_alignment`] exposes
//! Matryoshka-style truncation for models that support it (OpenAI v3,
//! nomic-embed, mxbai) and zero-padding for the rest (with documented
//! lossiness).
//!
//! ## WASM
//!
//! Vendor backends compile to `wasm32-unknown-unknown` with `reqwest`
//! limited to same-origin `fetch`. The `local` feature uses `ort` and
//! does **not** compile to WASM.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cache;
pub mod cohere;
pub mod dimension_alignment;
pub mod embedder;
pub mod error;
pub mod jina;
pub mod joule_estimator;
pub mod mistral;
pub mod openai;
pub mod voyage;

#[cfg(feature = "local")]
pub mod local;

// Re-export error module path used by integration tests.

pub use cache::{ContentHash, EmbeddingCache};
pub use cohere::CohereEmbedder;
pub use dimension_alignment::{project, requires_alignment};
pub use embedder::Embedder;
pub use error::{EmbeddingError, EmbeddingResult};
pub use jina::JinaEmbedder;
pub use joule_estimator::{EmbeddingEnergyProfile, JouleEstimator};
pub use mistral::MistralEmbedder;
pub use openai::OpenAiEmbedder;
pub use voyage::VoyageEmbedder;

#[cfg(feature = "local")]
pub use local::LocalEmbedder;
