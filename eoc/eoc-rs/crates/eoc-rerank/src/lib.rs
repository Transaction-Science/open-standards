//! EOC re-ranking and hybrid retrieval.
//!
//! Production RAG systems combine three retrieval signals:
//!
//! 1. **Dense retrieval** — cosine / inner-product search over an embedding
//!    space (see [`eoc_embeddings`]). Strong on semantic paraphrase, weak
//!    on rare-token / exact-match queries.
//! 2. **Sparse retrieval** — Okapi BM25 over an inverted index (see
//!    [`bm25`]) or learned-sparse models like SPLADE (see [`sparse_lex`]).
//!    Strong on rare tokens and exact matches.
//! 3. **Cross-encoder re-ranking** — a more accurate but slower scorer
//!    that jointly encodes `(query, document)` pairs. Used as a second
//!    stage on the top-K candidate set from the retriever.
//!
//! [`hybrid::HybridRetriever`] fuses dense and sparse rankings with
//! Reciprocal Rank Fusion (RRF) or min-max normalisation.
//! [`pipeline::RetrievalPipeline`] wires retrieve-then-rerank end to end,
//! the canonical RAG flow.
//!
//! The EOC cascade's KV stage calls into this crate via
//! [`cascade_integration`] when the cosine-similarity match lands below
//! a configurable threshold — the cross-encoder gets a chance to recover
//! candidates the embedder ranked low.
//!
//! ## Backends
//!
//! Re-ranker backends:
//!
//! * **Cohere** — `rerank-english-v3.0`, `rerank-multilingual-v3.0`,
//!   `rerank-v3.5`. See [`cohere_rerank`].
//! * **Voyage** — `rerank-2`, `rerank-2-lite`. See [`voyage_rerank`].
//! * **bge-reranker** (local, feature `local`) — `bge-reranker-base`,
//!   `bge-reranker-large`, `bge-reranker-v2-m3`. See [`bge_rerank`].
//! * **ms-marco MiniLM** (local, feature `local`) — see
//!   [`ms_marco_rerank`].
//! * **ColBERT v2** (local, feature `local`) — multi-vector late-
//!   interaction. See [`colbert`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bm25;
pub mod cascade_integration;
pub mod cohere_rerank;
pub mod dense;
pub mod error;
pub mod hybrid;
pub mod pipeline;
pub mod reranker;
pub mod sparse_lex;
pub mod voyage_rerank;

#[cfg(feature = "local")]
pub mod bge_rerank;
#[cfg(feature = "local")]
pub mod colbert;
#[cfg(feature = "local")]
pub mod ms_marco_rerank;

pub use bm25::{Bm25Config, Bm25Index, Document};
pub use cohere_rerank::CohereReranker;
pub use dense::DenseIndex;
pub use error::{RerankError, RerankResult};
pub use hybrid::{FusionMethod, HybridRetriever};
pub use pipeline::RetrievalPipeline;
pub use reranker::{Candidate, Reranker, Retriever, ScoredCandidate};
pub use sparse_lex::{SparseLexIndex, SparseVector};
pub use voyage_rerank::VoyageReranker;

#[cfg(feature = "local")]
pub use bge_rerank::BgeReranker;
#[cfg(feature = "local")]
pub use colbert::ColBertIndex;
#[cfg(feature = "local")]
pub use ms_marco_rerank::MsMarcoReranker;

/// Canonical document identifier — caller-defined opaque string.
pub type DocId = String;
