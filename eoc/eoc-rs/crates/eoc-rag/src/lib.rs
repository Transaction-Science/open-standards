//! EOC RAG orchestration.
//!
//! `eoc-rag` is the retrieval-augmented generation layer of the EOC
//! cascade. It wires together the strategies that have appeared in the
//! literature since the original "Retrieval-Augmented Generation for
//! Knowledge-Intensive NLP Tasks" (Lewis et al. 2020):
//!
//! * **Naive RAG** — embed query, top-K, stuff into the prompt
//!   ([`naive`]).
//! * **HyDE** — Hypothetical Document Embeddings, Gao et al. 2022
//!   ([`hyde`]).
//! * **RAG-Fusion** — multi-query expansion fused with Reciprocal Rank
//!   Fusion, Cormack et al. 2009 ([`fusion`]).
//! * **Self-RAG** — critique-token-driven self-reflective retrieval,
//!   Asai et al. 2023 ([`self_rag`]).
//! * **Corrective RAG (CRAG)** — retrieval evaluator triggers
//!   refinement or fallback, Yan et al. 2024 ([`crag`]).
//! * **Adaptive RAG** — classifier routes between no-retrieval,
//!   single-step, and multi-step pipelines, Jeong et al. 2024
//!   ([`adaptive`]).
//! * **Query rewriting** — multi-query, step-back, decomposition
//!   ([`rewrite`]).
//! * **Chunking** — sentence-window, semantic, recursive, late
//!   ([`chunk`]).
//! * **Citation/provenance** — span-backed citations ([`citation`]).
//! * **Hallucination guard** — SelfCheckGPT-style consistency
//!   ([`guard`], Manakul et al. 2023).
//!
//! ## Joule accounting
//!
//! Every stage records a [`eoc_core::JouleCost`] in the [`pipeline::Trace`]
//! so the resulting RAG call can be priced against the EOC budget the
//! same way as a cascade hit.
//!
//! ## Determinism
//!
//! Everything in this crate is deterministic given a fixed
//! [`store::DocumentStore`]. Vendor-API integrations (OpenAI / Cohere /
//! Voyage embedders, cross-encoder rerankers) live in `eoc-embeddings`
//! and `eoc-rerank` and plug in through the traits exposed here.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod adaptive;
pub mod chunk;
pub mod citation;
pub mod crag;
pub mod error;
pub mod fusion;
pub mod guard;
pub mod hyde;
pub mod naive;
pub mod pipeline;
pub mod rewrite;
pub mod self_rag;
pub mod store;

pub use adaptive::{AdaptiveRouter, Complexity};
pub use chunk::{Chunker, ChunkerKind, ChunkingConfig};
pub use citation::{CitationEnforcement, CitationPolicy, Cite};
pub use crag::{CragEvaluation, CragPipeline, RetrievalQuality};
pub use error::{RagError, RagResult};
pub use fusion::{RagFusionPipeline, reciprocal_rank_fusion};
pub use guard::{ConsistencyVerdict, SelfCheckGuard};
pub use hyde::{HydeGenerator, HydePipeline};
pub use naive::NaivePipeline;
pub use pipeline::{Answer, Pipeline, RagRequest, Stage, Trace, TraceEvent};
pub use rewrite::{QueryRewriter, RewriteKind};
pub use self_rag::{Critique, CritiqueToken, SelfRagPipeline};
pub use store::{Chunk, ChunkId, DocumentStore, InMemoryStore, RetrievedChunk};
