//! JouleClaw L2.5 — neural reranking tier.
//!
//! Sits between L2 federation and L3 LLM in the cascade. Takes a
//! federated candidate document set plus the original query and reorders
//! the candidates by query/document relevance, so the strongest hits
//! surface at the top **before** any L3 model burns joules on weaker
//! ones.
//!
//! ## Why a tier?
//!
//! The L2 retrieval tier optimises for *recall* (find every potentially
//! relevant document at low joule cost). The L3 model tier optimises
//! for *generation* (compose an answer at high joule cost). Between
//! them sits a gap: the top-N from L2 is rarely the ideal top-K for
//! L3 to read. A neural reranker (ColBERT, SPLADE, MiniLM-cross-encoder,
//! …) trades ~500 µJ of GPU energy for a much sharper ordering, which
//! typically cuts downstream L3 spend by 2–10× because the model sees
//! fewer, more relevant tokens.
//!
//! ## Standard surface
//!
//! This crate is **not** a reranker implementation. It is the *tier
//! adapter* that wires a consumer-supplied reranker into the JouleClaw
//! cascade. Consumers implement the [`Reranker`] trait over whatever
//! backend they like — candle ColBERT, ONNX SPLADE, llama.cpp
//! cross-encoder, a vendor API — and hand it to [`RerankTier`].
//!
//! For reference / smoke tests / Pi-class deployments where the
//! reranker budget is "effectively zero", the crate ships a default
//! [`Bm25Reranker`]: pure deterministic BM25 over the query and document
//! text. No neural deps, no model weights, ~10 µJ per document.
//!
//! ## Query envelope
//!
//! The tier consumes [`QueryInput::Structured`] with a canonical JSON
//! envelope:
//!
//! ```json
//! {
//!   "query": "what is the capital of france",
//!   "docs": [
//!     { "id": "doc-1", "text": "Paris is the capital of France." },
//!     { "id": "doc-2", "text": "Berlin is the capital of Germany." }
//!   ],
//!   "top_k": 10
//! }
//! ```
//!
//! and emits a `Structured` response listing the reranked candidates
//! in descending score order, truncated to `top_k`.

#![forbid(unsafe_code)]

pub mod bm25;
pub mod reranker;
pub mod tier;

pub use bm25::{Bm25Params, Bm25Reranker};
pub use reranker::{Doc, RerankError, RerankScore, Reranker};
pub use tier::{
    RerankEnvelope, RerankOutput, RerankTier, RERANK_CONFIDENCE_FLOOR,
    RERANK_JOULES, RERANK_LATENCY,
};
