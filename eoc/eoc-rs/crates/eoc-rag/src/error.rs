//! Errors raised by the RAG orchestrator.

use thiserror::Error;

/// RAG pipeline error.
#[derive(Debug, Error)]
pub enum RagError {
    /// The document store could not be queried.
    #[error("store error: {0}")]
    Store(String),

    /// An embedder backend failed.
    #[error("embedder error: {0}")]
    Embedder(String),

    /// A generator (LLM) backend failed.
    #[error("generator error: {0}")]
    Generator(String),

    /// A re-ranker backend failed.
    #[error("reranker error: {0}")]
    Reranker(String),

    /// The configuration is invalid (e.g. `top_k == 0`).
    #[error("invalid config: {0}")]
    Config(String),

    /// The pipeline produced an answer with no citation when one was
    /// required by policy.
    #[error("citation required but answer carries none")]
    CitationRequired,

    /// The pipeline produced an answer that failed the consistency
    /// guard.
    #[error("hallucination guard rejected answer: {0}")]
    GuardRejected(String),

    /// Retrieval returned zero chunks where at least one was required.
    #[error("no chunks retrieved for query")]
    NoChunks,
}

/// Convenience alias.
pub type RagResult<T> = std::result::Result<T, RagError>;
