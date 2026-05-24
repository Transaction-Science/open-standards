//! Shared traits and value types for retrievers and re-rankers.
//!
//! [`Retriever`] is the first-stage interface (cheap, recall-oriented).
//! [`Reranker`] is the second-stage interface (slower, precision-oriented).
//! [`Candidate`] and [`ScoredCandidate`] are the value types exchanged
//! between them.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::DocId;
use crate::error::RerankResult;

/// A candidate document — `(id, text)` — fed to a re-ranker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    /// Caller-defined document identifier.
    pub id: DocId,
    /// The document text the re-ranker will see.
    pub text: String,
}

impl Candidate {
    /// Construct a candidate.
    pub fn new(id: impl Into<DocId>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

/// A candidate with a relevance score (higher = better).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredCandidate {
    /// The candidate.
    pub candidate: Candidate,
    /// Re-ranker score.
    pub score: f32,
    /// Re-ranker rank (1-based).
    pub rank: usize,
}

/// A first-stage retriever — returns top-K `(doc_id, score)` candidates
/// from some index. Implementations include [`crate::bm25::Bm25Index`],
/// [`crate::dense::DenseIndex`], and [`crate::hybrid::HybridRetriever`].
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Retrieve up to `top_k` candidates for `query`.
    async fn retrieve(&self, query: &str, top_k: usize) -> RerankResult<Vec<(DocId, f32)>>;

    /// Look up a document's text by id. Used by
    /// [`crate::pipeline::RetrievalPipeline`] to assemble candidates for
    /// the re-ranker.
    fn document_text(&self, id: &DocId) -> Option<String>;

    /// Human-readable retriever name.
    fn name(&self) -> &str;
}

/// A second-stage cross-encoder re-ranker.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Score and sort `candidates` by `query`-relevance. Returns the
    /// candidates re-ordered with `score` and 1-based `rank` populated.
    async fn rerank(
        &self,
        query: &str,
        candidates: &[Candidate],
    ) -> RerankResult<Vec<ScoredCandidate>>;

    /// Canonical model identifier.
    fn model_name(&self) -> &str;

    /// Maximum number of `(query, document)` pairs in a single call.
    fn max_pairs(&self) -> usize;
}
