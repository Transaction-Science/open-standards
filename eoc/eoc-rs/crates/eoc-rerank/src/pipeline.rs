//! Retrieve → re-rank → return — the canonical RAG retrieval flow.
//!
//! ```text
//! query
//!   │
//!   ▼
//! retriever (hybrid: dense + sparse, top-K = 50..200)
//!   │
//!   ▼
//! reranker  (cross-encoder, scores all pairs, returns top-N)
//!   │
//!   ▼
//! context-window assembly  (top-N, N typically 5..10)
//! ```
//!
//! [`RetrievalPipeline`] composes any [`Retriever`] with an optional
//! [`Reranker`]. If the re-ranker is absent, the pipeline degrades
//! gracefully to "retrieve and trim".

use std::sync::Arc;

use crate::error::RerankResult;
use crate::reranker::{Candidate, Reranker, Retriever, ScoredCandidate};

/// End-to-end retrieve-then-rerank pipeline.
pub struct RetrievalPipeline {
    /// Retriever — typically [`crate::hybrid::HybridRetriever`].
    pub retriever: Arc<dyn Retriever>,
    /// Optional cross-encoder re-ranker.
    pub reranker: Option<Arc<dyn Reranker>>,
    /// Initial retrieval depth. Typical 50-200.
    pub top_k_retrieval: usize,
    /// Final result depth after re-ranking. Typical 5-20.
    pub top_k_rerank: usize,
}

impl RetrievalPipeline {
    /// Construct a pipeline.
    pub fn new(
        retriever: Arc<dyn Retriever>,
        reranker: Option<Arc<dyn Reranker>>,
        top_k_retrieval: usize,
        top_k_rerank: usize,
    ) -> Self {
        Self {
            retriever,
            reranker,
            top_k_retrieval,
            top_k_rerank,
        }
    }

    /// Run the full pipeline.
    pub async fn search(&self, query: &str) -> RerankResult<Vec<ScoredCandidate>> {
        let hits = self.retriever.retrieve(query, self.top_k_retrieval).await?;
        // Materialise candidate text.
        let candidates: Vec<Candidate> = hits
            .iter()
            .filter_map(|(id, _)| {
                self.retriever
                    .document_text(id)
                    .map(|text| Candidate::new(id.clone(), text))
            })
            .collect();

        if let Some(rr) = &self.reranker {
            let mut scored = rr.rerank(query, &candidates).await?;
            scored.truncate(self.top_k_rerank);
            return Ok(scored);
        }

        // No re-ranker — promote retriever scores directly.
        let mut out: Vec<ScoredCandidate> = candidates
            .into_iter()
            .zip(hits)
            .map(|(c, (_, score))| ScoredCandidate {
                candidate: c,
                score,
                rank: 0,
            })
            .collect();
        out.truncate(self.top_k_rerank);
        for (i, c) in out.iter_mut().enumerate() {
            c.rank = i + 1;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    use crate::DocId;

    struct StaticRetriever {
        docs: Vec<(DocId, String, f32)>,
    }

    #[async_trait]
    impl Retriever for StaticRetriever {
        async fn retrieve(&self, _query: &str, top_k: usize) -> RerankResult<Vec<(DocId, f32)>> {
            let mut v: Vec<(DocId, f32)> =
                self.docs.iter().map(|(id, _, s)| (id.clone(), *s)).collect();
            v.truncate(top_k);
            Ok(v)
        }
        fn document_text(&self, id: &DocId) -> Option<String> {
            self.docs.iter().find(|(d, _, _)| d == id).map(|(_, t, _)| t.clone())
        }
        fn name(&self) -> &str {
            "static"
        }
    }

    /// A re-ranker that puts `prefer` at rank 1 regardless of input order.
    struct ConstantPreferReranker {
        prefer: DocId,
    }

    #[async_trait]
    impl Reranker for ConstantPreferReranker {
        async fn rerank(
            &self,
            _query: &str,
            candidates: &[Candidate],
        ) -> RerankResult<Vec<ScoredCandidate>> {
            let mut out: Vec<ScoredCandidate> = candidates
                .iter()
                .map(|c| ScoredCandidate {
                    candidate: c.clone(),
                    score: if c.id == self.prefer { 100.0 } else { -1.0 },
                    rank: 0,
                })
                .collect();
            out.sort_by(|a, b| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (i, c) in out.iter_mut().enumerate() {
                c.rank = i + 1;
            }
            Ok(out)
        }
        fn model_name(&self) -> &str {
            "constant-prefer"
        }
        fn max_pairs(&self) -> usize {
            10_000
        }
    }

    #[tokio::test]
    async fn pipeline_retrieve_then_rerank_promotes_correct_top1() {
        // 50 docs; the retriever ranks "buried" 49th but the reranker
        // promotes it to rank 1.
        let mut docs: Vec<(DocId, String, f32)> = (0..50)
            .map(|i| {
                (
                    format!("d{i}"),
                    format!("body {i}"),
                    50.0 - i as f32, // d0 has the highest retriever score.
                )
            })
            .collect();
        docs[49] = ("buried".to_string(), "the actual answer".to_string(), 0.1);
        let retriever = Arc::new(StaticRetriever { docs });
        let reranker = Arc::new(ConstantPreferReranker {
            prefer: "buried".to_string(),
        });

        let pipe = RetrievalPipeline::new(retriever, Some(reranker), 50, 10);
        let result = pipe.search("anything").await.expect("ok");
        assert_eq!(result[0].candidate.id, "buried");
        assert_eq!(result[0].rank, 1);
        assert!(result.len() <= 10);
    }

    #[tokio::test]
    async fn pipeline_without_reranker_passes_through() {
        let docs: Vec<(DocId, String, f32)> = (0..5)
            .map(|i| (format!("d{i}"), format!("body {i}"), (5 - i) as f32))
            .collect();
        let retriever = Arc::new(StaticRetriever { docs });
        let pipe = RetrievalPipeline::new(retriever, None, 5, 3);
        let result = pipe.search("query").await.expect("ok");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].candidate.id, "d0");
    }
}
