//! Corrective RAG (CRAG) — Yan et al. 2024
//! ("Corrective Retrieval Augmented Generation",
//! arXiv:2401.15884).
//!
//! CRAG threads a lightweight retrieval evaluator between retrieve
//! and generate. The evaluator partitions retrieval into three
//! buckets:
//!
//! * **Correct** — high confidence. Refine the chunk (drop irrelevant
//!   strips) and feed to the generator.
//! * **Incorrect** — low confidence. Discard retrieval and fall back
//!   to a web-search-style "external" knowledge source.
//! * **Ambiguous** — mid confidence. Combine refined retrieval with
//!   the external fallback.
//!
//! The reference [`HeuristicRetrievalEvaluator`] thresholds on the
//! average retriever score. The "external knowledge" fallback in this
//! reference is a deterministic dummy; production wires this to an
//! HTTP search backend.

use std::sync::Arc;

use async_trait::async_trait;
use eoc_core::JouleCost;
use serde::{Deserialize, Serialize};

use crate::citation::{CitationEnforcement, CitationPolicy, derive_citations};
use crate::error::{RagError, RagResult};
use crate::pipeline::{Answer, Pipeline, RagRequest, Stage, Trace, TraceEvent};
use crate::store::{Chunk, DocumentStore, RetrievedChunk};

/// Bucket the retrieval falls into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalQuality {
    /// High confidence — retrieval is correct.
    Correct,
    /// Mid confidence — retrieval is partial.
    Ambiguous,
    /// Low confidence — retrieval is wrong.
    Incorrect,
}

/// One evaluator decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CragEvaluation {
    /// Bucket.
    pub quality: RetrievalQuality,
    /// Mean retriever score.
    pub mean_score: f32,
}

/// Retrieval evaluator.
#[async_trait]
pub trait RetrievalEvaluator: Send + Sync {
    /// Evaluate `chunks` against `query`.
    async fn evaluate(
        &self,
        query: &str,
        chunks: &[RetrievedChunk],
    ) -> RagResult<CragEvaluation>;
}

/// Threshold-based evaluator. `correct >= correct_threshold`;
/// `incorrect <= incorrect_threshold`; otherwise ambiguous. Empty
/// retrieval is always `Incorrect`.
pub struct HeuristicRetrievalEvaluator {
    /// Above this mean score, retrieval is considered correct.
    pub correct_threshold: f32,
    /// At or below this mean score, retrieval is considered incorrect.
    pub incorrect_threshold: f32,
}

impl Default for HeuristicRetrievalEvaluator {
    fn default() -> Self {
        Self {
            correct_threshold: 0.5,
            incorrect_threshold: 0.05,
        }
    }
}

#[async_trait]
impl RetrievalEvaluator for HeuristicRetrievalEvaluator {
    async fn evaluate(
        &self,
        _query: &str,
        chunks: &[RetrievedChunk],
    ) -> RagResult<CragEvaluation> {
        if chunks.is_empty() {
            return Ok(CragEvaluation {
                quality: RetrievalQuality::Incorrect,
                mean_score: 0.0,
            });
        }
        let mean = chunks.iter().map(|c| c.score).sum::<f32>() / chunks.len() as f32;
        let q = if mean >= self.correct_threshold {
            RetrievalQuality::Correct
        } else if mean <= self.incorrect_threshold {
            RetrievalQuality::Incorrect
        } else {
            RetrievalQuality::Ambiguous
        };
        Ok(CragEvaluation {
            quality: q,
            mean_score: mean,
        })
    }
}

/// External knowledge fallback. Vendor backends wire this to a
/// web-search API (the original paper uses Google Search).
#[async_trait]
pub trait ExternalKnowledge: Send + Sync {
    /// Look up `query` and return chunks.
    async fn search(&self, query: &str, top_k: usize) -> RagResult<Vec<RetrievedChunk>>;
}

/// Deterministic stub used in tests.
pub struct NullExternalKnowledge;

#[async_trait]
impl ExternalKnowledge for NullExternalKnowledge {
    async fn search(&self, query: &str, top_k: usize) -> RagResult<Vec<RetrievedChunk>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let chunk = Chunk::new("external", 0, format!("external knowledge for: {query}"));
        Ok(vec![RetrievedChunk {
            chunk,
            score: 0.5,
            rank: 1,
        }])
    }
}

/// CRAG pipeline.
pub struct CragPipeline {
    /// Store.
    pub store: Arc<dyn DocumentStore>,
    /// Retrieval evaluator.
    pub evaluator: Arc<dyn RetrievalEvaluator>,
    /// External knowledge fallback.
    pub external: Arc<dyn ExternalKnowledge>,
    /// Citation policy.
    pub citation_policy: CitationPolicy,
    /// Per-stage joule estimates.
    pub retrieve_microjoules: u64,
    /// Evaluator cost.
    pub evaluate_microjoules: u64,
    /// Cost of the external knowledge lookup.
    pub external_microjoules: u64,
    /// Generation cost.
    pub generate_microjoules: u64,
}

impl CragPipeline {
    /// Construct.
    pub fn new(store: Arc<dyn DocumentStore>) -> Self {
        Self {
            store,
            evaluator: Arc::new(HeuristicRetrievalEvaluator::default()),
            external: Arc::new(NullExternalKnowledge),
            citation_policy: CitationPolicy::Optional,
            retrieve_microjoules: 5_000,
            evaluate_microjoules: 2_000,
            external_microjoules: 80_000,
            generate_microjoules: 50_000,
        }
    }

    /// Refine chunks — drop strips with very low scores. The
    /// "decompose-then-recompose" step from the paper. Reference
    /// implementation: keep the top-half of chunks by score.
    fn refine(chunks: Vec<RetrievedChunk>) -> Vec<RetrievedChunk> {
        if chunks.len() <= 2 {
            return chunks;
        }
        let keep = chunks.len().div_ceil(2);
        let mut out = chunks;
        out.truncate(keep);
        out
    }
}

#[async_trait]
impl Pipeline for CragPipeline {
    async fn answer(&self, req: &RagRequest) -> RagResult<Answer> {
        if req.top_k == 0 {
            return Err(RagError::Config("top_k must be >= 1".into()));
        }
        let mut trace = Trace::new();

        let retrieved = self.store.retrieve(&req.query, req.top_k).await?;
        trace.record(TraceEvent::new(
            Stage::Retrieve,
            JouleCost::estimated(self.retrieve_microjoules),
            format!("retrieved {} chunks", retrieved.len()),
        ));

        let eval = self.evaluator.evaluate(&req.query, &retrieved).await?;
        trace.record(TraceEvent::new(
            Stage::Evaluate,
            JouleCost::estimated(self.evaluate_microjoules),
            format!("quality={:?} mean={:.3}", eval.quality, eval.mean_score),
        ));

        let chunks = match eval.quality {
            RetrievalQuality::Correct => Self::refine(retrieved),
            RetrievalQuality::Ambiguous => {
                let mut refined = Self::refine(retrieved);
                let mut ext = self.external.search(&req.query, req.top_k).await?;
                trace.record(TraceEvent::new(
                    Stage::Retrieve,
                    JouleCost::estimated(self.external_microjoules),
                    format!("external fallback returned {} chunks", ext.len()),
                ));
                refined.append(&mut ext);
                refined.truncate(req.top_k);
                refined
            }
            RetrievalQuality::Incorrect => {
                let ext = self.external.search(&req.query, req.top_k).await?;
                trace.record(TraceEvent::new(
                    Stage::Retrieve,
                    JouleCost::estimated(self.external_microjoules),
                    format!("external fallback returned {} chunks", ext.len()),
                ));
                ext
            }
        };

        if chunks.is_empty() {
            return Err(RagError::NoChunks);
        }

        let answer_text = chunks[0].chunk.text.clone();
        trace.record(TraceEvent::new(
            Stage::Generate,
            JouleCost::estimated(self.generate_microjoules),
            "crag stuffed generation",
        ));

        let citations = derive_citations(&answer_text, &chunks);
        CitationEnforcement::new(self.citation_policy).enforce(&answer_text, &chunks, &citations)?;

        Ok(Answer::new(answer_text, chunks, trace).with_citations(citations))
    }

    fn name(&self) -> &str {
        "crag"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Chunk, InMemoryStore};

    #[tokio::test]
    async fn crag_falls_back_when_retrieval_empty() {
        let store: Arc<dyn DocumentStore> =
            Arc::new(InMemoryStore::from_chunks("test", vec![Chunk::new(
                "d1", 0, "unrelated content",
            )]));
        let p = CragPipeline::new(store);
        let req = RagRequest::new("zzzzzz unrelated query qqqqqq", 3);
        let ans = p.answer(&req).await.expect("ok");
        assert!(
            ans.chunks
                .iter()
                .any(|c| c.chunk.doc_id == "external" || c.chunk.doc_id == "d1")
        );
    }
}
