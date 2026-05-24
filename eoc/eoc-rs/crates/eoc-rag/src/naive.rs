//! Naive RAG — embed query, top-K, stuff into the prompt.
//!
//! The original Lewis et al. 2020 recipe. Useful as a baseline; the
//! later pipelines in this crate add rewriting, fusion, evaluation,
//! and self-checking on top.

use std::sync::Arc;

use async_trait::async_trait;
use eoc_core::JouleCost;

use crate::citation::{CitationEnforcement, CitationPolicy, derive_citations};
use crate::error::{RagError, RagResult};
use crate::pipeline::{Answer, Pipeline, RagRequest, Stage, Trace, TraceEvent};
use crate::store::DocumentStore;

/// Naive RAG pipeline.
pub struct NaivePipeline {
    /// The store to retrieve from.
    pub store: Arc<dyn DocumentStore>,
    /// Citation policy.
    pub citation_policy: CitationPolicy,
    /// Estimated joules per retrieval call.
    pub retrieve_microjoules: u64,
    /// Estimated joules per generation call.
    pub generate_microjoules: u64,
}

impl NaivePipeline {
    /// Construct.
    pub fn new(store: Arc<dyn DocumentStore>) -> Self {
        Self {
            store,
            citation_policy: CitationPolicy::Required,
            retrieve_microjoules: 5_000,
            generate_microjoules: 50_000,
        }
    }

    /// Override citation policy (consumes `self`).
    pub fn with_citation_policy(mut self, policy: CitationPolicy) -> Self {
        self.citation_policy = policy;
        self
    }
}

#[async_trait]
impl Pipeline for NaivePipeline {
    async fn answer(&self, req: &RagRequest) -> RagResult<Answer> {
        if req.top_k == 0 {
            return Err(RagError::Config("top_k must be >= 1".into()));
        }
        let mut trace = Trace::new();

        let chunks = self.store.retrieve(&req.query, req.top_k).await?;
        trace.record(TraceEvent::new(
            Stage::Retrieve,
            JouleCost::estimated(self.retrieve_microjoules),
            format!("retrieved {} chunks", chunks.len()),
        ));
        if chunks.is_empty() {
            return Err(RagError::NoChunks);
        }

        // "Generation" — the reference pipeline returns a deterministic
        // stuffed answer (the highest-ranked chunk). Vendor-API
        // backends override this stage.
        let answer_text = chunks[0].chunk.text.clone();
        trace.record(TraceEvent::new(
            Stage::Generate,
            JouleCost::estimated(self.generate_microjoules),
            "naive stuffed generation",
        ));

        let citations = derive_citations(&answer_text, &chunks);
        let gate = CitationEnforcement::new(self.citation_policy);
        gate.enforce(&answer_text, &chunks, &citations)?;

        Ok(Answer::new(answer_text, chunks, trace).with_citations(citations))
    }

    fn name(&self) -> &str {
        "naive"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Chunk, InMemoryStore};

    fn store() -> Arc<dyn DocumentStore> {
        Arc::new(InMemoryStore::from_chunks(
            "test",
            vec![
                Chunk::new("d1", 0, "joules per byte is the EOC efficiency metric"),
                Chunk::new("d2", 0, "bicycles are a clean transport mode"),
            ],
        ))
    }

    #[tokio::test]
    async fn naive_returns_top_chunk() {
        let p = NaivePipeline::new(store()).with_citation_policy(CitationPolicy::Optional);
        let req = RagRequest::new("EOC efficiency metric", 2);
        let ans = p.answer(&req).await.expect("ok");
        assert!(ans.text.contains("joules"));
        assert!(ans.trace.total_microjoules() > 0);
    }
}
