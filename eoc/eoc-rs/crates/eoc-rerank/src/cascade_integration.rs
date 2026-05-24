//! Glue between the EOC cascade KV stage and the rerank pipeline.
//!
//! The KV stage in [`eoc_kv`] resolves a query with a cosine-similarity
//! hit when it scores above [`KvConfig::similarity_threshold`]. Below that
//! threshold the cascade falls through to graph / neural — but often the
//! correct answer is in the KV's embedding store and was just ranked low
//! by single-vector cosine. This module wires a [`RetrievalPipeline`] in
//! to give the cross-encoder a chance to recover it.
//!
//! Usage:
//!
//! ```ignore
//! let pipeline = Arc::new(RetrievalPipeline::new(/* ... */));
//! let recovery = RerankRecovery::new(pipeline, 0.85);
//! // call from the KV stage when the best cosine sim is in [0.85, threshold).
//! if let Some(top) = recovery.maybe_recover(&query.prompt, sim).await? { /* ... */ }
//! ```
//!
//! The recovery layer is *additive* — it never blocks; if the re-rank
//! pipeline fails for any reason (vendor outage, model file missing) the
//! caller falls through to the next cascade stage as normal.

use std::sync::Arc;

use crate::error::RerankResult;
use crate::pipeline::RetrievalPipeline;
use crate::reranker::ScoredCandidate;

/// Recovery shim: invoke the rerank pipeline when the KV cosine match
/// score sits between [`min_recovery`] and the KV's accept threshold.
pub struct RerankRecovery {
    /// The full retrieve-then-rerank pipeline.
    pub pipeline: Arc<RetrievalPipeline>,
    /// Lower bound on the KV cosine match. Below this the KV stage is
    /// hopeless — fall through to the next cascade stage immediately.
    pub min_recovery: f32,
    /// Minimum cross-encoder score for the recovered top-1 to be accepted.
    pub min_rerank_accept: f32,
}

impl RerankRecovery {
    /// Construct a recovery shim. `min_recovery` is the lower bound on the
    /// KV cosine score; below it, recovery is skipped.
    pub fn new(pipeline: Arc<RetrievalPipeline>, min_recovery: f32) -> Self {
        Self {
            pipeline,
            min_recovery,
            min_rerank_accept: 0.0,
        }
    }

    /// Override the minimum cross-encoder score for acceptance.
    pub fn with_min_rerank_accept(mut self, v: f32) -> Self {
        self.min_rerank_accept = v;
        self
    }

    /// Attempt recovery. Returns `Ok(None)` when the KV score is below
    /// `min_recovery` (cheap short-circuit) or when the reranker's top
    /// candidate's score is below `min_rerank_accept`.
    pub async fn maybe_recover(
        &self,
        query: &str,
        kv_cosine: f32,
    ) -> RerankResult<Option<ScoredCandidate>> {
        if kv_cosine < self.min_recovery {
            return Ok(None);
        }
        let mut results = self.pipeline.search(query).await?;
        if results.is_empty() {
            return Ok(None);
        }
        let top = results.remove(0);
        if top.score < self.min_rerank_accept {
            return Ok(None);
        }
        Ok(Some(top))
    }

    /// Whether `kv_cosine` falls into the band where recovery should run.
    pub fn should_attempt(&self, kv_cosine: f32, kv_accept_threshold: f32) -> bool {
        kv_cosine >= self.min_recovery && kv_cosine < kv_accept_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    use crate::DocId;
    use crate::reranker::{Candidate, Retriever, ScoredCandidate};

    struct StubRetriever {
        text: String,
    }

    #[async_trait]
    impl Retriever for StubRetriever {
        async fn retrieve(&self, _q: &str, _k: usize) -> RerankResult<Vec<(DocId, f32)>> {
            Ok(vec![("d1".into(), 0.5)])
        }
        fn document_text(&self, _id: &DocId) -> Option<String> {
            Some(self.text.clone())
        }
        fn name(&self) -> &str {
            "stub"
        }
    }

    #[tokio::test]
    async fn skips_recovery_below_min() {
        let pipe = Arc::new(RetrievalPipeline::new(
            Arc::new(StubRetriever {
                text: "doc text".into(),
            }),
            None,
            10,
            3,
        ));
        let rec = RerankRecovery::new(pipe, 0.85);
        assert!(rec.maybe_recover("q", 0.4).await.expect("ok").is_none());
    }

    #[tokio::test]
    async fn recovers_in_band() {
        let pipe = Arc::new(RetrievalPipeline::new(
            Arc::new(StubRetriever {
                text: "doc text".into(),
            }),
            None,
            10,
            3,
        ));
        let rec = RerankRecovery::new(pipe, 0.85);
        let r = rec.maybe_recover("q", 0.9).await.expect("ok");
        assert!(r.is_some());
        let r = r.expect("some");
        assert_eq!(r.candidate.id, "d1");
        let _ = Candidate::new("anything", "to keep import alive");
        let _: ScoredCandidate = r;
    }

    #[test]
    fn band_check_inclusive_on_min() {
        let pipe = Arc::new(RetrievalPipeline::new(
            Arc::new(StubRetriever { text: "x".into() }),
            None,
            1,
            1,
        ));
        let rec = RerankRecovery::new(pipe, 0.85);
        assert!(rec.should_attempt(0.85, 0.95));
        assert!(!rec.should_attempt(0.84, 0.95));
        assert!(!rec.should_attempt(0.96, 0.95));
    }
}
