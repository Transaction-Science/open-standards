//! Adaptive RAG — Jeong et al. 2024 ("Adaptive-RAG: Learning to Adapt
//! Retrieval-Augmented Large Language Models through Question
//! Complexity", arXiv:2403.14403).
//!
//! A classifier routes each query to one of three pipelines based on
//! estimated complexity:
//!
//! * [`Complexity::NoRetrieval`] — the LLM answers without retrieval.
//! * [`Complexity::SingleStep`] — one retrieve → generate pass.
//! * [`Complexity::MultiStep`] — multiple retrieval rounds (the
//!   pipeline reduces this to fusion).
//!
//! The reference [`HeuristicAdaptiveRouter`] uses surface features
//! (length, conjunctions, presence of question words). Vendor
//! backends override with a small trained classifier.

use std::sync::Arc;

use async_trait::async_trait;
use eoc_core::JouleCost;
use serde::{Deserialize, Serialize};

use crate::error::RagResult;
use crate::fusion::RagFusionPipeline;
use crate::naive::NaivePipeline;
use crate::pipeline::{Answer, Pipeline, RagRequest, Stage, Trace, TraceEvent};
use crate::store::DocumentStore;

/// Question-complexity bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Complexity {
    /// Trivial — no retrieval needed.
    NoRetrieval,
    /// One-shot retrieval is enough.
    SingleStep,
    /// Multi-hop — needs decomposition / fusion.
    MultiStep,
}

/// Question-complexity router.
#[async_trait]
pub trait ComplexityRouter: Send + Sync {
    /// Classify `query`.
    async fn classify(&self, query: &str) -> RagResult<Complexity>;
}

/// Heuristic router used as the reference.
pub struct HeuristicAdaptiveRouter;

#[async_trait]
impl ComplexityRouter for HeuristicAdaptiveRouter {
    async fn classify(&self, query: &str) -> RagResult<Complexity> {
        let lower = query.to_lowercase();
        let trimmed = lower.trim();
        // Multi-hop signal: explicit conjunctions or multiple question
        // marks.
        let conjunctions = [" and ", " then ", " versus ", " vs ", ";"];
        if conjunctions.iter().any(|c| trimmed.contains(c))
            || trimmed.matches('?').count() >= 2
        {
            return Ok(Complexity::MultiStep);
        }
        // Trivial signal: bare arithmetic or pure greetings.
        let trivial = ["hi", "hello", "thanks", "thank you", "ok", "yes", "no"];
        if trivial.contains(&trimmed.trim_end_matches('?').trim_end_matches('.')) {
            return Ok(Complexity::NoRetrieval);
        }
        Ok(Complexity::SingleStep)
    }
}

/// Adaptive router pipeline.
pub struct AdaptiveRouter {
    /// Store shared with the underlying pipelines.
    pub store: Arc<dyn DocumentStore>,
    /// Router used for classification.
    pub router: Arc<dyn ComplexityRouter>,
    /// Single-step backend.
    pub single_step: Arc<NaivePipeline>,
    /// Multi-step backend.
    pub multi_step: Arc<RagFusionPipeline>,
    /// Router joule cost.
    pub route_microjoules: u64,
    /// "No retrieval" generator cost.
    pub direct_generate_microjoules: u64,
}

impl AdaptiveRouter {
    /// Construct with the deterministic heuristic router and the
    /// default single/multi-step backends.
    pub fn new(store: Arc<dyn DocumentStore>) -> Self {
        let single = NaivePipeline::new(store.clone())
            .with_citation_policy(crate::citation::CitationPolicy::Optional);
        Self {
            store: store.clone(),
            router: Arc::new(HeuristicAdaptiveRouter),
            single_step: Arc::new(single),
            multi_step: Arc::new(RagFusionPipeline::new(store)),
            route_microjoules: 1_000,
            direct_generate_microjoules: 30_000,
        }
    }
}

#[async_trait]
impl Pipeline for AdaptiveRouter {
    async fn answer(&self, req: &RagRequest) -> RagResult<Answer> {
        let complexity = self.router.classify(&req.query).await?;
        let mut prelude = Trace::new();
        prelude.record(TraceEvent::new(
            Stage::Route,
            JouleCost::estimated(self.route_microjoules),
            format!("complexity={complexity:?}"),
        ));

        match complexity {
            Complexity::NoRetrieval => {
                let mut t = prelude;
                t.record(TraceEvent::new(
                    Stage::Generate,
                    JouleCost::estimated(self.direct_generate_microjoules),
                    "no-retrieval direct answer",
                ));
                Ok(Answer::new(
                    format!("(direct) {}", req.query),
                    Vec::new(),
                    t,
                ))
            }
            Complexity::SingleStep => {
                let mut ans = self.single_step.answer(req).await?;
                // Prepend the routing event.
                let mut events = prelude.events;
                events.extend(ans.trace.events);
                ans.trace.events = events;
                Ok(ans)
            }
            Complexity::MultiStep => {
                let mut ans = self.multi_step.answer(req).await?;
                let mut events = prelude.events;
                events.extend(ans.trace.events);
                ans.trace.events = events;
                Ok(ans)
            }
        }
    }

    fn name(&self) -> &str {
        "adaptive"
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
                Chunk::new("d2", 0, "wheel diameter is unrelated"),
            ],
        ))
    }

    #[tokio::test]
    async fn classifier_routes_multi_step() {
        let r = HeuristicAdaptiveRouter;
        let c = r
            .classify("explain BM25 and how HNSW differs from IVF")
            .await
            .expect("ok");
        assert_eq!(c, Complexity::MultiStep);
    }

    #[tokio::test]
    async fn classifier_routes_trivial() {
        let r = HeuristicAdaptiveRouter;
        let c = r.classify("hi").await.expect("ok");
        assert_eq!(c, Complexity::NoRetrieval);
    }

    #[tokio::test]
    async fn adaptive_pipeline_traces_route_event() {
        let p = AdaptiveRouter::new(store());
        let req = RagRequest::new("EOC efficiency metric", 2);
        let ans = p.answer(&req).await.expect("ok");
        assert!(ans.trace.events.iter().any(|e| e.stage == Stage::Route));
    }
}
