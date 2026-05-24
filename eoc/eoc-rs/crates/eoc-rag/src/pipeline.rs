//! Pipeline trait, stage enum, and trace types.
//!
//! Every RAG strategy in this crate implements [`Pipeline`]. A
//! pipeline takes a [`RagRequest`] and produces an [`Answer`] alongside
//! a [`Trace`] of every [`Stage`] that ran, each tagged with a
//! [`eoc_core::JouleCost`] for energy accounting.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use eoc_core::{JouleCost, QueryId};

use crate::citation::Cite;
use crate::error::RagResult;
use crate::store::RetrievedChunk;

/// A coarse-grained RAG stage. Mirrors the categories described in
/// "Retrieval-Augmented Generation for Large Language Models: A
/// Survey" (Gao et al. 2023).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stage {
    /// Routing / classification (no retrieval yet).
    Route,
    /// Query rewriting / expansion / decomposition.
    Rewrite,
    /// Hypothetical-document synthesis (HyDE).
    HypotheticalDocument,
    /// Embedding the query (or expansions) into vector space.
    Embed,
    /// First-stage retrieval (dense / sparse / hybrid).
    Retrieve,
    /// Rank-fusion over multiple retrievals.
    Fuse,
    /// Cross-encoder reranking of the candidate set.
    Rerank,
    /// Retrieval-quality evaluation (CRAG).
    Evaluate,
    /// Self-critique (Self-RAG).
    Critique,
    /// Final answer generation.
    Generate,
    /// Hallucination / consistency guard (SelfCheckGPT).
    Guard,
}

impl Stage {
    /// Stable string identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Stage::Route => "route",
            Stage::Rewrite => "rewrite",
            Stage::HypotheticalDocument => "hyde",
            Stage::Embed => "embed",
            Stage::Retrieve => "retrieve",
            Stage::Fuse => "fuse",
            Stage::Rerank => "rerank",
            Stage::Evaluate => "evaluate",
            Stage::Critique => "critique",
            Stage::Generate => "generate",
            Stage::Guard => "guard",
        }
    }
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One traced event in a pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    /// Which stage produced this event.
    pub stage: Stage,
    /// Energy attributed to this stage.
    pub joule_cost: JouleCost,
    /// Human-readable note (e.g. "retrieved 12 candidates").
    pub note: String,
}

impl TraceEvent {
    /// Construct a new trace event.
    pub fn new(stage: Stage, joule_cost: JouleCost, note: impl Into<String>) -> Self {
        Self {
            stage,
            joule_cost,
            note: note.into(),
        }
    }
}

/// A full ordered trace of one pipeline run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Trace {
    /// The events that ran, in order.
    pub events: Vec<TraceEvent>,
}

impl Trace {
    /// Construct an empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an event.
    pub fn record(&mut self, ev: TraceEvent) {
        self.events.push(ev);
    }

    /// Total micro-joules across every recorded stage.
    pub fn total_microjoules(&self) -> u64 {
        self.events.iter().map(|e| e.joule_cost.microjoules).sum()
    }

    /// Total joules across every recorded stage.
    pub fn total_joules(&self) -> f64 {
        (self.total_microjoules() as f64) / 1_000_000.0
    }

    /// Stages that ran (deduplicated, in order of first occurrence).
    pub fn stages(&self) -> Vec<Stage> {
        let mut out: Vec<Stage> = Vec::new();
        for ev in &self.events {
            if !out.contains(&ev.stage) {
                out.push(ev.stage);
            }
        }
        out
    }
}

/// A request to a [`Pipeline`].
#[derive(Debug, Clone)]
pub struct RagRequest {
    /// The user's question.
    pub query: String,
    /// Final number of chunks to keep in the context window.
    pub top_k: usize,
    /// Optional per-call metadata (tenant, request-id, etc.).
    pub metadata: BTreeMap<String, String>,
}

impl RagRequest {
    /// Construct a request.
    pub fn new(query: impl Into<String>, top_k: usize) -> Self {
        Self {
            query: query.into(),
            top_k,
            metadata: BTreeMap::new(),
        }
    }

    /// Insert metadata (consumes `self`).
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Content-addressed query id.
    pub fn query_id(&self) -> QueryId {
        QueryId::from_prompt(&self.query)
    }
}

/// The final answer the pipeline returns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    /// The user-facing answer text.
    pub text: String,
    /// The chunks supporting the answer.
    pub chunks: Vec<RetrievedChunk>,
    /// Span-backed citations.
    pub citations: Vec<Cite>,
    /// Full trace, including joule accounting.
    pub trace: Trace,
}

impl Answer {
    /// Construct an answer.
    pub fn new(text: impl Into<String>, chunks: Vec<RetrievedChunk>, trace: Trace) -> Self {
        Self {
            text: text.into(),
            chunks,
            citations: Vec::new(),
            trace,
        }
    }

    /// Attach citations (consumes `self`).
    pub fn with_citations(mut self, citations: Vec<Cite>) -> Self {
        self.citations = citations;
        self
    }
}

/// A RAG pipeline.
#[async_trait]
pub trait Pipeline: Send + Sync {
    /// Run the pipeline.
    async fn answer(&self, req: &RagRequest) -> RagResult<Answer>;

    /// Pipeline name (used in traces / receipts).
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use eoc_core::{JouleCost, JouleSource};

    #[test]
    fn trace_accumulates_microjoules() {
        let mut t = Trace::new();
        t.record(TraceEvent::new(
            Stage::Retrieve,
            JouleCost::estimated(1_000),
            "retrieve",
        ));
        t.record(TraceEvent::new(
            Stage::Generate,
            JouleCost::estimated(9_000),
            "generate",
        ));
        assert_eq!(t.total_microjoules(), 10_000);
        assert!((t.total_joules() - 0.01).abs() < 1e-9);
        assert_eq!(t.stages(), vec![Stage::Retrieve, Stage::Generate]);
        assert_eq!(t.events[0].joule_cost.source, JouleSource::Estimated);
    }

    #[test]
    fn stage_as_str_round_trip() {
        for s in [
            Stage::Route,
            Stage::Rewrite,
            Stage::HypotheticalDocument,
            Stage::Embed,
            Stage::Retrieve,
            Stage::Fuse,
            Stage::Rerank,
            Stage::Evaluate,
            Stage::Critique,
            Stage::Generate,
            Stage::Guard,
        ] {
            assert!(!s.as_str().is_empty());
        }
    }
}
