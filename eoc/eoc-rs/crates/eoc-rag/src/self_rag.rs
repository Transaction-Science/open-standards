//! Self-RAG — self-reflective retrieval-augmented generation.
//!
//! Asai et al. 2023 ("Self-RAG: Learning to Retrieve, Generate, and
//! Critique through Self-Reflection", arXiv:2310.11511) trains a
//! generator to emit four classes of critique tokens:
//!
//! * `[Retrieve]` / `[NoRetrieve]` — should the model retrieve for
//!   this prefix?
//! * `[ISREL]` / `[NotRel]` — is each retrieved passage relevant?
//! * `[ISSUP]` / `[NoSupport]` — is the model's draft supported by the
//!   retrieved passages?
//! * `[ISUSE]` — overall utility score for the draft answer.
//!
//! The reference implementation here exposes the critique-token
//! protocol as a strongly-typed [`CritiqueToken`] enum so vendor
//! backends can map their decoder output 1:1. The bundled
//! [`HeuristicCritic`] is a deterministic critic used in tests.

use std::sync::Arc;

use async_trait::async_trait;
use eoc_core::JouleCost;
use serde::{Deserialize, Serialize};

use crate::citation::{CitationEnforcement, CitationPolicy, derive_citations};
use crate::error::{RagError, RagResult};
use crate::pipeline::{Answer, Pipeline, RagRequest, Stage, Trace, TraceEvent};
use crate::store::{DocumentStore, RetrievedChunk};

/// One Self-RAG critique token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CritiqueToken {
    /// `[Retrieve]` — retrieve for this segment.
    Retrieve,
    /// `[NoRetrieve]` — skip retrieval.
    NoRetrieve,
    /// `[ISREL]` — passage is relevant.
    IsRelevant,
    /// `[NotRel]` — passage is not relevant.
    NotRelevant,
    /// `[ISSUP]` — answer is supported by the passages.
    IsSupported,
    /// `[NoSupport]` — answer is not supported.
    NoSupport,
}

impl CritiqueToken {
    /// Stable string identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            CritiqueToken::Retrieve => "Retrieve",
            CritiqueToken::NoRetrieve => "NoRetrieve",
            CritiqueToken::IsRelevant => "ISREL",
            CritiqueToken::NotRelevant => "NotRel",
            CritiqueToken::IsSupported => "ISSUP",
            CritiqueToken::NoSupport => "NoSupport",
        }
    }
}

/// A critique over one pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Critique {
    /// Retrieval decision.
    pub retrieve: CritiqueToken,
    /// Per-passage relevance tokens (parallel to the retrieved set).
    pub passage_relevance: Vec<CritiqueToken>,
    /// Final support token.
    pub support: CritiqueToken,
    /// Utility score `[ISUSE]` in `[1, 5]`.
    pub utility: u8,
}

/// A critic produces a [`Critique`] from `(query, draft, passages)`.
#[async_trait]
pub trait Critic: Send + Sync {
    /// Decide whether to retrieve.
    async fn should_retrieve(&self, query: &str) -> RagResult<CritiqueToken>;

    /// Score relevance of each passage.
    async fn score_passages(
        &self,
        query: &str,
        chunks: &[RetrievedChunk],
    ) -> RagResult<Vec<CritiqueToken>>;

    /// Score whether the draft is supported by the passages.
    async fn score_support(
        &self,
        draft: &str,
        chunks: &[RetrievedChunk],
    ) -> RagResult<CritiqueToken>;

    /// Overall utility score in `[1, 5]`.
    async fn score_utility(&self, draft: &str) -> RagResult<u8>;
}

/// Deterministic reference critic.
///
/// * Retrieve iff the query contains at least one alphanumeric token
///   of length >= 3.
/// * A passage is relevant iff it shares >= 1 alphanumeric token of
///   length >= 3 with the query.
/// * The draft is supported iff at least one passage was relevant.
/// * Utility = 3 + (relevant_passages > 0 ? 2 : 0).
pub struct HeuristicCritic;

fn long_tokens(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            for low in ch.to_lowercase() {
                cur.push(low);
            }
        } else {
            if cur.len() >= 3 {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if cur.len() >= 3 {
        out.push(cur);
    }
    out
}

#[async_trait]
impl Critic for HeuristicCritic {
    async fn should_retrieve(&self, query: &str) -> RagResult<CritiqueToken> {
        if long_tokens(query).is_empty() {
            Ok(CritiqueToken::NoRetrieve)
        } else {
            Ok(CritiqueToken::Retrieve)
        }
    }

    async fn score_passages(
        &self,
        query: &str,
        chunks: &[RetrievedChunk],
    ) -> RagResult<Vec<CritiqueToken>> {
        let qt = long_tokens(query);
        Ok(chunks
            .iter()
            .map(|c| {
                let pt = long_tokens(&c.chunk.text);
                if qt.iter().any(|q| pt.contains(q)) {
                    CritiqueToken::IsRelevant
                } else {
                    CritiqueToken::NotRelevant
                }
            })
            .collect())
    }

    async fn score_support(
        &self,
        draft: &str,
        chunks: &[RetrievedChunk],
    ) -> RagResult<CritiqueToken> {
        let dt = long_tokens(draft);
        for c in chunks {
            let pt = long_tokens(&c.chunk.text);
            if dt.iter().any(|d| pt.contains(d)) {
                return Ok(CritiqueToken::IsSupported);
            }
        }
        Ok(CritiqueToken::NoSupport)
    }

    async fn score_utility(&self, draft: &str) -> RagResult<u8> {
        Ok(if draft.is_empty() { 1 } else { 4 })
    }
}

/// Self-RAG pipeline.
pub struct SelfRagPipeline {
    /// Store.
    pub store: Arc<dyn DocumentStore>,
    /// Critic — defaults to [`HeuristicCritic`].
    pub critic: Arc<dyn Critic>,
    /// Citation policy.
    pub citation_policy: CitationPolicy,
    /// Joule budget per stage.
    pub critic_microjoules: u64,
    /// Retrieval cost.
    pub retrieve_microjoules: u64,
    /// Generation cost.
    pub generate_microjoules: u64,
}

impl SelfRagPipeline {
    /// Construct.
    pub fn new(store: Arc<dyn DocumentStore>) -> Self {
        Self {
            store,
            critic: Arc::new(HeuristicCritic),
            citation_policy: CitationPolicy::Optional,
            critic_microjoules: 10_000,
            retrieve_microjoules: 5_000,
            generate_microjoules: 50_000,
        }
    }
}

#[async_trait]
impl Pipeline for SelfRagPipeline {
    async fn answer(&self, req: &RagRequest) -> RagResult<Answer> {
        if req.top_k == 0 {
            return Err(RagError::Config("top_k must be >= 1".into()));
        }
        let mut trace = Trace::new();

        let retrieve_tok = self.critic.should_retrieve(&req.query).await?;
        trace.record(TraceEvent::new(
            Stage::Critique,
            JouleCost::estimated(self.critic_microjoules),
            format!("[{}]", retrieve_tok.as_str()),
        ));

        let chunks = match retrieve_tok {
            CritiqueToken::Retrieve => {
                let c = self.store.retrieve(&req.query, req.top_k).await?;
                trace.record(TraceEvent::new(
                    Stage::Retrieve,
                    JouleCost::estimated(self.retrieve_microjoules),
                    format!("retrieved {} chunks", c.len()),
                ));
                c
            }
            _ => Vec::new(),
        };

        // Filter by relevance critique tokens.
        let rel = self.critic.score_passages(&req.query, &chunks).await?;
        let kept: Vec<RetrievedChunk> = chunks
            .iter()
            .zip(rel.iter())
            .filter(|(_, t)| **t == CritiqueToken::IsRelevant)
            .map(|(c, _)| c.clone())
            .collect();
        trace.record(TraceEvent::new(
            Stage::Critique,
            JouleCost::estimated(self.critic_microjoules),
            format!("{} relevant / {} retrieved", kept.len(), chunks.len()),
        ));

        // Draft = top-kept (or empty fallback).
        let draft = if kept.is_empty() {
            String::from("I don't have enough information to answer.")
        } else {
            kept[0].chunk.text.clone()
        };
        trace.record(TraceEvent::new(
            Stage::Generate,
            JouleCost::estimated(self.generate_microjoules),
            "self-rag draft",
        ));

        let support = self.critic.score_support(&draft, &kept).await?;
        trace.record(TraceEvent::new(
            Stage::Critique,
            JouleCost::estimated(self.critic_microjoules),
            format!("[{}]", support.as_str()),
        ));

        if matches!(support, CritiqueToken::NoSupport) && !kept.is_empty() {
            return Err(RagError::GuardRejected(
                "Self-RAG support critique failed".into(),
            ));
        }

        let citations = derive_citations(&draft, &kept);
        CitationEnforcement::new(self.citation_policy).enforce(&draft, &kept, &citations)?;

        Ok(Answer::new(draft, kept, trace).with_citations(citations))
    }

    fn name(&self) -> &str {
        "self-rag"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Chunk, InMemoryStore};

    #[tokio::test]
    async fn self_rag_keeps_relevant_only() {
        let store: Arc<dyn DocumentStore> = Arc::new(InMemoryStore::from_chunks(
            "test",
            vec![
                Chunk::new("d1", 0, "joules per byte is the EOC efficiency metric"),
                Chunk::new("d2", 0, "wheel diameter is unrelated"),
            ],
        ));
        let p = SelfRagPipeline::new(store);
        let req = RagRequest::new("EOC efficiency", 5);
        let ans = p.answer(&req).await.expect("ok");
        assert!(ans.chunks.iter().all(|c| c.chunk.text.contains("EOC")));
    }
}
