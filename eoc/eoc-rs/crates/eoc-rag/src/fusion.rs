//! RAG-Fusion — multi-query expansion + Reciprocal Rank Fusion.
//!
//! Cormack et al. 2009 introduced RRF; Adrian Raudaschl's "RAG-Fusion"
//! (2023) wired RRF into a RAG pipeline that issues several rewritten
//! queries in parallel, then fuses the per-query rankings into a
//! single consensus list. The fused ranking is more robust than any
//! single retrieval because it cancels the random noise of any one
//! rewrite.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use eoc_core::JouleCost;

use crate::citation::{CitationEnforcement, CitationPolicy, derive_citations};
use crate::error::{RagError, RagResult};
use crate::pipeline::{Answer, Pipeline, RagRequest, Stage, Trace, TraceEvent};
use crate::rewrite::{HeuristicRewriter, QueryRewriter, RewriteKind};
use crate::store::{ChunkId, DocumentStore, RetrievedChunk};

/// RAG-Fusion pipeline.
pub struct RagFusionPipeline {
    /// Store.
    pub store: Arc<dyn DocumentStore>,
    /// Rewriter — defaults to [`HeuristicRewriter`].
    pub rewriter: Arc<dyn QueryRewriter>,
    /// Per-rewrite retrieval depth.
    pub per_query_top_k: usize,
    /// RRF damping constant. Defaults to 60.0 per the original paper.
    pub rrf_k: f32,
    /// Citation policy.
    pub citation_policy: CitationPolicy,
    /// Per-stage joule estimates.
    pub rewrite_microjoules: u64,
    /// Cost per retrieval call.
    pub retrieve_microjoules: u64,
    /// Cost of the fusion step.
    pub fuse_microjoules: u64,
    /// Cost of the generation step.
    pub generate_microjoules: u64,
}

impl RagFusionPipeline {
    /// Construct with deterministic defaults.
    pub fn new(store: Arc<dyn DocumentStore>) -> Self {
        Self {
            store,
            rewriter: Arc::new(HeuristicRewriter),
            per_query_top_k: 10,
            rrf_k: 60.0,
            citation_policy: CitationPolicy::Optional,
            rewrite_microjoules: 25_000,
            retrieve_microjoules: 5_000,
            fuse_microjoules: 500,
            generate_microjoules: 50_000,
        }
    }
}

#[async_trait]
impl Pipeline for RagFusionPipeline {
    async fn answer(&self, req: &RagRequest) -> RagResult<Answer> {
        if req.top_k == 0 {
            return Err(RagError::Config("top_k must be >= 1".into()));
        }
        let mut trace = Trace::new();

        let rewrites = self
            .rewriter
            .rewrite(&req.query, RewriteKind::MultiQuery)
            .await?;
        trace.record(TraceEvent::new(
            Stage::Rewrite,
            JouleCost::estimated(self.rewrite_microjoules),
            format!("{} rewrites", rewrites.len()),
        ));

        let mut rankings: Vec<Vec<RetrievedChunk>> = Vec::with_capacity(rewrites.len());
        for r in &rewrites {
            let hits = self.store.retrieve(r, self.per_query_top_k).await?;
            trace.record(TraceEvent::new(
                Stage::Retrieve,
                JouleCost::estimated(self.retrieve_microjoules),
                format!("retrieve top-{}: \"{}\"", self.per_query_top_k, r),
            ));
            rankings.push(hits);
        }

        let fused = reciprocal_rank_fusion(&rankings, self.rrf_k, req.top_k);
        trace.record(TraceEvent::new(
            Stage::Fuse,
            JouleCost::estimated(self.fuse_microjoules),
            format!("RRF over {} rankings", rankings.len()),
        ));

        if fused.is_empty() {
            return Err(RagError::NoChunks);
        }

        let answer_text = fused[0].chunk.text.clone();
        trace.record(TraceEvent::new(
            Stage::Generate,
            JouleCost::estimated(self.generate_microjoules),
            "fusion stuffed generation",
        ));

        let citations = derive_citations(&answer_text, &fused);
        CitationEnforcement::new(self.citation_policy).enforce(&answer_text, &fused, &citations)?;

        Ok(Answer::new(answer_text, fused, trace).with_citations(citations))
    }

    fn name(&self) -> &str {
        "rag-fusion"
    }
}

/// Reciprocal Rank Fusion over per-query rankings. Returns a fused
/// ranking truncated to `top_k`. Each input is assumed sorted in
/// descending relevance.
pub fn reciprocal_rank_fusion(
    rankings: &[Vec<RetrievedChunk>],
    k: f32,
    top_k: usize,
) -> Vec<RetrievedChunk> {
    let mut scores: HashMap<ChunkId, f32> = HashMap::new();
    let mut by_id: HashMap<ChunkId, RetrievedChunk> = HashMap::new();
    for ranking in rankings {
        for (rank, rc) in ranking.iter().enumerate() {
            let r = rank as f32 + 1.0;
            *scores.entry(rc.chunk.id).or_insert(0.0) += 1.0 / (k + r);
            by_id.entry(rc.chunk.id).or_insert_with(|| rc.clone());
        }
    }
    let mut out: Vec<(ChunkId, f32)> = scores.into_iter().collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.0.cmp(&b.0.0))
    });
    out.truncate(top_k);
    out.into_iter()
        .enumerate()
        .filter_map(|(i, (id, s))| {
            by_id.remove(&id).map(|mut rc| {
                rc.score = s;
                rc.rank = i + 1;
                rc
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Chunk, InMemoryStore};

    fn rc(doc: &str, text: &str, score: f32, rank: usize) -> RetrievedChunk {
        RetrievedChunk {
            chunk: Chunk::new(doc, 0, text),
            score,
            rank,
        }
    }

    #[test]
    fn rrf_consensus_promotes_shared_top() {
        let a = vec![
            rc("d1", "alpha", 1.0, 1),
            rc("d2", "beta", 0.5, 2),
            rc("d3", "gamma", 0.1, 3),
        ];
        let b = vec![
            rc("d2", "beta", 1.0, 1),
            rc("d1", "alpha", 0.5, 2),
            rc("d4", "delta", 0.1, 3),
        ];
        let fused = reciprocal_rank_fusion(&[a, b], 60.0, 3);
        // d1 and d2 share top-2 in both -> fused top-2.
        let ids: Vec<&str> = fused
            .iter()
            .map(|r| r.chunk.doc_id.as_str())
            .take(2)
            .collect();
        assert!(ids.contains(&"d1"));
        assert!(ids.contains(&"d2"));
    }

    #[tokio::test]
    async fn fusion_pipeline_returns_answer() {
        let store: Arc<dyn DocumentStore> = Arc::new(InMemoryStore::from_chunks(
            "test",
            vec![
                Chunk::new("d1", 0, "compute joules per byte for EOC efficiency"),
                Chunk::new("d2", 0, "compute the joules drawn during inference"),
                Chunk::new("d3", 0, "wheel diameter is unrelated"),
            ],
        ));
        let p = RagFusionPipeline::new(store);
        let req = RagRequest::new("compute joules", 2);
        let ans = p.answer(&req).await.expect("ok");
        assert!(!ans.chunks.is_empty());
        assert!(ans.trace.events.iter().any(|e| e.stage == Stage::Fuse));
    }
}
