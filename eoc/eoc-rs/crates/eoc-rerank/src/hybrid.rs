//! Hybrid retrieval — fuse multiple rankers.
//!
//! Two fusion methods:
//!
//! * **Reciprocal Rank Fusion (RRF)**, Cormack et al. 2009:
//!
//!   ```text
//!   score(d) = Σ_r  1 / (k + rank_r(d))
//!   ```
//!
//!   Robust to score-distribution differences across retrievers (BM25
//!   produces unbounded positive scores; cosine produces `[-1, 1]`).
//!   Default `k = 60` per the original paper.
//!
//! * **Min-max normalisation + weighted sum** — normalise each retriever's
//!   scores into `[0, 1]` and sum with caller-supplied weights. Useful
//!   when the user wants to dial up dense vs sparse explicitly.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::DocId;
use crate::error::RerankResult;
use crate::reranker::Retriever;

/// How to fuse two or more rankings into one.
#[derive(Debug, Clone)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion with constant `k` (default 60).
    Rrf {
        /// Damping constant. Higher = ranks below ~k matter less.
        k: f32,
    },
    /// Min-max normalise each retriever's scores into `[0, 1]`, then take
    /// a weighted sum. `weights[i]` matches `rankers[i]`.
    MinMaxWeighted {
        /// Per-retriever weights. Must match the number of rankers.
        weights: Vec<f32>,
    },
}

impl Default for FusionMethod {
    fn default() -> Self {
        FusionMethod::Rrf { k: 60.0 }
    }
}

/// A hybrid retriever — runs several rankers and fuses their rankings.
pub struct HybridRetriever {
    /// Underlying rankers (typically one dense + one sparse).
    pub rankers: Vec<Arc<dyn Retriever>>,
    /// Fusion method.
    pub fusion: FusionMethod,
    /// Per-ranker top-K. Caller should set this >= the requested
    /// pipeline top-K so the fusion has enough material to work with.
    pub top_k: usize,
    name: String,
}

impl HybridRetriever {
    /// Construct a hybrid retriever.
    pub fn new(rankers: Vec<Arc<dyn Retriever>>, fusion: FusionMethod, top_k: usize) -> Self {
        let name = format!(
            "hybrid({})",
            rankers.iter().map(|r| r.name()).collect::<Vec<_>>().join("+")
        );
        Self {
            rankers,
            fusion,
            top_k,
            name,
        }
    }

    /// Fuse pre-computed `rankings` (one Vec per retriever) using the
    /// configured method. Exposed for testing.
    pub fn fuse(&self, rankings: &[Vec<(DocId, f32)>]) -> Vec<(DocId, f32)> {
        match &self.fusion {
            FusionMethod::Rrf { k } => rrf(rankings, *k),
            FusionMethod::MinMaxWeighted { weights } => min_max_weighted(rankings, weights),
        }
    }
}

#[async_trait]
impl Retriever for HybridRetriever {
    async fn retrieve(&self, query: &str, top_k: usize) -> RerankResult<Vec<(DocId, f32)>> {
        let mut rankings: Vec<Vec<(DocId, f32)>> = Vec::with_capacity(self.rankers.len());
        for r in &self.rankers {
            let inner = r.retrieve(query, self.top_k.max(top_k)).await?;
            rankings.push(inner);
        }
        let mut fused = self.fuse(&rankings);
        fused.truncate(top_k);
        Ok(fused)
    }

    fn document_text(&self, id: &DocId) -> Option<String> {
        for r in &self.rankers {
            if let Some(t) = r.document_text(id) {
                return Some(t);
            }
        }
        None
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Reciprocal Rank Fusion. Each ranking contributes `1 / (k + rank)`
/// (1-based rank) per document.
pub fn rrf(rankings: &[Vec<(DocId, f32)>], k: f32) -> Vec<(DocId, f32)> {
    let mut scores: HashMap<DocId, f32> = HashMap::new();
    for ranking in rankings {
        for (rank, (id, _)) in ranking.iter().enumerate() {
            let r = rank as f32 + 1.0;
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + r);
        }
    }
    let mut out: Vec<(DocId, f32)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Min-max normalise per-ranking then take a weighted sum.
pub fn min_max_weighted(rankings: &[Vec<(DocId, f32)>], weights: &[f32]) -> Vec<(DocId, f32)> {
    let n = rankings.len().min(weights.len());
    let mut scores: HashMap<DocId, f32> = HashMap::new();
    for i in 0..n {
        let ranking = &rankings[i];
        if ranking.is_empty() {
            continue;
        }
        let min = ranking
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::INFINITY, f32::min);
        let max = ranking
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(1e-9);
        let w = weights[i];
        for (id, s) in ranking {
            let normed = (s - min) / range;
            *scores.entry(id.clone()).or_insert(0.0) += w * normed;
        }
    }
    let mut out: Vec<(DocId, f32)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_fuses_synthetic_rankings() {
        let dense = vec![
            ("a".to_string(), 0.95),
            ("b".to_string(), 0.80),
            ("c".to_string(), 0.50),
        ];
        let sparse = vec![
            ("b".to_string(), 12.0),
            ("a".to_string(), 9.0),
            ("d".to_string(), 7.0),
        ];
        let fused = rrf(&[dense, sparse], 60.0);
        // `a` and `b` are top-2 in both -> top of fused.
        let top: Vec<&String> = fused.iter().take(2).map(|(id, _)| id).collect();
        assert!(top.contains(&&"a".to_string()));
        assert!(top.contains(&&"b".to_string()));
    }

    #[test]
    fn rrf_handles_empty_ranker() {
        let dense = vec![("a".to_string(), 0.9), ("b".to_string(), 0.5)];
        let sparse: Vec<(DocId, f32)> = vec![];
        let fused = rrf(&[dense, sparse], 60.0);
        assert_eq!(fused[0].0, "a");
    }

    #[test]
    fn min_max_weighted_combines() {
        let dense = vec![("a".to_string(), 0.9), ("b".to_string(), 0.1)];
        let sparse = vec![("a".to_string(), 1.0), ("b".to_string(), 0.0)];
        let fused = min_max_weighted(&[dense, sparse], &[0.5, 0.5]);
        assert_eq!(fused[0].0, "a");
    }
}
