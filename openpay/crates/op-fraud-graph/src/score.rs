//! Combiners that turn the various graph signals into one risk score.
//!
//! These are deliberately small, transparent functions: every operator
//! ends up tuning them, so making the formulas opaque buys nothing. A
//! more sophisticated downstream learner (in `op-fraud`) can ingest the
//! raw component scores instead of the combined value.

use crate::components::ConnectedComponents;
use crate::pagerank::PageRankResult;
use crate::ring::{Ring, RingDetector};
use crate::synthetic::SyntheticIdentityScorer;
use crate::graph::{FraudGraph, VertexId};

/// Coarse risk band for human-readable output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RiskBand {
    /// `[0.00, 0.30)`
    Low,
    /// `[0.30, 0.60)`
    Medium,
    /// `[0.60, 0.85)`
    High,
    /// `[0.85, 1.00]`
    Severe,
}

impl RiskBand {
    /// Bucket a `[0,1]` score.
    pub fn from_score(s: f32) -> Self {
        if s < 0.30 {
            Self::Low
        } else if s < 0.60 {
            Self::Medium
        } else if s < 0.85 {
            Self::High
        } else {
            Self::Severe
        }
    }
}

/// Combined risk for a single entity vertex.
#[derive(Debug, Clone)]
pub struct EntityRisk {
    /// The vertex this score belongs to.
    pub vertex: VertexId,
    /// Component-size contribution (`[0, 1]`): saturates at 100 members.
    pub component_score: f32,
    /// PageRank centrality (`[0, 1]`): proportional to PR / max(PR).
    pub centrality_score: f32,
    /// Ring participation (`[0, 1]`): max of [`Ring::score`] across
    /// rings this vertex belongs to.
    pub ring_score: f32,
    /// Synthetic-identity score (`[0, 1]`).
    pub synthetic_score: f32,
    /// Weighted combination in `[0, 1]`.
    pub combined: f32,
    /// Coarse band.
    pub band: RiskBand,
}

/// Combined risk for one transaction (a small set of co-occurring entities).
#[derive(Debug, Clone)]
pub struct TransactionRisk {
    /// The participating vertices.
    pub vertices: Vec<VertexId>,
    /// Per-entity risk, in the same order as `vertices`.
    pub per_entity: Vec<EntityRisk>,
    /// Worst-of combiner. Most fraud teams use max() across the
    /// involved entities — one bad neighbour ruins the whole tx.
    pub combined: f32,
    /// Coarse band derived from `combined`.
    pub band: RiskBand,
}

/// Score a single vertex by combining the four component scores.
///
/// Weights are deliberately uniform-ish:
/// - ring_score: 0.40 (the loudest signal)
/// - synthetic: 0.25
/// - component_score: 0.20
/// - centrality: 0.15
pub fn score_entity(
    g: &FraudGraph,
    v: VertexId,
    components: &ConnectedComponents,
    pagerank: &PageRankResult,
    rings: &[Ring],
    synthetic: &SyntheticIdentityScorer,
    now_unix: i64,
) -> EntityRisk {
    let component_score = components
        .component_of(v)
        .and_then(|c| components.component_size(c))
        .map(|n| (n.min(100) as f32) / 100.0)
        .unwrap_or(0.0);

    let max_pr = pagerank
        .scores
        .iter()
        .copied()
        .fold(0.0_f32, f32::max)
        .max(1e-12);
    let centrality_score = pagerank.score(v).unwrap_or(0.0) / max_pr;

    let ring_score = rings
        .iter()
        .filter(|r| r.hub == v || r.members.contains(&v))
        .map(Ring::score)
        .fold(0.0_f32, f32::max);

    let synthetic_score = synthetic.score(g, v, now_unix);

    let combined =
        (0.40 * ring_score + 0.25 * synthetic_score + 0.20 * component_score + 0.15 * centrality_score)
            .clamp(0.0, 1.0);
    let band = RiskBand::from_score(combined);

    EntityRisk {
        vertex: v,
        component_score,
        centrality_score,
        ring_score,
        synthetic_score,
        combined,
        band,
    }
}

/// Score a transaction = max over participating entities.
pub fn score_transaction(
    g: &FraudGraph,
    vertices: &[VertexId],
    components: &ConnectedComponents,
    pagerank: &PageRankResult,
    rings: &[Ring],
    synthetic: &SyntheticIdentityScorer,
    now_unix: i64,
) -> TransactionRisk {
    let per: Vec<EntityRisk> = vertices
        .iter()
        .map(|&v| score_entity(g, v, components, pagerank, rings, synthetic, now_unix))
        .collect();
    let combined = per
        .iter()
        .map(|r| r.combined)
        .fold(0.0_f32, f32::max);
    let band = RiskBand::from_score(combined);
    TransactionRisk {
        vertices: vertices.to_vec(),
        per_entity: per,
        combined,
        band,
    }
}

/// Convenience: drive the full pipeline against a graph with default
/// detectors, and score every vertex.
pub fn score_all(g: &FraudGraph, now_unix: i64) -> Vec<EntityRisk> {
    let components = ConnectedComponents::from_graph(g);
    let pagerank = crate::pagerank::PageRank::default()
        .run(g)
        .unwrap_or(crate::pagerank::PageRankResult {
            scores: vec![0.0; g.vertex_count()],
            iterations: 0,
            converged: false,
        });
    let rings = RingDetector::default().detect(g);
    let synthetic = SyntheticIdentityScorer::default();
    g.vertices()
        .map(|v| score_entity(g, v, &components, &pagerank, &rings, &synthetic, now_unix))
        .collect()
}
