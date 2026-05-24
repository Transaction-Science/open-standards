//! Sealed error type for `op-fraud-graph`.

use thiserror::Error;

/// Result alias for fallible graph operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Every failure mode the fraud-graph layer can surface.
///
/// Variants intentionally carry plain strings rather than nested error
/// chains: the caller (`op-orchestrator` or operator code) does not
/// branch on the inner kind, only on the outer category.
#[derive(Debug, Error)]
pub enum Error {
    /// A vertex id was used after the vertex was removed, or an id from a
    /// different graph was passed in.
    #[error("vertex not found: {0:?}")]
    VertexNotFound(crate::graph::VertexId),

    /// Caller asked for an entity by [`crate::EntityKey`] that has never
    /// been ingested.
    #[error("entity not found")]
    EntityNotFound,

    /// Edge weight was non-finite (NaN / ±inf) or strictly negative.
    /// PageRank and Louvain assume well-defined positive weights.
    #[error("edge weight {0} is not a finite non-negative number")]
    InvalidEdgeWeight(f32),

    /// Algorithm asked for a configuration that doesn't make sense
    /// (e.g. PageRank with damping outside `(0.0, 1.0)`,
    /// velocity window of zero).
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),

    /// Algorithm hit its iteration budget without converging. The caller
    /// gets partial results back; this variant exists so they can log it.
    #[error("algorithm did not converge in {0} iterations")]
    NotConverged(u32),
}
