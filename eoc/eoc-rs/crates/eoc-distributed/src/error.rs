//! Typed errors for `eoc-distributed`.

use thiserror::Error;

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, DistributedError>;

/// All failures surfaced by distributed scheduling, routing, and cluster
/// coordination.
#[derive(Debug, Error)]
pub enum DistributedError {
    /// The cluster has no live workers that can satisfy the request.
    #[error("no workers available")]
    NoWorkers,

    /// The requested worker id is not in the cluster.
    #[error("unknown worker: {0}")]
    UnknownWorker(String),

    /// The requested capability is not advertised by any live worker.
    #[error("no worker satisfies capability: {0}")]
    UnsatisfiedCapability(String),

    /// The scheduler queue is full and cannot accept new work.
    #[error("queue full (cap={0})")]
    QueueFull(usize),

    /// Heartbeat budget exceeded — the worker is presumed dead.
    #[error("heartbeat timeout for worker {0}")]
    HeartbeatTimeout(String),

    /// Replica controller refused a scale operation (cap hit / cooldown).
    #[error("scale rejected: {0}")]
    ScaleRejected(String),

    /// Batch scheduler refused to admit a request.
    #[error("batch admission rejected: {0}")]
    BatchRejected(String),

    /// Topology builder rejected the configuration.
    #[error("invalid topology: {0}")]
    InvalidTopology(String),

    /// Catch-all.
    #[error("{0}")]
    Other(String),
}
