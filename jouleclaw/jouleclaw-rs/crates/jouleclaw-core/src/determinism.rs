//! Determinism contract types.
//!
//! See spec 04.

use crate::energy::JouleAccounting;
use crate::kernel::KernelId;
use crate::graph::NodeId;
use std::time::Duration;

/// Determinism class of an operation, kernel, or graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeterminismClass {
    /// Identical inputs always produce identical outputs.
    Deterministic,

    /// Output depends on a seed; given the seed, deterministic.
    SeededStochastic,

    /// Genuinely non-deterministic.
    Stochastic,
}

impl DeterminismClass {
    /// Composition rule: a graph's class is the most permissive of its nodes' classes.
    /// Ordering: Deterministic < SeededStochastic (with seed) < Stochastic.
    pub fn join(self, other: Self) -> Self {
        use DeterminismClass::*;
        match (self, other) {
            (Stochastic, _) | (_, Stochastic) => Stochastic,
            (SeededStochastic, _) | (_, SeededStochastic) => SeededStochastic,
            (Deterministic, Deterministic) => Deterministic,
        }
    }
}

/// Hash of a graph's structure (topology + ops + attrs + constant hashes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphHash(pub [u8; 32]);

/// Hash of a tensor's contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorHash(pub [u8; 32]);

/// Identifier of a calibrated topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TopologyId(pub u64);

/// Hash of a memory placement plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanHash(pub [u8; 32]);

/// A trace of one graph execution.
///
/// In deterministic mode, two traces of the same graph + inputs + topology
/// must be byte-identical except for `wall_clock` and per-call `joule_accounting`
/// noise.
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    pub graph_hash: GraphHash,
    pub input_hashes: Vec<TensorHash>,
    pub output_hashes: Vec<TensorHash>,
    pub topology_id: TopologyId,
    pub kernel_selections: Vec<(NodeId, KernelId)>,
    pub memory_plan_hash: PlanHash,
    pub joule_accounting: JouleAccounting,
    pub wall_clock: Duration,
    pub determinism_mode: DeterminismMode,
}

/// Whether the execution was performed in strict deterministic mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeterminismMode {
    /// Strict: planner rejected any non-deterministic kernel.
    Strict,
    /// Stochastic: at least one non-deterministic kernel was permitted.
    Stochastic,
}
