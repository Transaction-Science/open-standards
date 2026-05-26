//! Energy schema.
//!
//! Joules are the native unit of cost. Every operation has a measured cost.
//! See spec 05.

use crate::backend::BackendId;
use crate::op::OpKind;
use std::collections::HashMap;
use std::time::Duration;

/// Identifier of an energy source (e.g., CPU package, GPU, ANE).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnergySourceId(pub u32);

/// Identifier of a compute unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeUnitId(pub u32);

/// Identifier of a memory tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryTierId(pub u32);

/// Identifier of an interconnect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InterconnectId(pub u32);

/// Estimated joule cost (compile-time, used by the scheduler).
#[derive(Debug, Clone, Copy)]
pub struct JouleEstimate {
    pub mean_joules: f64,
    pub stddev_joules: f64,
    pub source: EstimateSource,
}

#[derive(Debug, Clone, Copy)]
pub enum EstimateSource {
    /// Calibrated from this machine's measurements.
    Calibration,
    /// Predicted by a model (interpolation across calibration data).
    Model,
    /// Default heuristic; least trustworthy.
    Default,
}

/// Measured joule cost (run-time).
#[derive(Debug, Clone, Copy)]
pub struct JouleMeasurement {
    pub joules: f64,
    pub energy_source: EnergySourceId,
    pub measurement_window: Duration,
    /// 1.0 if this measurement is exclusive to one operation;
    /// less than 1.0 when concurrent ops share an energy source.
    pub attribution_confidence: f64,
}

/// Aggregate joule accounting for a request.
#[derive(Debug, Clone)]
pub struct JouleAccounting {
    pub total_joules: f64,
    pub by_op_kind: HashMap<OpKind, f64>,
    pub by_compute_unit: HashMap<ComputeUnitId, f64>,
    pub by_memory_tier: HashMap<MemoryTierId, f64>,
    pub by_interconnect: HashMap<InterconnectId, f64>,
    pub by_backend: HashMap<BackendId, f64>,
    pub deterministic_joules: f64,
    pub stochastic_joules: f64,
}

impl JouleAccounting {
    pub fn empty() -> Self {
        Self {
            total_joules: 0.0,
            by_op_kind: HashMap::new(),
            by_compute_unit: HashMap::new(),
            by_memory_tier: HashMap::new(),
            by_interconnect: HashMap::new(),
            by_backend: HashMap::new(),
            deterministic_joules: 0.0,
            stochastic_joules: 0.0,
        }
    }
}

/// Per-request joule budget.
#[derive(Debug, Clone, Copy)]
pub struct JouleBudget {
    pub max_joules: f64,
    pub on_exceed: BudgetPolicy,
}

#[derive(Debug, Clone, Copy)]
pub enum BudgetPolicy {
    /// Reject the request when budget would be exceeded.
    Reject,
    /// Fall back to cheaper paths (smaller models, cached results).
    Degrade,
    /// Charge the overage to the caller; complete the request.
    Charge,
}
