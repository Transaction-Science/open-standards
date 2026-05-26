//! Error types.

use crate::backend::BackendId;
use crate::op::OpKind;
use crate::tensor::{Dtype, Shape};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Type(TypeError),
    Execution(ExecutionError),
    Determinism(DeterminismError),
    Topology(TopologyError),
}

#[derive(Debug)]
pub enum TypeError {
    ArityMismatch { op: OpKind, expected: usize, got: usize },
    DtypeMismatch { op: OpKind, expected: Dtype, got: Dtype, input_index: usize },
    ShapeMismatch { op: OpKind, expected: Shape, got: Shape, input_index: usize },
    UnsupportedDtype { op: OpKind, dtype: Dtype },
}

#[derive(Debug)]
pub enum ExecutionError {
    KernelFailed { op: OpKind, backend: BackendId, reason: String },
    OutOfMemory { tier: String, bytes_requested: u64 },
    BudgetExceeded { joules_used: f64, joules_budget: f64 },
    TopologyMismatch { reason: String },
}

#[derive(Debug)]
pub enum DeterminismError {
    StochasticOpInDeterministicGraph { op: OpKind },
    UnseededSampler,
    NonDeterministicKernelSelected { op: OpKind, backend: BackendId },
}

#[derive(Debug)]
pub enum TopologyError {
    NoBackendForOp { op: OpKind },
    CalibrationFailed { reason: String },
    EnergyCounterUnavailable { source: String },
}

impl From<TypeError> for Error { fn from(e: TypeError) -> Self { Error::Type(e) } }
impl From<ExecutionError> for Error { fn from(e: ExecutionError) -> Self { Error::Execution(e) } }
impl From<DeterminismError> for Error { fn from(e: DeterminismError) -> Self { Error::Determinism(e) } }
impl From<TopologyError> for Error { fn from(e: TopologyError) -> Self { Error::Topology(e) } }
