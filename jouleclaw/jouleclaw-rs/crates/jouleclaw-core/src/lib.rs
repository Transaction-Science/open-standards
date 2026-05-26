//! # jouleclaw-core
//!
//! Core types for JouleClaw: tensors, ops, graphs, determinism, energy.
//!
//! This crate defines the contracts that the rest of the runtime is built against.
//! No implementations live here — only types, traits, and minimal helpers.
//!
//! Foundation crate of the [JouleClaw open standard][jouleclaw], stewarded by
//! Transaction Science. Ported from `pattern-lang/joule-core`.
//!
//! See [`SPEC.md`][spec] in the standard root for the formal specification.
//!
//! [jouleclaw]: https://jouleclaw.transaction.science
//! [spec]: ../../../SPEC.md

pub mod tensor;
pub mod op;
pub mod kernel;
pub mod graph;
pub mod determinism;
pub mod energy;
pub mod backend;
pub mod error;
pub mod hash;
pub mod blocks;

pub use tensor::{Dtype, Shape, Tensor, TensorMeta, TensorRef, TensorView, TensorViewMut, LifetimeTier};
pub use op::{Op, OpKind, OpAttrs};
pub use kernel::{Kernel, KernelId, KernelResult, ExecutionContext};
pub use graph::{Graph, GraphBuilder, NodeId, NodeKind};
pub use determinism::{DeterminismClass, ExecutionTrace, GraphHash, TensorHash};
pub use energy::{JouleEstimate, JouleMeasurement, JouleAccounting, JouleBudget, BudgetPolicy};
pub use backend::{BackendId, Capabilities};
pub use error::{Error, Result, TypeError, ExecutionError};
