//! Smart Byte CRDT engine.
//!
//! Provides Conflict-Free Replicated Data Types (CRDTs) for collaborative
//! cargo: shared documents, shared inventories, collaborative budgets.
//! Every replica converges on the same state without coordination.
//!
//! ## Modules
//!
//! * [`types`] — native CRDTs: LWW register, G/PN counters, OR-set, LWW
//!   map, RGA list, two-phase set, plus the [`Crdt`](types::Crdt) trait.
//! * [`hlc`] — Hybrid Logical Clock (Kulkarni et al. 2014) used as the
//!   universal timestamp for LWW-style merges.
//! * [`vector_clock`] — vector clocks for causal-history comparisons and
//!   delta computation.
//! * [`document`] — tree-structured [`CrdtDocument`](document::CrdtDocument)
//!   with addressable paths.
//! * [`ops`] — content-addressed operations and the op log.
//! * [`sync`] — delta computation and idempotent application.
//! * [`cargo_bridge`] — wraps a [`CrdtDocument`](document::CrdtDocument)
//!   as a Smart Byte `Cargo::Custom` envelope.
//! * [`sharing`] — Simplex (one writer, many readers) and Multiplex
//!   (many writers) sharing patterns.
//! * [`automerge_interop`] — feature `automerge-interop`.
//! * [`yjs_interop`] — feature `yjs-interop`.
//! * [`error`] — typed errors.
//!
//! ## Design
//!
//! Native CRDT types do not depend on `automerge` or `yrs`; those crates
//! are used only behind their feature flags for ecosystem interop.
//! Operations are content-addressed (BLAKE3 over canonical CBOR) and
//! ordered by HLC, mirroring Smart Byte's envelope model.

#![forbid(unsafe_code)]

pub mod cargo_bridge;
pub mod document;
pub mod error;
pub mod hlc;
pub mod ops;
pub mod sharing;
pub mod sync;
pub mod types;
pub mod vector_clock;

#[cfg(feature = "automerge-interop")]
pub mod automerge_interop;

#[cfg(feature = "yjs-interop")]
pub mod yjs_interop;

pub use document::{CrdtDocument, CrdtNode, DocumentId, Value};
pub use error::{CrdtError, Result};
pub use hlc::{HybridLogicalClock, HlcClock, ReplicaId};
pub use ops::{Op, OpId, OpKind, Path};
pub use sharing::{ShareMultiplex, ShareSimplex};
pub use sync::{apply, diff};
pub use types::{
    Crdt, CrdtId, GCounter, LwwMap, LwwRegister, OrSet, PnCounter, RgaList, TwoPhaseSet,
    UniqueTag,
};
pub use vector_clock::VectorClock;
