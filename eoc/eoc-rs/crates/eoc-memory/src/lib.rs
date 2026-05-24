//! EOC long-term memory primitives.
//!
//! `eoc-memory` provides the memory substrate for agentic loops in
//! the EOC stack: episodic event logs, semantic knowledge graphs,
//! working-memory scratchpads, conversational summary buffers,
//! sliding windows, dream-cycle consolidation, the Ebbinghaus
//! forgetting curve, retrieval triangulation (similarity + recency
//! + frequency) and a MemGPT-style hierarchical shape.
//!
//! ## Modules
//!
//! * [`memory`] — base [`Memory`] trait, [`MemoryKind`], identifiers.
//! * [`episodic`] — event log with temporal indexing.
//! * [`semantic`] — entity-relation triple store / knowledge graph.
//! * [`working`] — scratchpad with attention budget.
//! * [`summary`] — rolling summary buffer.
//! * [`window`] — sliding-window conversational memory.
//! * [`consolidate`] — dream-cycle batch: episodic → semantic.
//! * [`forget`] — Ebbinghaus time-decay scorer.
//! * [`retrieve`] — similarity + recency + frequency triangulation.
//! * [`inject`] — render memory items into a prompt context block.
//! * [`memgpt`] — MemGPT hierarchical memory (main / recall / archival).
//! * [`error`] — error / result aliases.
//!
//! Everything is deterministic given fixed inputs and a fixed clock,
//! and the crate is `#![forbid(unsafe_code)]`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod consolidate;
pub mod episodic;
pub mod error;
pub mod forget;
pub mod inject;
pub mod memgpt;
pub mod memory;
pub mod retrieve;
pub mod semantic;
pub mod summary;
pub mod window;
pub mod working;

pub use consolidate::{ConsolidateConfig, ConsolidationReport, consolidate};
pub use episodic::{Episode, EpisodicLog, TimeIndex};
pub use error::{MemoryError, MemoryResult};
pub use forget::{EbbinghausScorer, ForgetConfig};
pub use inject::{InjectConfig, InjectedContext, inject};
pub use memgpt::{MemGpt, MemGptConfig, Tier};
pub use memory::{EpisodeId, Memory, MemoryItem, MemoryKind, MemoryRef};
pub use retrieve::{RetrievalConfig, RetrievalScore, Retriever};
pub use semantic::{SemanticGraph, Triple};
pub use summary::SummaryBuffer;
pub use window::SlidingWindow;
pub use working::Scratchpad;
