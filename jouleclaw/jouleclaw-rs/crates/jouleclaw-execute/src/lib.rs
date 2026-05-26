//! The Execute Pillar (spec §5).
//!
//! Three layers, top-down:
//!
//! - [`orchestrator`] — `execute()` takes a `QueryPlan` plus a RAP
//!   registry and a retriever registry, topologically sorts the
//!   sub-queries, dispatches each level concurrently, enforces the
//!   budget, and accumulates [`jouleclaw_schema::RetrievedItem`]s.
//! - [`rap`] — the RAP executor walks one RAP's steps respecting
//!   conditions (`ALWAYS`, `ON_EMPTY`, `ON_ERROR`, `ON_TIMEOUT`,
//!   `ON_LOW_CONFIDENCE`) and per-step timeouts.
//! - [`retriever`] — the `Retriever` trait every store implements.
//!   [`retrievers::fixture`] is the test/CI shim; [`retrievers::wikidata`]
//!   is the live SPARQL retriever.
//!
//! [`authority`] derives an `AuthorityRecord` for each item by
//! inspecting source identity + provenance metadata.

pub mod authority;
pub mod orchestrator;
pub mod rap;
pub mod retriever;
pub mod retrievers;

pub use authority::{score_authority, AuthorityScorer};
pub use orchestrator::{execute, ExecuteError, ExecutionResult, OrchestratorConfig};
pub use rap::{rap_execute, RapExecError, RapOutcome};
pub use retriever::{Retriever, RetrieverError, RetrieverRegistry};
