//! The Plan Pillar (spec §4).
//!
//! Three modules:
//!
//! - [`understanding`] — `QueryUnderstanding` trait + LLM-backed impl
//!   that turns raw user input into a [`analysis::QueryAnalysis`].
//!   Implementations call out to whichever reasoner the deployment
//!   provides (joule cascade L3, frontier API, …).
//! - [`csp`] + [`planner`] — the constraint-satisfaction planner that
//!   takes a [`analysis::QueryAnalysis`] plus a
//!   [`jouleclaw_schema::SystemCapabilities`] snapshot and produces a
//!   validated [`jouleclaw_schema::QueryPlan`].
//! - [`self_model`] — `SelfModel` that observes retriever performance
//!   and emits a fresh `SystemCapabilities` snapshot.

pub mod analysis;
pub mod csp;
pub mod planner;
pub mod self_model;
pub mod understanding;

pub use analysis::{Entity, QueryAnalysis, RawSubQuery, Relation, StakesSignal};
pub use csp::{CspError, CspSolver};
pub use planner::{plan, PlanError, RetrieverProfile, StoreCatalog};
pub use self_model::{Observation, SelfModel};
pub use understanding::{
    FixtureUnderstanding, QueryUnderstanding, UnderstandingError,
};
