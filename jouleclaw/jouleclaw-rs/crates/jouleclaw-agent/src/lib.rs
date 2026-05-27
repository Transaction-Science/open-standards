//! L6 — agent (multi-step recursive cascade).
//!
//! Most tiers answer a query in one shot. L6 is different: it
//! *decomposes* a complex query into sub-queries, dispatches each one
//! back through the cascade, and *composes* the partial answers into a
//! single result. Its energy is the sum of the sub-dispatches — which is
//! why it sits near the top of the cascade and only fires when the
//! cheaper single-shot tiers have refused.
//!
//! ## Why a trait shim instead of a `Runtime` handle
//!
//! The agent must call "the cascade" to resolve each sub-query, but the
//! cascade also *contains* the agent — a direct `Runtime` field would be
//! a dependency cycle. [`AgentCascade`] breaks it: the consumer wires
//! their live `Runtime` (or a test double) behind the trait, and the
//! agent talks to that. The agent crate stays free of any concrete
//! runtime.
//!
//! ## Pluggable everywhere
//!
//! - [`AgentPlanner`] decides how to split a query. Default
//!   [`KeywordPlanner`] splits on conjunctions ("and", "then", ";").
//! - [`Composer`] joins sub-answers. Default [`Concatenator`].
//! - [`AgentCascade`] resolves a single sub-query.
//!
//! All three are traits, so an L6 deployment can range from this
//! keyword-split reference all the way to an LLM-driven planner without
//! touching the tier logic.

#![forbid(unsafe_code)]

mod cascade_trait;
mod composer;
mod planner;
mod tier;

pub use cascade_trait::{AgentCascade, MockCascade};
pub use composer::{Composer, Concatenator};
pub use planner::{AgentPlanner, KeywordPlanner, SubQuery};
pub use tier::{AgentError, AgentTier, AGENT_TYPICAL_JOULES};
