//! The Compose Layer (spec §7).
//!
//! Two stages:
//!
//! - [`draft`] — turn `QueryPlan + items + authority` into a draft
//!   answer expressed as `(segment_id, text)` pairs in verification
//!   order (well-grounded content first). Two impls: a templated
//!   composer (no LLM) for the minimum-viable path, and an LLM-
//!   backed trait surface for future deployments.
//! - [`verified`] — consume the draft + the diagnose pillar's
//!   [`jouleclaw_schema::VerificationReport`] and produce the final
//!   [`jouleclaw_schema::Answer`] (or [`jouleclaw_schema::Refusal`]).
//!   Mechanical: drop unsupported claims, hedge weak ones, label
//!   inferences, surface conflicts.
//! - [`provenance`] — Provenance-as-cache (§7.3) cache-key
//!   derivation from sub-queries.

pub mod cache;
pub mod draft;
pub mod provenance;
pub mod verified;

pub use cache::{cache_key_for_plan, CacheError, CacheStore, CachedAnswer};
pub use draft::{
    DraftComposer, DraftError, DraftSegment, TemplateComposer,
};
pub use provenance::cache_key_for_subquery;
pub use verified::{compose_verified_answer, AnswerOrRefusal, ComposeError, ComposeInputs};
