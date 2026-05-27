//! JouleClaw L4 — cross-model verification tier.
//!
//! When you really do need the answer and you're willing to pay 4 J
//! for it: dispatch the query to ≥2 *different* LLM backends in
//! parallel, then ask an [`AgreementChecker`] whether the candidates
//! agree. Agreement → very-high-confidence answer; disagreement →
//! refuse, forcing the cascade to fall through (or fail).
//!
//! Cost model (sum across configured backends):
//!
//! ```text
//! joules           = Σ backend.estimate_joules(request)   ≈ 4 J typical
//! latency          = 6 s   (two ~3 s cheap-LLM calls, run in parallel)
//! confidence_floor = 0.9   (agreement-only; refused otherwise)
//! ```
//!
//! Constructors require ≥2 backends; the cascade treats this tier as
//! the most expensive non-meta tier and never reaches it unless cheaper
//! tiers have already refused.
//!
//! ## Ported subset
//!
//! The donor `verity-cascade::layers::l4_verification` is a single-
//! shot "ask a second model to grade the first" verifier with
//! deterministic pre-filters. The JouleClaw port generalises that
//! pattern to N≥2 cross-model voting with a pluggable
//! [`AgreementChecker`], because the JouleClaw cascade reaches L4
//! *without* a prior L3 answer (L3 either resolved or refused). The
//! deterministic pre-filters in the donor (refusal-pattern, empty,
//! short-answer) are tier-shape-neutral and will be re-introduced as
//! a [`AgreementChecker`] decorator in a follow-up.
//!
//! ## Backend trait
//!
//! [`LlmBackend`] is defined locally in [`crate::llm`] as a
//! placeholder while the sibling `jouleclaw-llm-cheap` crate is being
//! ported in parallel. The trait shape was specified to be identical
//! to that crate's; once it lands on disk this module's `llm.rs`
//! should be deleted and `LlmBackend` re-exported from there.

#![forbid(unsafe_code)]

mod checker;
mod llm;
mod tier;

pub use checker::{
    AgreementChecker, AgreementVerdict, JaccardChecker, StringMatchChecker,
    jaccard, normalise,
};
pub use llm::{
    FailingBackend, LlmBackend, LlmError, LlmRequest, LlmResponse, StaticBackend,
};
pub use tier::{
    VerificationTier, VerificationTierError, VERIFICATION_TIER_CONFIDENCE_FLOOR,
    VERIFICATION_TIER_LATENCY, VERIFICATION_TIER_MAX_TOKENS,
};
