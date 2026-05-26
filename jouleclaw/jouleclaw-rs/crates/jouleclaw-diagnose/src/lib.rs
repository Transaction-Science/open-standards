//! The Diagnose Pillar (spec §6).
//!
//! Five modules implement the conflict-directed verification
//! pipeline:
//!
//! - [`atomizer`] — decompose a draft into [`AtomicClaim`]s. Trait +
//!   sentence-splitting default impl; LLM-backed impls land later as
//!   pluggable backends.
//! - [`entailer`] — `Entailer` trait abstracting an NLI engine.
//!   [`entailer::DebertaEntailer`] wraps `jouleclaw_deberta::NliEngine`
//!   so the diagnose pillar runs against the real DeBERTa-v3
//!   model verified end-to-end in phase 4f.
//! - [`valid_answer`] — the 5 ValidAnswerModel checkers (coverage,
//!   grounding, authority, freshness, consistency) from spec §6.1.
//! - [`verdict`] + [`recovery`] — verdict determination (§6.6) and
//!   recovery-action synthesis (§6.5).
//! - [`report`] — the top-level [`verify`] that ties them all
//!   together via conflict-directed search (§6.2): cheap checks
//!   first, focused entailment only on at-risk claims.

pub mod atomizer;
pub mod entailer;
pub mod recovery;
pub mod report;
pub mod valid_answer;
pub mod verdict;

pub use atomizer::{atomize_sentences, AtomizeError, Atomizer, SentenceAtomizer};
pub use entailer::{DebertaEntailer, EntailError, Entailer, FixtureEntailer};
pub use recovery::recovery_actions_for_violations;
pub use report::{verify, VerifyError, VerifyInputs};
pub use valid_answer::{
    authority_violations, consistency_violations, coverage_violations,
    freshness_violations, grounding_violations,
};
pub use verdict::determine_verdict;
