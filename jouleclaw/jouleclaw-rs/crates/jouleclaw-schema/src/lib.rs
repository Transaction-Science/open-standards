//! Edge-First AI Architecture v6 schemas.
//!
//! Pure data types implementing the contracts in
//! `edge-architecture/edge_ai_architecture_spec_v6.md`. Every schema
//! carries `schema_version` (first field) and `metadata` (last field)
//! per Section 3.1; timestamps are `chrono::DateTime<Utc>`; instance
//! identifiers are UUIDv4.
//!
//! The v6 spec was written in Python (Pydantic). This crate is the
//! direct Rust port of that schema layer. Where v6 defers to "v1 spec"
//! for AuthorityRecord / AtomicClaim / EntailmentResult /
//! VerificationReport / Answer, those types are reconstructed from
//! their usage in Sections 5-8 and flagged in the relevant module docs.

pub mod anytime;
pub mod answer;
pub mod atomic_claim;
pub mod authority;
pub mod common;
pub mod entailment;
pub mod epistemic;
pub mod invariants;
pub mod knowledge_axes;
pub mod query_plan;
pub mod rap;
pub mod retrieved_item;
pub mod system_capabilities;
pub mod verification;

pub use anytime::{AnytimeResult, CompletionStatus};
pub use answer::{Answer, AnswerSegment, AnswerStatus, Provenance, Refusal};
pub use atomic_claim::{AtomicClaim, ClaimStakes};
pub use authority::{AuthorityRecord, AuthorityTier};
pub use common::{Metadata, SchemaVersion};
pub use entailment::{EntailmentLabel, EntailmentProbabilities, EntailmentResult};
pub use epistemic::{ClaimAttribution, EpistemicMode, PriorClaimClass};
pub use invariants::{Invariant, InvariantsVerified};
pub use knowledge_axes::{GranularityClass, KnowledgeAxes, ScopeClass, TemporalStabilityClass};
pub use query_plan::{
    Budget, Constraints, Intent, Modality, OriginalQuery, PlanInvariants, QueryPlan, SubQuery,
    TemporalScope,
};
pub use rap::{RapStep, RapStepCondition, ReactiveActionPackage};
pub use retrieved_item::{
    Attribution, Content, ExcerptSpan, FreshnessClass, RetrievalContext, RetrievalMethod,
    RetrievedItem, ScoreType, SourceType, Temporal,
};
pub use system_capabilities::{
    CapabilityStatus, ReasonerCapability, RetrieverCapability, SystemCapabilities,
};
pub use verification::{
    RecommendedAction, RecoveryAction, VerificationAction, VerificationReport, Violation,
    ViolationSeverity,
};
