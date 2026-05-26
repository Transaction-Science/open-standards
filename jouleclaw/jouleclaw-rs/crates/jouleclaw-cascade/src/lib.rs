//! Joule cascade — the runtime entry point.
//!
//! `Runtime::answer(query)` walks tiers in cost order. The first tier
//! that can answer within budget at the required quality wins. If no
//! tier fits, the runtime returns a budget-exhausted error.
//!
//! See `specs/r0-overview.md`, `specs/r0.1-query-answer-tier.md`, and
//! `specs/r0.2-budget-determinism.md` for design rationale.
//!
//! R1 deliverable: `Query`, `Answer`, `Tier`, `Cascade`, `Runtime`, and
//! `L0Cache` — the first tier and the cascade walker that calls it.

pub mod types;
pub mod tier;
pub mod l0_cache;
pub mod history;
pub mod router;
pub mod l2_service;
pub mod calibration;
pub mod disk_calibration;
pub mod coord;
pub mod cost;
pub mod coord_route;
pub mod coord_router_impl;
pub mod verification;
pub mod active;
pub mod body;
pub mod body_safety;

pub use coord::{
    Coord, Zone, Entity, Thermo, PrimitiveSet, NamedPrimitive,
    Interface, Verify, Encoding, prebuilt,
};
pub use cost::{
    CostEstimate, MultiSubstrateCost, Substrate, WorkloadShape,
};
pub use coord_route::{CoordPredicate, SortStrategy};
pub use coord_router_impl::{CoordRouter, CoordRule};
pub use disk_calibration::{PersistentCalibration, DiskCalibrationError};
pub use verification::{
    VerificationStatus, VerificationToken, VerificationOutcome,
    VerificationLedger, ResolvedDispatch,
};
pub use active::{ActiveTier, ActiveRegistry};
pub use body::{BodyTier, BodyDispatch, Plan, BodyError, FileWriter};
pub use body_safety::{SafetyPolicy, SafetyState, DenyReason};

pub use types::*;
pub use tier::{Cascade, Runtime, Tier, TierEstimate};
pub use l0_cache::{L0Cache, L0Stats};
pub use history::{
    HistoryLayer, HistoryEntry, HistoryAnswer, HistoryError, HistoryStats,
    EntryKey, answer_to_history, key_for, now_secs,
};
pub use router::{Router, RoutingPlan};
pub use l2_service::{
    EmbeddingService, EmbeddingResult, IntentClassifier, ClassificationResult,
    L2Error, cosine_sim,
};
pub use calibration::{TierCalibration, CalibrationReport};
