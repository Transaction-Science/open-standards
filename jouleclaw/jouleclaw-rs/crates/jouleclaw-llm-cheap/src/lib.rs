//! JouleClaw L3 — cheap-LLM tier.
//!
//! Ported from `verity-cascade::layers::l3_llm_cheap` (donor ~521 LOC). The
//! donor talks directly to a `verity_llm::router::LlmRouter` and a
//! `verity_joule::sampler::PowerSampler`; this open-standard port abstracts
//! both behind the [`LlmBackend`] trait so any backend (HTTP, gRPC, on-host
//! shim, fake) can plug in without dragging private deps into the public
//! crate.
//!
//! ## Tier role
//!
//! L3 (`TierId::L3(L3ModelId(model_id))`) — the cheapest acceptable
//! stochastic-model dispatch in the cascade. Three-phase energy:
//!
//! * client send  — micro/milli-joules (already accounted by the runtime)
//! * remote compute — the bulk; reported by the backend if available, else
//!   the backend's [`LlmBackend::typical_joules_per_call`] estimate
//! * client receive — micro/milli-joules (folded into the client share)
//!
//! Donor's modeled total: ~2,001,000 µJ ≈ 2.001 J — the default
//! [`EchoBackend`] reports exactly that for parity with the donor's
//! conformance fixtures.
//!
//! ## Architecture
//!
//! The donor is `async` and routes via `LlmRouter::complete_eco()`. We
//! collapse that to a synchronous [`LlmBackend::complete`] call — the
//! cascade [`jouleclaw_cascade::tier::Tier`] surface is itself sync, so the
//! caller's runtime is what bridges to async (e.g. by blocking the
//! backend's `complete` impl on `Runtime::block_on` for tokio-based
//! providers). Streaming, structured contrast, SSM drafts, conversation
//! history, and JCI persona injection — all pieces of the donor that were
//! private-IP-adjacent — are dropped from the open-standard surface.
//! Consumers that want them re-introduce them via a richer backend.
//!
//! ## Energy provenance
//!
//! Backends MUST set [`LlmResponse::energy_joules`] honestly: `Some(j)`
//! when the value came from a real upstream meter (HwShunt /
//! ModelBased) and `None` when no meter exists. On `None` the tier
//! falls back to [`LlmBackend::typical_joules_per_call`] — an
//! [`jouleclaw_energy::Provenance::Estimator`]-class number — and tags
//! the `Answer` accordingly via [`LlmCheapTier::report_provenance`].

#![forbid(unsafe_code)]

mod backend;
mod tier;

pub use backend::{
    EchoBackend, FinishReason, LlmBackend, LlmError, LlmRequest, LlmResponse,
    DEFAULT_TYPICAL_JOULES,
};
pub use tier::{
    LlmCheapTier, DEFAULT_CONFIDENCE_FLOOR, DEFAULT_LATENCY, DEFAULT_MODEL_ID,
};
