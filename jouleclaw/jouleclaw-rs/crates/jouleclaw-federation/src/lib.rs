//! # jouleclaw-federation
//!
//! L2 of the JouleClaw cascade — **multi-provider federated search**.
//!
//! Fans a single text query out to N pluggable search providers
//! (web-search APIs, vertical APIs, local indices) **in parallel**,
//! fuses the per-provider hit lists into a single ranked set, and
//! returns the fused set as `AnswerOutput::Structured(json)` so the
//! downstream L2.5 reranker / L3 reader can consume it without
//! re-issuing network calls.
//!
//! ## Doctrine
//!
//! - **Fresh retrieval beats frozen weights.** The federation is the
//!   first tier in the cascade that touches the live world — it
//!   exists so the L3/L4 model tiers are not asked to remember the
//!   internet.
//! - **Inference is the last resort.** Federation never invents an
//!   answer; it returns a hit list. If the hit list is empty or the
//!   fuser produces a low-confidence top result, the tier refuses
//!   and the cascade continues.
//! - **Honest provenance.** Each provider self-reports
//!   `typical_joules_per_call`; the tier sums those into its energy
//!   estimate and tags the spend [`Provenance::Estimator`] —
//!   network energy is not measured by a hardware shunt.
//! - **No async runtime.** Providers are dispatched via
//!   `std::thread::scope` so the crate works in `no_tokio` deployments
//!   (Pi-class, embedded harness, CLI smoke tests).
//!
//! ## Wiring it up
//!
//! ```rust,no_run
//! use jouleclaw_cascade::tier::Cascade;
//! use jouleclaw_cascade::types::L2ModelId;
//! use jouleclaw_federation::{Federation, MockProvider, LinearFuser};
//!
//! let providers: Vec<Box<dyn jouleclaw_federation::SearchProvider>> = vec![
//!     Box::new(MockProvider::named("brave")),
//!     Box::new(MockProvider::named("wikipedia")),
//! ];
//! let federation = Federation::new(L2ModelId(0), providers, Box::new(LinearFuser::default()));
//!
//! let mut cascade = Cascade::new();
//! cascade.register(Box::new(federation));
//! ```
//!
//! ## What was ported
//!
//! Ported from
//! `joulesperbit/crates/verity-cascade/src/layers/l2_federation.rs`
//! (~184 LOC). The donor's confidence formula (top fused score plus a
//! provider-diversity bonus) is preserved in spirit but rescaled to
//! the JouleClaw `[0.0, 1.0]` confidence range.
//!
//! What was replaced:
//!
//! - **`OmniRouter` + the 28-provider `verity-federation` crate** — replaced by
//!   the consumer-supplied [`SearchProvider`] trait. JouleClaw ships no
//!   live providers in this crate; the donor's adapters (Brave, Bing,
//!   Wikipedia, arXiv, …) belong in downstream consumer crates.
//! - **`async` + `tokio`** — replaced by `std::thread::scope` parallel
//!   dispatch.
//! - **`PowerSampler`** — replaced by the per-provider self-reported
//!   joule estimate; the runtime's outer meter measures actual spend.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod fuser;
pub mod handoff;
pub mod provider;
pub mod tier;

pub use fuser::{FusedHit, FuseReport, Fuser, LinearFuser, RrfFuser};
pub use handoff::{
    AgentError, AgentInput, AgentResponse, CallableAgent, Capability,
    CheapestCapable, HandoffRegistry, HandoffSelector,
};
pub use provider::{MockProvider, ProviderError, SearchHit, SearchProvider};
pub use tier::{
    Federation, FederationError, FederationOutput, FederationProviderReport,
    DEFAULT_FEDERATION_LATENCY, FEDERATION_CONFIDENCE_FLOOR, MAX_HITS_OUT,
};
