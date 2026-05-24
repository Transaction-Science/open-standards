//! EOC speculative decoding.
//!
//! Speculative decoding (Leviathan, Kalman & Matias 2022, *Fast Inference
//! from Transformers via Speculative Decoding*; Chen et al. 2023,
//! *Accelerating Large Language Model Decoding with Speculative
//! Sampling*) lets a small "draft" model propose `K` tokens that a large
//! "target" model verifies in a *single* forward pass. The total joule
//! cost per generated token drops sharply: the draft is cheap, and the
//! target only runs once per accepted block instead of once per token.
//!
//! This crate ingests speculative decoding into EOC at two levels:
//!
//! 1. **Orchestration.** Any pair of [`NeuralBackend`](eoc_neural::NeuralBackend)
//!    implementations — vendor APIs from `eoc-vendor-api`, on-host
//!    runtimes from `eoc-local`, or [`synthetic`] test backends — can be
//!    glued into a [`SpeculativeDecoder`](orchestrator::SpeculativeDecoder).
//!    Draft and target joule costs are tracked separately and surfaced
//!    on the resulting [`Generation`](orchestrator::Generation).
//!
//! 2. **Canonical algorithms.** The [`algorithms`] module ships the
//!    standard recipes: vanilla greedy speculative decoding, SpS with
//!    temperature (Chen et al.), lookahead decoding (Jacobi iteration +
//!    n-gram cache), and document-as-stub skeletons for EAGLE and
//!    Medusa. EAGLE and Medusa require backend-internal hooks (target
//!    activations, parallel decoding heads) that the
//!    [`NeuralBackend`](eoc_neural::NeuralBackend) trait deliberately
//!    abstracts away; the stubs nail down the trait surface so a future
//!    implementation slotted behind a local backend can plug in without
//!    touching the orchestrator.
//!
//! ## Joule attribution
//!
//! Every draft proposal and every target verification reports its own
//! joule cost. The orchestrator sums them, and [`SpeculativeBackend`](wrapper::SpeculativeBackend)
//! exposes the total as a single [`JouleCost`](eoc_core::JouleCost) so
//! the cascade can keep using one number per query. The breakdown
//! (draft µJ, target µJ, target forward-pass count, acceptance rate)
//! stays available on [`Generation`](orchestrator::Generation) for
//! callers who want to see where the energy went.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod algorithms;
pub mod draft;
pub mod error;
pub mod orchestrator;
pub mod sampler;
pub mod synthetic;
pub mod target;
pub mod wrapper;

pub use algorithms::{
    SpeculativeAlgorithm, eagle::LocalEagleDraft, lookahead::LookaheadDecoding,
    medusa::LocalMedusaDraft, sps_with_temperature::SpsWithTemperature, vanilla::VanillaSpeculative,
};
pub use draft::{DraftModel, DraftSequence, TokenId};
pub use error::{SpecDecodeError, SpecDecodeResult};
pub use orchestrator::{Generation, SpeculativeDecoder};
pub use sampler::{GreedySampler, Sampler, TemperatureSampler, TopKSampler, TopPSampler};
pub use synthetic::{SyntheticDraft, SyntheticTarget};
pub use target::{TargetModel, VerificationResult};
pub use wrapper::SpeculativeBackend;
