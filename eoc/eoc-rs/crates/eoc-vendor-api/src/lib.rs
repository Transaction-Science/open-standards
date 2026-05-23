//! EOC vendor LLM API backends.
//!
//! This crate ships [`eoc_neural::NeuralBackend`] implementations for the
//! major commercial inference endpoints — Anthropic, OpenAI, Google
//! (Gemini), Mistral, Cohere, Groq, Together, Fireworks — together with
//! the shared joule estimator that attributes energy cost to each
//! inference.
//!
//! Each backend:
//!
//! * Issues a single HTTP request (or SSE stream) per [`Query`](eoc_core::Query).
//! * Returns a [`Response`](eoc_core::Response) tagged with the resolving
//!   stage (`Stage::Neural`) and a [`JouleCost`](eoc_core::JouleCost).
//! * Reports `JouleSource::Estimated` because no commercial vendor today
//!   exposes per-call hardware energy counters; the estimator multiplies
//!   token counts by the per-model coefficients in
//!   [`data/model_energy_profiles.json`](../../data/model_energy_profiles.json).
//!
//! API keys are *never* logged. Structured tracing uses
//! [`tracing::field::Empty`] for credential fields.
//!
//! ## WASM
//!
//! `reqwest` is built with `rustls-tls`, so the crate compiles for
//! `wasm32-unknown-unknown`. Vendor endpoints do not set permissive CORS
//! headers, so direct browser calls require a same-origin proxy.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod anthropic;
pub mod auth;
pub mod cohere;
pub mod config;
pub mod error;
pub mod fireworks;
pub mod google;
pub mod groq;
pub mod joule_estimator;
pub mod mistral;
pub mod openai;
pub mod openai_compat;
pub mod together;

pub use anthropic::AnthropicBackend;
pub use auth::Auth;
pub use cohere::CohereBackend;
pub use config::{RetryPolicy, VendorConfig};
pub use error::{VendorError, VendorResult};
pub use fireworks::FireworksBackend;
pub use google::GoogleBackend;
pub use groq::GroqBackend;
pub use joule_estimator::{DefaultEstimator, JouleEstimator, ModelEnergyProfile};
pub use mistral::MistralBackend;
pub use openai::OpenAiBackend;
pub use together::TogetherBackend;
