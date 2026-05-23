//! # `op-fraud` — On-device fraud scoring
//!
//! Critical for `OpenPay`'s A2A rails. Card chargebacks give merchants a
//! 60-120 day window to dispute fraud. `FedNow`, PIX, SEPA Instant transfers
//! are **irrevocable**. A fraud decision after submit is useless. The
//! orchestrator must score *before* it routes.
//!
//! ## Design
//!
//! Three layers:
//!
//! 1. **Feature extraction** ([`features`]) — Deterministically turns a
//!    [`op_core::Payment<Created>`] plus context into a fixed-length
//!    `[f32; FEATURES]` vector with hashed identifiers. No raw PAN, name,
//!    or account number reaches the model.
//!
//! 2. **Scoring** ([`scorer`]) — A [`Scorer`] trait with three ships:
//!    - [`HeuristicScorer`] — Rule-based, pure Rust, no runtime
//!      dependencies. Ships in every build and gives a defensible
//!      baseline.
//!    - [`onnx::OnnxScorer`] — Loads an operator-trained ONNX model via
//!      the `ort` crate. Feature-gated behind `onnx`. Best path when the
//!      operator's data team trained their model in `PyTorch` / TensorFlow
//!      / sklearn and exported to ONNX. Requires shipping the ONNX
//!      Runtime shared library alongside the binary.
//!    - [`burn_backend::BurnScorer`] — Loads a Burn-trained MLP record.
//!      Feature-gated behind `burn-backend`. Pure Rust, no shared
//!      library. Best path for WASM, kiosk-Linux, and any deployment
//!      where the ~10-30 MB ONNX Runtime is too large or the operator
//!      wants to avoid the FFI surface entirely.
//!
//! 3. **Decision** ([`decision`]) — Maps a raw `score: f32` in `[0.0, 1.0]`
//!    to a [`FraudDecision`] (`Approve`, `Review`, `Decline`, `Freeze`)
//!    using calibrated thresholds.
//!
//! ## Privacy
//!
//! By construction, the input to the scorer is a `[f32; FEATURES]` array
//! with no string fields. Identifiers (account numbers, names, geo
//! locations) are hashed via SHA-256 and the upper 32 bits projected to a
//! single `f32`. The model cannot reconstruct PII even if exfiltrated.
//!
//! ## What this crate does NOT do
//!
//! - It does not train models. Operators train offline on their own
//!   fraud data and export ONNX.
//! - It does not call any external service. All scoring is local.
//! - It does not log scores or decisions. The orchestrator chooses
//!   whether and how to persist.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]

pub mod context;
pub mod decision;
pub mod error;
pub mod features;
pub mod scorer;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "burn-backend")]
pub mod burn_backend;

pub use context::ScoringContext;
pub use decision::{FraudDecision, Thresholds};
pub use error::{Error, Result};
pub use features::{FEATURES, FeatureVector, extract_features};
pub use scorer::{HeuristicScorer, Scorer};
