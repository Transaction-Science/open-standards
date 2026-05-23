//! EOC learned router — predictive stage selection for the four-stage cascade.
//!
//! The default `eoc_cascade::Cascade` walks `cache → kv → graph → neural` in
//! strict order. That's optimal when most queries land in the cheap stages.
//! For workloads where many queries unavoidably need the neural stage, walking
//! the cheaper stages first is wasted joules. A *learned router* predicts the
//! cheapest stage that is likely to satisfy the query and lets us **skip
//! ahead** when the prediction is confident.
//!
//! This crate ships three orthogonal router families plus a threshold policy
//! and an end-to-end cascade extension:
//!
//! - [`matrix_factorization::MfRouter`] — RouteLLM-style latent factor model
//!   (ICLR 2025). Each query cluster gets a `k`-dim latent vector; each stage
//!   gets a `k`-dim latent vector; the dot product predicts success
//!   probability. Trained by SGD on `(query_embedding, stage, success)`.
//! - [`classifier::LogRegRouter`] — multinomial logistic regression over query
//!   embeddings, outputting a softmax over the four stages.
//! - [`bandit::LinUcbRouter`] / [`bandit::ThompsonSamplingRouter`] — online
//!   bandit learners for the regime where ground-truth labels arrive late
//!   (downstream feedback).
//! - [`threshold::ThresholdPolicy`] — turn a `StagePrediction` into a concrete
//!   "skip-ahead to stage X" decision.
//! - [`inference::LearnedCascade`] — wraps an `eoc_cascade::Cascade` and
//!   consults the router before dispatching.
//!
//! Everything is `nalgebra`-only — no Burn, no Candle, no ONNX, WASM-ready.
//!
//! ```no_run
//! use eoc_route_learned::{
//!     classifier::LogRegRouter,
//!     training::{Example, TrainingConfig, train_logreg},
//! };
//! use eoc_core::Stage;
//!
//! let examples: Vec<Example> = vec![]; // load from JSONL
//! let cfg = TrainingConfig::default();
//! let router = train_logreg(&examples, cfg);
//! let _ = router; // wire into a LearnedCascade
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod bandit;
pub mod classifier;
pub mod error;
pub mod evaluation;
pub mod inference;
pub mod matrix_factorization;
pub mod router;
pub mod threshold;
pub mod training;

pub use bandit::{LinUcbArm, LinUcbRouter, ThompsonSamplingRouter};
pub use classifier::LogRegRouter;
pub use error::{Error, Result};
pub use evaluation::{RouterMetrics, evaluate};
pub use inference::LearnedCascade;
pub use matrix_factorization::MfRouter;
pub use router::{LearnedRouter, RouterState, StagePrediction};
pub use threshold::{ThresholdDecision, ThresholdPolicy};
pub use training::{Example, TrainingConfig, train_logreg, train_mf};
