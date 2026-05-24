//! `eoc-eval` — canonical LLM evaluation harnesses for the EOC stack.
//!
//! Each module wraps one published evaluation suite (MMLU, GPQA,
//! HumanEval, ...) behind a uniform [`Harness`] trait so the
//! [`runner::EvalRunner`] can drive any harness through any
//! [`eoc_neural::NeuralBackend`] and emit a comparable score plus
//! `joules-per-correct`.
//!
//! ## Quick start
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use eoc_eval::{mmlu::Mmlu, runner::EvalRunner};
//! # use eoc_neural::EchoBackend;
//! # async fn ex() {
//! let runner = EvalRunner::new(
//!     Box::new(Mmlu::new()),
//!     Arc::new(EchoBackend::new()),
//! )
//! .with_model("echo");
//! let report = runner.run().await.expect("run");
//! println!("{} score={:.3} J/correct={}", report.harness, report.score, report.joules_per_correct);
//! # }
//! ```
//!
//! ## Crate layout
//!
//! - [`harness`] — the [`harness::Harness`] trait, [`harness::Metric`],
//!   [`harness::DatasetSource`], [`harness::EvalCase`].
//! - [`runner`] — drives a harness end-to-end against a backend.
//! - One module per published harness (`mmlu`, `mmlu_pro`, `gpqa`,
//!   `humaneval`, `bbh`, `ifeval`, `alpaca_eval`, `agi_eval`,
//!   `hellaswag`, `arc`, `truthfulqa`, `boolq`, `gsm8k`, `math`,
//!   `winogrande`).
//! - [`dataset_loader`] — HuggingFace Hub fetcher (behind the
//!   `download` feature).
//! - [`builtin_samples`] — small hand-curated sample data embedded in
//!   the crate; available without any features.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod builtin_samples;
pub mod error;
pub mod harness;
pub mod mcq;
pub mod runner;

pub mod agi_eval;
pub mod alpaca_eval;
pub mod arc;
pub mod bbh;
pub mod boolq;
pub mod gpqa;
pub mod gsm8k;
pub mod hellaswag;
pub mod humaneval;
pub mod ifeval;
pub mod math;
pub mod mmlu;
pub mod mmlu_pro;
pub mod truthfulqa;
pub mod winogrande;

#[cfg(feature = "download")]
pub mod dataset_loader;

pub use error::{EvalError, Result};
pub use harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, IfEvalConstraint, Metric, Response,
};
pub use runner::{EvalReport, EvalRunner};
