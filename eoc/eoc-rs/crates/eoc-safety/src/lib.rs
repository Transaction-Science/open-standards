//! `eoc-safety` — AI safety primitives for the EOC stack.
//!
//! This crate provides composable, dependency-light building blocks
//! for input/output safety around LLM applications:
//!
//! - [`injection`] — prompt-injection signature detector
//!   (PromptInject / Garak / llm-guard patterns, all Apache-2.0 / MIT).
//! - [`jailbreak`] — DAN / DUDE / Grandma / STAN / AIM / developer-mode
//!   family detector.
//! - [`pii`] — Presidio-style PII redactor (email, SSN, phone,
//!   Luhn-validated credit card, IPv4/v6, US address, common given
//!   names).
//! - [`toxicity`] — pluggable toxicity classifier (trait +
//!   lexicon baseline).
//! - [`bias`] — gender / race / age / occupation / religion stereotype
//!   detector (trait + lexicon baseline).
//! - [`nsfw`] — text NSFW detector with hard-reject for
//!   sexual-content-involving-minors.
//! - [`constitutional`] — Anthropic-style critique→revise loop with a
//!   pluggable [`constitutional::CritiqueModel`].
//! - [`guard`] — composable input and output pipelines a la Llama
//!   Guard / NeMo Guardrails.
//! - [`structure`] — JSON-Schema-subset validator for structured
//!   model output.
//! - [`rate_limit`] — token-bucket rate limiter with abuse detection.
//! - [`red_team`] — replay harness for attack corpora plus a small
//!   bundled smoke set.
//!
//! All baselines are intentionally deterministic and CPU-cheap. Each
//! detector exposes a trait so callers can swap in a learned
//! classifier without changing the [`guard`] pipeline.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bias;
pub mod constitutional;
pub mod error;
pub mod guard;
pub mod injection;
pub mod jailbreak;
pub mod nsfw;
pub mod pii;
pub mod rate_limit;
pub mod red_team;
pub mod structure;
pub mod toxicity;

pub use error::{Result, SafetyError};
