//! LLM backend surface for the L4 verification tier.
//!
//! The canonical [`LlmBackend`] trait and its [`LlmRequest`] /
//! [`LlmResponse`] / [`LlmError`] envelopes live in the sibling crate
//! `jouleclaw-llm-cheap` (the L3 tier). L4 sits one step above L3 in the
//! cascade — it dispatches ≥2 of *the same kind of* backend in parallel
//! and compares their answers — so it consumes that crate's trait
//! directly rather than defining its own. We re-export the canonical
//! types here so downstream code can `use jouleclaw_verification_tier::{
//! LlmBackend, ...}` without reaching across crates.
//!
//! Two deterministic reference backends ship for tests and conformance:
//! [`StaticBackend`] (fixed reply, configurable energy) and
//! [`FailingBackend`] (always errors). `jouleclaw-llm-cheap`'s own
//! `EchoBackend` is unsuitable for verification tests because two echoes
//! of the same prompt always agree and cannot be made to disagree.

pub use jouleclaw_llm_cheap::{FinishReason, LlmBackend, LlmError, LlmRequest, LlmResponse};

/// A deterministic backend that returns a fixed reply regardless of
/// input, reporting a configurable per-call energy. Used to script
/// agreement / disagreement scenarios in tests and to give downstream
/// integration tests a backend that needs no inference stack.
#[derive(Debug, Clone)]
pub struct StaticBackend {
    pub id: String,
    pub reply: String,
    pub joules_per_call: f64,
}

impl StaticBackend {
    pub fn new(id: impl Into<String>, reply: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            reply: reply.into(),
            // ~2 J per cheap LLM call — half of the 4 J L4 budget.
            joules_per_call: 2.0,
        }
    }

    pub fn with_joules(mut self, joules: f64) -> Self {
        self.joules_per_call = joules;
        self
    }
}

impl LlmBackend for StaticBackend {
    fn model_name(&self) -> &str {
        &self.id
    }

    fn complete(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        let text = self.reply.clone();
        let output_tokens = text.chars().count() as u32;
        Ok(LlmResponse {
            text,
            finish_reason: FinishReason::Stop,
            input_tokens: 0,
            output_tokens,
            // Static backend models a known per-call cost — report it so
            // the tier attributes real joules rather than the estimator
            // fallback.
            energy_joules: Some(self.joules_per_call),
        })
    }

    fn typical_joules_per_call(&self) -> f64 {
        self.joules_per_call
    }
}

/// A backend that always errors. Exercises the failure path of
/// [`crate::VerificationTier::try_answer`] — any participant failing
/// forces the whole cross-model dispatch to refuse.
#[derive(Debug, Clone)]
pub struct FailingBackend {
    pub id: String,
}

impl FailingBackend {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

impl LlmBackend for FailingBackend {
    fn model_name(&self) -> &str {
        &self.id
    }

    fn complete(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        Err(LlmError::Upstream(format!(
            "synthetic failure from `{}`",
            self.id
        )))
    }

    fn typical_joules_per_call(&self) -> f64 {
        2.0
    }
}
