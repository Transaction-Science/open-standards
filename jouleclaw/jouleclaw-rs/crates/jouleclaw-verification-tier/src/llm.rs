//! Local placeholder for the `jouleclaw-llm-cheap` LLM backend surface.
//!
//! The sibling crate `jouleclaw-llm-cheap` is being ported in parallel
//! and will define the canonical `LlmBackend` trait plus the
//! `LlmRequest` / `LlmResponse` envelopes. The L4 verification tier
//! lives one step above it in the cascade (we dispatch ≥2 of its
//! backends in parallel and compare answers), so we need the trait
//! visible at *compile time* — even before that crate has settled.
//!
//! The trait shape was specified to both agents to be identical. Once
//! `jouleclaw-llm-cheap` lands on disk and we wire it into `Cargo.toml`,
//! delete this module and re-export the canonical types from there.
//! Until then, this file is the load-bearing definition.

use serde::{Deserialize, Serialize};

/// A single completion request issued to an [`LlmBackend`].
///
/// `prompt` is the (already-rendered) user text. `max_tokens` caps the
/// completion length; backends should treat `0` as "use backend default".
/// `temperature` is the sampling temperature on `[0, 2]`; for
/// verification work the tier always sets `0.0` (deterministic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl LlmRequest {
    /// Build a deterministic, short-answer request — the usual shape
    /// for an L4 verification dispatch.
    pub fn deterministic(prompt: impl Into<String>, max_tokens: u32) -> Self {
        Self {
            prompt: prompt.into(),
            max_tokens,
            temperature: 0.0,
        }
    }
}

/// The completion produced by an [`LlmBackend`].
///
/// `joules` is the self-reported energy spend of this single call,
/// summed across prefill + decode + network. Backends that cannot
/// measure return their best estimator value here; the L4 tier sums
/// these to attribute the dispatch's total joule cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub text: String,
    pub joules: f64,
    pub model_id: String,
}

/// A pluggable LLM completion backend.
///
/// L4 verification holds `Vec<Box<dyn LlmBackend>>` and dispatches all
/// of them in parallel via [`std::thread::scope`]. Implementations must
/// be `Send + Sync` so the tier can hand each backend to a worker
/// thread. Backends should *not* mutate themselves on `complete` —
/// stateful work belongs in a `Mutex` interior — because each call
/// rides on a borrowed reference shared across threads.
pub trait LlmBackend: Send + Sync {
    /// Stable identifier used in [`LlmResponse::model_id`] and in
    /// diagnostics. Should distinguish different *models*, not just
    /// different backend instances — the whole point of L4 is to
    /// dispatch ≥2 *different* models.
    fn model_id(&self) -> &str;

    /// Estimated joule cost of one dispatch of this backend on a
    /// prompt of `request.prompt.len()` bytes. Used by
    /// [`crate::VerificationTier::estimate_cost`] to attribute a
    /// summed cost up-front; the real spend is whatever the backend
    /// reports back in [`LlmResponse::joules`].
    fn estimate_joules(&self, request: &LlmRequest) -> f64;

    /// Run the completion. Returning `Err` causes the verification
    /// tier to refuse the whole dispatch — we cannot certify cross-
    /// model agreement when a participant failed.
    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError>;
}

/// Errors that an [`LlmBackend`] can surface.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("llm backend `{model_id}` failed: {reason}")]
    BackendFailed { model_id: String, reason: String },
    #[error("llm request rejected: {0}")]
    InvalidRequest(String),
    #[error("llm backend timed out after {seconds} s")]
    Timeout { seconds: u32 },
}

/// Convenience static backend used in tests. Returns a fixed string
/// regardless of input. Public so that downstream crates can wire
/// integration tests against `VerificationTier` without spinning up a
/// real inference stack.
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
    fn model_id(&self) -> &str {
        &self.id
    }
    fn estimate_joules(&self, _request: &LlmRequest) -> f64 {
        self.joules_per_call
    }
    fn complete(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        Ok(LlmResponse {
            text: self.reply.clone(),
            joules: self.joules_per_call,
            model_id: self.id.clone(),
        })
    }
}

/// A backend that always errors. Used in tests to exercise the
/// failure path of [`crate::VerificationTier::try_answer`].
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
    fn model_id(&self) -> &str {
        &self.id
    }
    fn estimate_joules(&self, _request: &LlmRequest) -> f64 {
        2.0
    }
    fn complete(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        Err(LlmError::BackendFailed {
            model_id: self.id.clone(),
            reason: "synthetic failure".into(),
        })
    }
}
