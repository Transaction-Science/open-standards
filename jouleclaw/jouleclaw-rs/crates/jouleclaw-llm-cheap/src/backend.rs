//! The [`LlmBackend`] trait + supporting request / response types.
//!
//! Backends are consumer-supplied. The open-standard crate ships only
//! [`EchoBackend`], a deterministic fake that returns the prompt prefixed
//! with `"echo: "` — useful for tests, conformance vectors, and as a
//! reference for the smallest possible compliant backend.

use serde::{Deserialize, Serialize};

/// Default "typical joules per call" for backends that do not override it.
///
/// Matches the donor's modeled L3 dispatch cost of ~2,001,000 µJ — three
/// phases (client send + remote compute + client receive) collapsed into a
/// single scalar. Real backends should override
/// [`LlmBackend::typical_joules_per_call`] with the actual catalogued
/// number for their provider/model.
pub const DEFAULT_TYPICAL_JOULES: f64 = 2.001;

/// One inference request submitted to a backend.
///
/// Fields beyond `prompt` mirror the common subset of provider APIs
/// (OpenAI, Anthropic, llama.cpp, vLLM, …) so backends can map them
/// without invention. Grammar is pre-translated by the caller — see
/// `jouleclaw-decode` — and passed through opaquely.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmRequest {
    /// The user prompt (final turn).
    pub prompt: String,
    /// Maximum tokens to generate. Backends may clamp; they SHOULD NOT
    /// silently exceed.
    pub max_tokens: u32,
    /// Sampling temperature. `0.0` = greedy.
    pub temperature: f32,
    /// Stop sequences. The first match terminates generation.
    pub stop: Vec<String>,
    /// Optional system prompt (`None` = backend default).
    pub system: Option<String>,
    /// Optional grammar — opaque to this crate; pre-compiled by
    /// `jouleclaw-decode` (or a peer) and forwarded to constraint-capable
    /// backends. Backends that don't support constrained decoding MUST
    /// silently ignore.
    pub grammar: Option<String>,
}

impl LlmRequest {
    /// Construct a minimal request from prompt + token cap, with the
    /// remaining fields at sensible defaults (temperature 0.3 — matches
    /// the donor — and no stop sequences / system / grammar).
    pub fn from_prompt(prompt: impl Into<String>, max_tokens: u32) -> Self {
        Self {
            prompt: prompt.into(),
            max_tokens,
            temperature: 0.3,
            stop: Vec::new(),
            system: None,
            grammar: None,
        }
    }
}

/// One inference response returned by a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// Generated text.
    pub text: String,
    /// Why generation stopped. Drives the L3 tier's confidence mapping.
    pub finish_reason: FinishReason,
    /// Token usage — input side. Used by callers for accounting; this
    /// crate does not interpret it.
    pub input_tokens: u32,
    /// Token usage — output side.
    pub output_tokens: u32,
    /// Energy reported by the backend, if it can measure or model the
    /// upstream call's joule cost. `Some(j)` when the value came from a
    /// real meter (`HwShunt` or `ModelBased`); `None` when the backend
    /// has no meter — the tier then falls back to
    /// [`LlmBackend::typical_joules_per_call`].
    pub energy_joules: Option<f64>,
}

/// Why generation stopped. Mirrors the union of common provider finish
/// reasons so backends can map without invention.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// Natural stop — stop sequence hit or model emitted EOS.
    Stop,
    /// Hit `max_tokens`. Output is truncated.
    Length,
    /// Backend-side content filter / safety policy refused.
    ContentFilter,
    /// Hard error reported by the backend with a free-form message.
    Error(String),
}

impl Default for FinishReason {
    fn default() -> Self {
        Self::Stop
    }
}

/// Errors a backend can report. Backends that already use `thiserror`
/// SHOULD surface their own enum and convert via `From`; this crate keeps
/// a thin enum so the open-standard surface doesn't dictate an internal
/// taxonomy.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// The backend could not reach the upstream provider.
    #[error("backend unavailable: {0}")]
    Unavailable(String),
    /// The request was malformed for this backend (token cap too large,
    /// unsupported field, etc.).
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    /// The upstream provider refused or errored.
    #[error("upstream error: {0}")]
    Upstream(String),
    /// Catch-all for backend-internal failures.
    #[error("backend error: {0}")]
    Other(String),
}

/// Pluggable LLM backend.
///
/// Consumer-supplied. The open-standard ships only [`EchoBackend`].
/// Backends MUST be `Send + Sync` so the tier (which is `Send`) can carry
/// a reference behind a generic parameter.
///
/// Backends SHOULD report `energy_joules` honestly: `Some(j)` for real
/// measurements, `None` when no meter exists. Lying here breaks the
/// JouleClaw provenance contract.
pub trait LlmBackend: Send + Sync {
    /// Human-readable model name surfaced in answers and receipts. SHOULD
    /// be stable across calls for the same configuration.
    fn model_name(&self) -> &str;

    /// Run one completion. Synchronous — async backends bridge inside
    /// their own runtime (e.g. `tokio::runtime::Handle::block_on`).
    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError>;

    /// Typical joule cost per call for this backend / model. Used by
    /// [`crate::LlmCheapTier::estimate_cost`] and as the fallback for
    /// `try_answer` when [`LlmResponse::energy_joules`] is `None`. SHOULD
    /// be the published catalog cost for the model; the donor's default
    /// (~2.001 J) is appropriate for the cheapest hosted LLMs as of 2026.
    fn typical_joules_per_call(&self) -> f64 {
        DEFAULT_TYPICAL_JOULES
    }
}

/// A deterministic, dependency-free backend that echoes the prompt back
/// with an `"echo: "` prefix. Honors `max_tokens` by truncating the
/// echoed text to that many characters (a token-shaped stand-in) and
/// surfaces `FinishReason::Length` when truncation occurred.
///
/// Useful for:
/// * unit tests in this and downstream crates,
/// * smoke tests that need the cascade walked end-to-end without a real
///   LLM,
/// * a minimum-viable reference implementation of [`LlmBackend`].
pub struct EchoBackend {
    name: String,
    typical_joules: f64,
}

impl EchoBackend {
    /// Construct an echo backend with the default name (`"echo"`) and
    /// the default typical-joules constant.
    pub fn new() -> Self {
        Self {
            name: "echo".into(),
            typical_joules: DEFAULT_TYPICAL_JOULES,
        }
    }

    /// Override the reported model name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Override the reported typical-joules-per-call.
    pub fn with_typical_joules(mut self, joules: f64) -> Self {
        self.typical_joules = joules;
        self
    }
}

impl Default for EchoBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmBackend for EchoBackend {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        // Deterministic transform: "echo: " + prompt, character-truncated
        // to the (token-shaped) max_tokens.
        let full = format!("echo: {}", request.prompt);
        let max = request.max_tokens.max(1) as usize;
        let (text, finish_reason) = if full.chars().count() > max {
            let truncated: String = full.chars().take(max).collect();
            (truncated, FinishReason::Length)
        } else {
            (full, FinishReason::Stop)
        };

        let input_tokens = request.prompt.chars().count() as u32;
        let output_tokens = text.chars().count() as u32;

        Ok(LlmResponse {
            text,
            finish_reason,
            input_tokens,
            output_tokens,
            // EchoBackend has no real meter — leave `None` so the tier
            // surfaces the Estimator-provenance fallback.
            energy_joules: None,
        })
    }

    fn typical_joules_per_call(&self) -> f64 {
        self.typical_joules
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn echo_backend_default_name() {
        let b = EchoBackend::new();
        assert_eq!(b.model_name(), "echo");
        assert!((b.typical_joules_per_call() - DEFAULT_TYPICAL_JOULES).abs() < f64::EPSILON);
    }

    #[test]
    fn echo_backend_with_name_overrides() {
        let b = EchoBackend::new().with_name("test-model");
        assert_eq!(b.model_name(), "test-model");
    }

    #[test]
    fn echo_backend_with_typical_joules_overrides() {
        let b = EchoBackend::new().with_typical_joules(0.5);
        assert!((b.typical_joules_per_call() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn echo_backend_stops_naturally_under_cap() {
        let b = EchoBackend::new();
        let req = LlmRequest::from_prompt("hello", 1024);
        let resp = b.complete(&req).expect("echo never fails");
        assert_eq!(resp.text, "echo: hello");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        assert_eq!(resp.input_tokens, 5);
        assert_eq!(resp.output_tokens, "echo: hello".chars().count() as u32);
        assert!(resp.energy_joules.is_none());
    }

    #[test]
    fn echo_backend_truncates_at_max_tokens() {
        let b = EchoBackend::new();
        let req = LlmRequest::from_prompt("abcdefghij", 4);
        let resp = b.complete(&req).expect("echo never fails");
        assert_eq!(resp.text.chars().count(), 4);
        assert_eq!(resp.finish_reason, FinishReason::Length);
    }

    #[test]
    fn llm_request_from_prompt_defaults() {
        let r = LlmRequest::from_prompt("hi", 64);
        assert_eq!(r.prompt, "hi");
        assert_eq!(r.max_tokens, 64);
        assert!((r.temperature - 0.3).abs() < f32::EPSILON);
        assert!(r.stop.is_empty());
        assert!(r.system.is_none());
        assert!(r.grammar.is_none());
    }

    #[test]
    fn finish_reason_default_is_stop() {
        assert_eq!(FinishReason::default(), FinishReason::Stop);
    }
}
