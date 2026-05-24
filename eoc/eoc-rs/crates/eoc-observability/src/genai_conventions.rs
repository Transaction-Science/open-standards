//! OpenTelemetry semantic conventions for GenAI.
//!
//! These keys mirror the [GenAI semconv working draft]. We expose them as
//! `pub const &str` so callers get spell-check at compile time, plus a
//! [`GenAiAttributes`] helper that produces a ready-made attribute list.
//!
//! [GenAI semconv working draft]: https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai

use crate::span::AttrValue;

// ---- Request shape ----

/// Operation name, e.g. `"chat"`, `"text_completion"`, `"embedding"`.
pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";

/// System / provider, e.g. `"openai"`, `"anthropic"`, `"eoc.local"`.
pub const GEN_AI_SYSTEM: &str = "gen_ai.system";

/// Requested model name, e.g. `"gpt-4o"`, `"claude-opus-4-7"`.
pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";

/// Sampling temperature.
pub const GEN_AI_REQUEST_TEMPERATURE: &str = "gen_ai.request.temperature";

/// Top-p (nucleus) cutoff.
pub const GEN_AI_REQUEST_TOP_P: &str = "gen_ai.request.top_p";

/// Top-k cutoff.
pub const GEN_AI_REQUEST_TOP_K: &str = "gen_ai.request.top_k";

/// Maximum tokens requested.
pub const GEN_AI_REQUEST_MAX_TOKENS: &str = "gen_ai.request.max_tokens";

/// Stop sequences (concatenated string).
pub const GEN_AI_REQUEST_STOP_SEQUENCES: &str = "gen_ai.request.stop_sequences";

// ---- Response shape ----

/// Actual model used, e.g. `"gpt-4o-2024-08-06"`.
pub const GEN_AI_RESPONSE_MODEL: &str = "gen_ai.response.model";

/// Finish reason(s), comma-separated, e.g. `"stop"`, `"length"`,
/// `"tool_calls"`, `"content_filter"`.
pub const GEN_AI_RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";

/// Response id from the provider.
pub const GEN_AI_RESPONSE_ID: &str = "gen_ai.response.id";

// ---- Token usage ----

/// Input/prompt tokens.
pub const GEN_AI_USAGE_PROMPT_TOKENS: &str = "gen_ai.usage.prompt_tokens";

/// Output/completion tokens.
pub const GEN_AI_USAGE_COMPLETION_TOKENS: &str = "gen_ai.usage.completion_tokens";

/// Total tokens (prompt + completion).
pub const GEN_AI_USAGE_TOTAL_TOKENS: &str = "gen_ai.usage.total_tokens";

// ---- EOC extensions ----

/// Cascade stage that produced the response (`"cache"`, `"kv"`, `"graph"`,
/// `"neural"`).
pub const EOC_STAGE: &str = "eoc.stage";

/// Joule cost attributed to this call (microjoules).
pub const EOC_JOULES_MICRO: &str = "eoc.joules.microjoules";

/// Joule source (`"measured"` | `"estimated"`).
pub const EOC_JOULES_SOURCE: &str = "eoc.joules.source";

/// gCO2e attributed to this call.
pub const EOC_GCO2E: &str = "eoc.co2e_grams";

/// USD attributed to this call.
pub const EOC_COST_USD: &str = "eoc.cost.usd";

/// Helper for building the canonical attribute set for a GenAI span.
///
/// All fields are optional; missing fields are omitted from the emitted
/// attribute list.
#[derive(Debug, Default, Clone)]
pub struct GenAiAttributes {
    /// `gen_ai.operation.name`.
    pub operation_name: Option<String>,
    /// `gen_ai.system`.
    pub system: Option<String>,
    /// `gen_ai.request.model`.
    pub request_model: Option<String>,
    /// `gen_ai.request.temperature`.
    pub temperature: Option<f64>,
    /// `gen_ai.request.top_p`.
    pub top_p: Option<f64>,
    /// `gen_ai.request.top_k`.
    pub top_k: Option<i64>,
    /// `gen_ai.request.max_tokens`.
    pub max_tokens: Option<i64>,
    /// `gen_ai.response.model`.
    pub response_model: Option<String>,
    /// `gen_ai.response.finish_reasons`.
    pub finish_reasons: Option<String>,
    /// `gen_ai.response.id`.
    pub response_id: Option<String>,
    /// `gen_ai.usage.prompt_tokens`.
    pub prompt_tokens: Option<i64>,
    /// `gen_ai.usage.completion_tokens`.
    pub completion_tokens: Option<i64>,
    /// `gen_ai.usage.total_tokens`.
    pub total_tokens: Option<i64>,
}

impl GenAiAttributes {
    /// Render this struct into the canonical attribute list. Order is
    /// deterministic so snapshot tests stay stable.
    pub fn to_attributes(&self) -> Vec<(String, AttrValue)> {
        let mut out: Vec<(String, AttrValue)> = Vec::new();

        if let Some(s) = &self.operation_name {
            out.push((GEN_AI_OPERATION_NAME.to_string(), AttrValue::String(s.clone())));
        }
        if let Some(s) = &self.system {
            out.push((GEN_AI_SYSTEM.to_string(), AttrValue::String(s.clone())));
        }
        if let Some(s) = &self.request_model {
            out.push((GEN_AI_REQUEST_MODEL.to_string(), AttrValue::String(s.clone())));
        }
        if let Some(v) = self.temperature {
            out.push((GEN_AI_REQUEST_TEMPERATURE.to_string(), AttrValue::Float(v)));
        }
        if let Some(v) = self.top_p {
            out.push((GEN_AI_REQUEST_TOP_P.to_string(), AttrValue::Float(v)));
        }
        if let Some(v) = self.top_k {
            out.push((GEN_AI_REQUEST_TOP_K.to_string(), AttrValue::Int(v)));
        }
        if let Some(v) = self.max_tokens {
            out.push((GEN_AI_REQUEST_MAX_TOKENS.to_string(), AttrValue::Int(v)));
        }
        if let Some(s) = &self.response_model {
            out.push((
                GEN_AI_RESPONSE_MODEL.to_string(),
                AttrValue::String(s.clone()),
            ));
        }
        if let Some(s) = &self.finish_reasons {
            out.push((
                GEN_AI_RESPONSE_FINISH_REASONS.to_string(),
                AttrValue::String(s.clone()),
            ));
        }
        if let Some(s) = &self.response_id {
            out.push((GEN_AI_RESPONSE_ID.to_string(), AttrValue::String(s.clone())));
        }
        if let Some(v) = self.prompt_tokens {
            out.push((GEN_AI_USAGE_PROMPT_TOKENS.to_string(), AttrValue::Int(v)));
        }
        if let Some(v) = self.completion_tokens {
            out.push((GEN_AI_USAGE_COMPLETION_TOKENS.to_string(), AttrValue::Int(v)));
        }
        if let Some(v) = self.total_tokens {
            out.push((GEN_AI_USAGE_TOTAL_TOKENS.to_string(), AttrValue::Int(v)));
        }

        out
    }
}
