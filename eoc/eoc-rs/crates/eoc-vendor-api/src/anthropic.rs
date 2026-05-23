//! Anthropic Messages API backend.
//!
//! Endpoint: `https://api.anthropic.com/v1/messages`.
//!
//! Differences from the OpenAI-compatible schema:
//!
//! * Auth uses the `x-api-key` header.
//! * The request requires an `anthropic-version` header.
//! * `max_tokens` is required, not optional.
//! * Streaming uses Anthropic-specific SSE events
//!   (`message_start`, `content_block_delta`, `message_delta`,
//!   `message_stop`).
//! * Prompt caching is opted-in per content block via
//!   `cache_control: { "type": "ephemeral" }`. When the EOC layer tags a
//!   system prompt as cacheable, set [`AnthropicBackend::with_system_cached`].

use std::time::Duration;

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource, Query, Response, Stage};
use eoc_neural::NeuralBackend;
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use serde::Deserialize;
use tracing::{debug, field, warn};

use crate::auth::Auth;
use crate::config::VendorConfig;
use crate::error::{VendorError, VendorResult};

/// Default Anthropic endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
/// Default Anthropic API version header value.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Messages backend.
pub struct AnthropicBackend {
    client: reqwest::Client,
    api_key: String,
    model: String,
    max_tokens: u32,
    stream: bool,
    system: Option<String>,
    system_cached: bool,
    config: VendorConfig,
}

impl AnthropicBackend {
    /// Construct with API key + model (e.g. `claude-3-5-sonnet-20241022`).
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            max_tokens: 1024,
            stream: true,
            system: None,
            system_cached: false,
            config: VendorConfig::new(),
        }
    }

    /// Override the `max_tokens` cap (Anthropic requires this field).
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    /// Attach a system prompt (consumes `self`).
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Mark the system prompt as cacheable (sends
    /// `cache_control: { type: "ephemeral" }`). Caller is responsible
    /// for only setting this when EOC's caching layer has tagged the
    /// system prompt as appropriate.
    pub fn with_system_cached(mut self, cached: bool) -> Self {
        self.system_cached = cached;
        self
    }

    /// Override the [`VendorConfig`].
    pub fn with_config(mut self, config: VendorConfig) -> Self {
        self.config = config;
        self
    }

    /// Disable SSE streaming.
    pub fn without_stream(mut self) -> Self {
        self.stream = false;
        self
    }

    fn endpoint(&self) -> &str {
        self.config
            .endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
    }

    /// Build the Anthropic request body (public for snapshot testing).
    pub fn build_request_body(&self, q: &Query) -> serde_json::Value {
        let system_field = self.system.as_deref().map(|s| {
            if self.system_cached {
                serde_json::json!([{
                    "type": "text",
                    "text": s,
                    "cache_control": {"type": "ephemeral"}
                }])
            } else {
                serde_json::Value::String(s.to_string())
            }
        });

        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "messages": [{
                "role": "user",
                "content": q.prompt,
            }],
        });

        if let Some(sys) = system_field
            && let Some(map) = body.as_object_mut()
        {
            map.insert("system".to_string(), sys);
        }
        body
    }

    async fn try_infer(&self, q: &Query) -> VendorResult<Response> {
        tracing::debug!(
            target: "anthropic.infer",
            model = %self.model,
            api_key = field::Empty,
            "dispatching anthropic inference"
        );

        let policy = self.config.retry_policy;
        let mut attempt: u32 = 0;
        loop {
            let outcome = self.single_call(q).await;
            match outcome {
                Ok((text, in_tok, out_tok)) => {
                    let cost = self.config.joule_estimator.estimate(in_tok, out_tok, &self.model);
                    return Ok(Response::new(q.id, text, Stage::Neural, cost));
                }
                Err(e) if is_retryable(&e) && attempt < policy.max_retries => {
                    warn!(attempt, error = %e, "retryable anthropic error; backing off");
                    tokio::time::sleep(policy.backoff_for(attempt)).await;
                    attempt += 1;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn single_call(&self, q: &Query) -> VendorResult<(String, u32, u32)> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, "application/json".parse().expect("static"));
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        Auth::ApiKey(self.api_key.clone()).apply(&mut headers)?;

        let body = self.build_request_body(q);
        let resp = self
            .client
            .post(self.endpoint())
            .timeout(self.config.timeout)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            return Err(classify_error(status, resp).await);
        }

        if self.stream {
            decode_stream(resp, &q.prompt).await
        } else {
            let payload: AnthropicResponse = resp.json().await?;
            let text = payload.content.into_iter().map(|b| b.text).collect::<Vec<_>>().join("");
            let in_tok = payload.usage.input_tokens;
            let out_tok = payload.usage.output_tokens;
            Ok((text, in_tok, out_tok))
        }
    }
}

#[async_trait]
impl NeuralBackend for AnthropicBackend {
    async fn infer(&self, q: &Query) -> Response {
        match self.try_infer(q).await {
            Ok(r) => r,
            Err(e) => Response::new(
                q.id,
                format!("[anthropic-error: {e}]"),
                Stage::Neural,
                JouleCost { microjoules: 0, source: JouleSource::Estimated },
            ),
        }
    }
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicTextBlock>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct AnthropicTextBlock {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize, Default, Clone, Copy)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: StreamMessageStart },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { delta: StreamDelta },
    #[serde(rename = "message_delta")]
    MessageDelta { usage: Option<StreamDeltaUsage> },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct StreamMessageStart {
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum StreamDelta {
    #[serde(rename = "text_delta")]
    Text {
        #[serde(default)]
        text: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize, Default)]
struct StreamDeltaUsage {
    #[serde(default)]
    output_tokens: u32,
}

async fn decode_stream(resp: reqwest::Response, prompt: &str) -> VendorResult<(String, u32, u32)> {
    let mut text = String::new();
    let mut input_tokens: u32 = 0;
    let mut output_tokens: u32 = 0;
    let mut buf = String::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(VendorError::from)?;
        buf.push_str(&String::from_utf8_lossy(&bytes));
        while let Some(idx) = buf.find("\n\n") {
            let event = buf[..idx].to_string();
            buf.drain(..idx + 2);
            for line in event.lines() {
                let line = line.trim_start();
                let Some(payload) = line.strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.trim();
                if payload.is_empty() {
                    continue;
                }
                match serde_json::from_str::<StreamEvent>(payload) {
                    Ok(StreamEvent::MessageStart { message }) => {
                        input_tokens = message.usage.input_tokens
                            + message.usage.cache_read_input_tokens
                            + message.usage.cache_creation_input_tokens;
                    }
                    Ok(StreamEvent::ContentBlockDelta {
                        delta: StreamDelta::Text { text: t },
                    }) => text.push_str(&t),
                    Ok(StreamEvent::MessageDelta { usage }) => {
                        if let Some(u) = usage {
                            output_tokens = u.output_tokens;
                        }
                    }
                    Ok(StreamEvent::MessageStop)
                    | Ok(StreamEvent::ContentBlockDelta { delta: StreamDelta::Other })
                    | Ok(StreamEvent::Other) => {}
                    Err(e) => debug!(?e, "skipping malformed anthropic SSE frame"),
                }
            }
        }
    }
    if input_tokens == 0 {
        input_tokens = ((prompt.chars().count() as f64) / 4.0).ceil().max(1.0) as u32;
    }
    Ok((text, input_tokens, output_tokens))
}

async fn classify_error(status: reqwest::StatusCode, resp: reqwest::Response) -> VendorError {
    let retry_after = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    let body = resp
        .text()
        .await
        .unwrap_or_default()
        .chars()
        .take(512)
        .collect::<String>();
    match status.as_u16() {
        429 => VendorError::RateLimited { retry_after_secs: retry_after },
        401 | 403 => VendorError::InvalidApiKey,
        404 => VendorError::ModelNotFound(body),
        s => VendorError::Unexpected { status: s, body },
    }
}

fn is_retryable(e: &VendorError) -> bool {
    match e {
        VendorError::RateLimited { .. }
        | VendorError::Timeout
        | VendorError::NetworkError(_) => true,
        VendorError::Unexpected { status, .. } => (500..=599).contains(status),
        _ => false,
    }
}

#[allow(dead_code)]
const _BACKOFF_HINT: Duration = Duration::from_millis(250);
