//! Shared helper for OpenAI-compatible Chat-Completions endpoints.
//!
//! OpenAI, Mistral, Groq, Together, and Fireworks all accept the same
//! request shape (`{ model, messages: [{role, content}], stream }`) and
//! return the same response shape. This module owns that shared logic;
//! per-vendor modules only provide the endpoint URL and the
//! [`Auth`](crate::Auth) variant.

use std::time::Duration;

use eoc_core::{Query, Response, Stage};
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::auth::Auth;
use crate::config::{RetryPolicy, VendorConfig};
use crate::error::{VendorError, VendorResult};

/// One chat message in the OpenAI schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `system` | `user` | `assistant`.
    pub role: String,
    /// Free-form text content.
    pub content: String,
}

/// OpenAI-compatible request body.
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    /// Vendor-qualified model name.
    pub model: String,
    /// Ordered messages.
    pub messages: Vec<ChatMessage>,
    /// Stream incremental tokens (SSE).
    pub stream: bool,
    /// Optional max-tokens cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

/// OpenAI-compatible non-streaming response body.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    /// Per-choice completions; we use the first.
    pub choices: Vec<ChatChoice>,
    /// Token usage (always present for OpenAI / clones).
    pub usage: Option<ChatUsage>,
}

/// One completion alternative.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatChoice {
    /// The completed assistant message.
    pub message: Option<ChatMessage>,
    /// Reason the generation stopped (`stop`, `length`,
    /// `content_filter`, etc).
    pub finish_reason: Option<String>,
}

/// Token usage counters.
#[derive(Debug, Clone, Copy, Deserialize, Default)]
pub struct ChatUsage {
    /// Prompt-side tokens.
    pub prompt_tokens: u32,
    /// Completion-side tokens.
    pub completion_tokens: u32,
}

/// Streaming SSE delta frame.
#[derive(Debug, Clone, Deserialize)]
struct ChatStreamFrame {
    choices: Vec<ChatStreamChoice>,
    usage: Option<ChatUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatStreamChoice {
    delta: ChatStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatStreamDelta {
    content: Option<String>,
}

/// Build the `ChatRequest` for a [`Query`]. Wraps the prompt in a single
/// `user` message; system prompts are not part of the [`Query`] surface
/// yet (they belong on the EOC side, not the vendor side).
pub fn build_request(model: &str, q: &Query, stream: bool) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: q.prompt.clone(),
        }],
        stream,
        max_tokens: None,
    }
}

/// Estimate prompt tokens via a 4-chars-per-token heuristic. Used only
/// for joule estimation when the vendor does not return a `usage` block.
fn estimate_tokens(s: &str) -> u32 {
    ((s.chars().count() as f64) / 4.0).ceil().max(1.0) as u32
}

/// Execute an OpenAI-compatible chat-completions call.
///
/// `endpoint` is the absolute URL (e.g. `https://api.openai.com/v1/chat/completions`).
/// `auth` provides the bearer credentials. `stream` selects between
/// SSE and one-shot.
#[allow(clippy::too_many_arguments)]
pub async fn execute(
    client: &reqwest::Client,
    endpoint: &str,
    auth: &Auth,
    model: &str,
    q: &Query,
    stream: bool,
    config: &VendorConfig,
) -> VendorResult<Response> {
    let body = build_request(model, q, stream);
    let mut attempt: u32 = 0;
    let policy = config.retry_policy;

    loop {
        let outcome = single_call(client, endpoint, auth, &body, stream, config.timeout).await;
        match outcome {
            Ok((text, usage)) => {
                let in_tokens = usage
                    .map(|u| u.prompt_tokens)
                    .unwrap_or_else(|| estimate_tokens(&q.prompt));
                let out_tokens = usage
                    .map(|u| u.completion_tokens)
                    .unwrap_or_else(|| estimate_tokens(&text));
                let cost = config.joule_estimator.estimate(in_tokens, out_tokens, model);
                return Ok(Response::new(q.id, text, Stage::Neural, cost));
            }
            Err(e) if is_retryable(&e) && attempt < policy.max_retries => {
                warn!(attempt, error = %e, "retryable vendor error; backing off");
                tokio::time::sleep(policy.backoff_for(attempt)).await;
                attempt += 1;
                continue;
            }
            Err(e) => return Err(e),
        }
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

/// Single HTTP call — returns `(decoded_text, optional_usage)`.
async fn single_call(
    client: &reqwest::Client,
    endpoint: &str,
    auth: &Auth,
    body: &ChatRequest,
    stream: bool,
    timeout: Duration,
) -> VendorResult<(String, Option<ChatUsage>)> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "application/json".parse().expect("static"));
    auth.apply(&mut headers)?;

    let resp = client
        .post(endpoint)
        .timeout(timeout)
        .headers(headers)
        .json(body)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        return Err(classify_error(status, resp).await);
    }

    if stream {
        decode_stream(resp).await
    } else {
        let parsed: ChatResponse = resp.json().await?;
        let text = parsed
            .choices
            .first()
            .and_then(|c| c.message.as_ref())
            .map(|m| m.content.clone())
            .unwrap_or_default();
        Ok((text, parsed.usage))
    }
}

async fn decode_stream(resp: reqwest::Response) -> VendorResult<(String, Option<ChatUsage>)> {
    let mut text = String::new();
    let mut usage: Option<ChatUsage> = None;
    let mut buf = String::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(VendorError::from)?;
        buf.push_str(&String::from_utf8_lossy(&bytes));
        // Drain complete SSE events (terminated by blank line).
        while let Some(idx) = buf.find("\n\n") {
            let event = buf[..idx].to_string();
            buf.drain(..idx + 2);
            for line in event.lines() {
                let line = line.trim_start();
                let Some(payload) = line.strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.trim();
                if payload == "[DONE]" || payload.is_empty() {
                    continue;
                }
                match serde_json::from_str::<ChatStreamFrame>(payload) {
                    Ok(frame) => {
                        if let Some(u) = frame.usage {
                            usage = Some(u);
                        }
                        for c in frame.choices {
                            if let Some(delta) = c.delta.content {
                                text.push_str(&delta);
                            }
                            if c.finish_reason.as_deref() == Some("content_filter") {
                                return Err(VendorError::ContentFiltered(
                                    "content_filter finish_reason".to_string(),
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        debug!(?e, "skipping malformed SSE frame");
                    }
                }
            }
        }
    }
    Ok((text, usage))
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
        429 => VendorError::RateLimited {
            retry_after_secs: retry_after,
        },
        401 | 403 => VendorError::InvalidApiKey,
        404 => VendorError::ModelNotFound(body),
        s => VendorError::Unexpected { status: s, body },
    }
}

#[allow(dead_code)]
pub(crate) const DEFAULT_RETRY: RetryPolicy = RetryPolicy {
    max_retries: 3,
    initial_backoff: Duration::from_millis(250),
    backoff_factor: 2.0,
};
