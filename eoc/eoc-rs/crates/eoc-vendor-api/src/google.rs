//! Google Gemini backend.
//!
//! Endpoint:
//! `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
//! (or `:streamGenerateContent?alt=sse` for the SSE variant).
//!
//! Auth is via `?key=<api-key>` query parameter, not a header.

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource, Query, Response, Stage};
use eoc_neural::NeuralBackend;
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, CONTENT_TYPE};
use serde::Deserialize;
use tracing::{debug, field, warn};

use crate::auth::Auth;
use crate::config::VendorConfig;
use crate::error::{VendorError, VendorResult};

/// Default Google API base. Per-call path includes the model name.
pub const DEFAULT_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Gemini backend.
pub struct GoogleBackend {
    client: reqwest::Client,
    auth: Auth,
    model: String,
    stream: bool,
    config: VendorConfig,
}

impl GoogleBackend {
    /// Construct with API key + model (e.g. `gemini-1.5-pro`).
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            auth: Auth::GoogleApiKey(api_key.into()),
            model: model.into(),
            stream: true,
            config: VendorConfig::new(),
        }
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

    fn endpoint_for(&self) -> String {
        if let Some(custom) = self.config.endpoint.as_deref() {
            // For tests, the custom endpoint stands in for the full URL.
            let key = self.auth.google_key().unwrap_or("");
            if custom.contains('?') {
                format!("{custom}&key={key}")
            } else {
                format!("{custom}?key={key}")
            }
        } else {
            let verb = if self.stream {
                "streamGenerateContent?alt=sse"
            } else {
                "generateContent"
            };
            let key = self.auth.google_key().unwrap_or("");
            let sep = if verb.contains('?') { "&" } else { "?" };
            format!("{}/{}:{}{}key={}", DEFAULT_BASE, self.model, verb, sep, key)
        }
    }

    /// Build the Google request body (public for snapshot testing).
    pub fn build_request_body(&self, q: &Query) -> serde_json::Value {
        serde_json::json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": q.prompt}]
            }]
        })
    }

    async fn try_infer(&self, q: &Query) -> VendorResult<Response> {
        tracing::debug!(
            target: "google.infer",
            model = %self.model,
            api_key = field::Empty,
            "dispatching gemini inference"
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
                    warn!(attempt, error = %e, "retryable google error; backing off");
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

        let body = self.build_request_body(q);
        let resp = self
            .client
            .post(self.endpoint_for())
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
            let payload: GeminiResponse = resp.json().await?;
            let text = payload
                .candidates
                .into_iter()
                .flat_map(|c| c.content.parts)
                .map(|p| p.text)
                .collect::<Vec<_>>()
                .join("");
            let in_tok = payload
                .usage_metadata
                .as_ref()
                .map(|u| u.prompt_token_count)
                .unwrap_or_else(|| estimate_tokens(&q.prompt));
            let out_tok = payload
                .usage_metadata
                .as_ref()
                .map(|u| u.candidates_token_count)
                .unwrap_or_else(|| estimate_tokens(&text));
            Ok((text, in_tok, out_tok))
        }
    }
}

#[async_trait]
impl NeuralBackend for GoogleBackend {
    async fn infer(&self, q: &Query) -> Response {
        match self.try_infer(q).await {
            Ok(r) => r,
            Err(e) => Response::new(
                q.id,
                format!("[google-error: {e}]"),
                Stage::Neural,
                JouleCost { microjoules: 0, source: JouleSource::Estimated },
            ),
        }
    }
}

#[derive(Deserialize)]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Deserialize)]
struct GeminiContent {
    #[serde(default)]
    parts: Vec<GeminiPart>,
}

#[derive(Deserialize)]
struct GeminiPart {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize, Default, Clone, Copy)]
struct GeminiUsage {
    #[serde(rename = "promptTokenCount", default)]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount", default)]
    candidates_token_count: u32,
}

async fn decode_stream(resp: reqwest::Response, prompt: &str) -> VendorResult<(String, u32, u32)> {
    let mut text = String::new();
    let mut in_tok: u32 = 0;
    let mut out_tok: u32 = 0;
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
                match serde_json::from_str::<GeminiResponse>(payload) {
                    Ok(frame) => {
                        for c in frame.candidates {
                            for p in c.content.parts {
                                text.push_str(&p.text);
                            }
                        }
                        if let Some(u) = frame.usage_metadata {
                            if u.prompt_token_count > 0 {
                                in_tok = u.prompt_token_count;
                            }
                            if u.candidates_token_count > 0 {
                                out_tok = u.candidates_token_count;
                            }
                        }
                    }
                    Err(e) => debug!(?e, "skipping malformed gemini SSE frame"),
                }
            }
        }
    }
    if in_tok == 0 {
        in_tok = estimate_tokens(prompt);
    }
    if out_tok == 0 {
        out_tok = estimate_tokens(&text);
    }
    Ok((text, in_tok, out_tok))
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

fn estimate_tokens(s: &str) -> u32 {
    ((s.chars().count() as f64) / 4.0).ceil().max(1.0) as u32
}
