//! Cohere v2 Chat backend.
//!
//! Endpoint: `https://api.cohere.com/v2/chat`.
//!
//! Request shape:
//! ```json
//! {
//!   "model": "command-r-plus",
//!   "messages": [{"role": "user", "content": "..."}],
//!   "stream": false
//! }
//! ```
//!
//! Response shape uses `message.content[].text` instead of OpenAI's
//! `choices[].message.content`, and reports tokens via
//! `usage.tokens.{input_tokens,output_tokens}`.

use async_trait::async_trait;
use eoc_core::{JouleCost, JouleSource, Query, Response, Stage};
use eoc_neural::NeuralBackend;
use reqwest::header::{HeaderMap, CONTENT_TYPE};
use serde::Deserialize;
use tracing::{field, warn};

use crate::auth::Auth;
use crate::config::VendorConfig;
use crate::error::{VendorError, VendorResult};

/// Default Cohere chat endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://api.cohere.com/v2/chat";

/// Cohere v2 chat backend.
pub struct CohereBackend {
    client: reqwest::Client,
    auth: Auth,
    model: String,
    config: VendorConfig,
}

impl CohereBackend {
    /// Construct with API key + model (e.g. `command-r-plus`).
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            auth: Auth::Bearer(api_key.into()),
            model: model.into(),
            config: VendorConfig::new(),
        }
    }

    /// Override the [`VendorConfig`].
    pub fn with_config(mut self, config: VendorConfig) -> Self {
        self.config = config;
        self
    }

    fn endpoint(&self) -> &str {
        self.config
            .endpoint
            .as_deref()
            .unwrap_or(DEFAULT_ENDPOINT)
    }

    /// Build the Cohere request body (public for snapshot testing).
    pub fn build_request_body(&self, q: &Query) -> serde_json::Value {
        serde_json::json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": q.prompt,
            }],
            "stream": false,
        })
    }

    async fn try_infer(&self, q: &Query) -> VendorResult<Response> {
        tracing::debug!(
            target: "cohere.infer",
            model = %self.model,
            api_key = field::Empty,
            "dispatching cohere inference"
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
                    warn!(attempt, error = %e, "retryable cohere error; backing off");
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
        self.auth.apply(&mut headers)?;

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
        let payload: CohereResponse = resp.json().await?;
        let text = payload
            .message
            .map(|m| {
                m.content
                    .into_iter()
                    .map(|b| b.text)
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();
        let in_tok = payload
            .usage
            .as_ref()
            .and_then(|u| u.tokens.as_ref())
            .map(|t| t.input_tokens)
            .unwrap_or_else(|| estimate_tokens(&q.prompt));
        let out_tok = payload
            .usage
            .as_ref()
            .and_then(|u| u.tokens.as_ref())
            .map(|t| t.output_tokens)
            .unwrap_or_else(|| estimate_tokens(&text));
        Ok((text, in_tok, out_tok))
    }
}

#[async_trait]
impl NeuralBackend for CohereBackend {
    async fn infer(&self, q: &Query) -> Response {
        match self.try_infer(q).await {
            Ok(r) => r,
            Err(e) => Response::new(
                q.id,
                format!("[cohere-error: {e}]"),
                Stage::Neural,
                JouleCost { microjoules: 0, source: JouleSource::Estimated },
            ),
        }
    }
}

#[derive(Deserialize)]
struct CohereResponse {
    message: Option<CohereMessage>,
    usage: Option<CohereUsage>,
}

#[derive(Deserialize)]
struct CohereMessage {
    #[serde(default)]
    content: Vec<CohereTextBlock>,
}

#[derive(Deserialize)]
struct CohereTextBlock {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct CohereUsage {
    tokens: Option<CohereTokens>,
}

#[derive(Deserialize)]
struct CohereTokens {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
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
