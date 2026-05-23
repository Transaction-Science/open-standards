//! Mistral embeddings backend.
//!
//! Implements [`Embedder`] for `mistral-embed` (1024d).
//!
//! Endpoint: `POST https://api.mistral.ai/v1/embeddings`. The response
//! shape is OpenAI-compatible.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tracing::warn;

use eoc_core::JouleCost;

use crate::embedder::Embedder;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::joule_estimator::JouleEstimator;

/// Default endpoint for Mistral v1.
pub const DEFAULT_ENDPOINT: &str = "https://api.mistral.ai/v1/embeddings";

/// Mistral embeddings client.
pub struct MistralEmbedder {
    api_key: String,
    model: String,
    dimensions: usize,
    endpoint: String,
    http: reqwest::Client,
    estimator: JouleEstimator,
    max_retries: u32,
}

impl MistralEmbedder {
    /// Build a `MistralEmbedder` for `model`.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> EmbeddingResult<Self> {
        let model = model.into();
        let dimensions = match model.as_str() {
            "mistral-embed" => 1024,
            other => return Err(EmbeddingError::ModelNotFound(other.to_string())),
        };
        Ok(Self {
            api_key: api_key.into(),
            model,
            dimensions,
            endpoint: DEFAULT_ENDPOINT.to_string(),
            http: reqwest::Client::new(),
            estimator: JouleEstimator::default(),
            max_retries: 3,
        })
    }

    /// Override the endpoint (used for tests).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Override the retry budget.
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Build the JSON request body for snapshot tests.
    pub fn build_body(&self, texts: &[&str]) -> serde_json::Value {
        json!({
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        })
    }
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    data: Vec<ApiDatum>,
}

#[derive(Debug, Deserialize)]
struct ApiDatum {
    embedding: Vec<f32>,
}

#[async_trait]
impl Embedder for MistralEmbedder {
    async fn embed(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
        let body = self.build_body(texts);
        let mut attempt = 0u32;
        loop {
            let resp = self
                .http
                .post(&self.endpoint)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await?;
            let status = resp.status();
            if status.is_success() {
                let parsed: ApiResponse = resp.json().await?;
                return Ok(parsed.data.into_iter().map(|d| d.embedding).collect());
            }
            if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(EmbeddingError::InvalidApiKey);
            }
            if status.as_u16() == 429 {
                let retry_after = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok());
                if attempt >= self.max_retries {
                    return Err(EmbeddingError::RateLimited {
                        retry_after_secs: retry_after,
                    });
                }
                let wait = retry_after.unwrap_or(1u64 << attempt.min(4));
                warn!(attempt, wait, "mistral 429; backing off");
                tokio::time::sleep(Duration::from_secs(wait)).await;
                attempt += 1;
                continue;
            }
            let body = resp.text().await.unwrap_or_default();
            let truncated: String = body.chars().take(256).collect();
            return Err(EmbeddingError::Unexpected {
                status: status.as_u16(),
                body: truncated,
            });
        }
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn joule_estimate(&self, text_len_chars: usize) -> JouleCost {
        self.estimator.estimate(&self.model, text_len_chars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_known() {
        let e = MistralEmbedder::new("k", "mistral-embed").expect("ok");
        assert_eq!(e.dimensions(), 1024);
        assert_eq!(e.model_name(), "mistral-embed");
    }

    #[test]
    fn body_shape() {
        let e = MistralEmbedder::new("k", "mistral-embed").expect("ok");
        let body = e.build_body(&["hi"]);
        assert_eq!(body["model"], "mistral-embed");
        assert_eq!(body["input"][0], "hi");
    }
}
