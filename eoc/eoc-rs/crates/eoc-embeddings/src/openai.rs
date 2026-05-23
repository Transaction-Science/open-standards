//! OpenAI embeddings backend (v1).
//!
//! Implements [`Embedder`] for:
//!
//! * `text-embedding-3-small` (1536d, Matryoshka-truncatable)
//! * `text-embedding-3-large` (3072d, Matryoshka-truncatable)
//! * `text-embedding-ada-002` (1536d, legacy)
//!
//! Endpoint: `POST https://api.openai.com/v1/embeddings`.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::warn;

use eoc_core::JouleCost;

use crate::embedder::Embedder;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::joule_estimator::JouleEstimator;

/// Default endpoint for OpenAI v1.
pub const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/embeddings";

/// OpenAI embeddings client.
pub struct OpenAiEmbedder {
    api_key: String,
    model: String,
    dimensions: usize,
    endpoint: String,
    http: reqwest::Client,
    estimator: JouleEstimator,
    max_retries: u32,
}

impl OpenAiEmbedder {
    /// Build an `OpenAiEmbedder` for `model` with the given API key.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> EmbeddingResult<Self> {
        let model = model.into();
        let dimensions = match model.as_str() {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
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

#[derive(Debug, Serialize)]
struct _BodyShape<'a> {
    model: &'a str,
    input: &'a [&'a str],
    encoding_format: &'static str,
}

#[async_trait]
impl Embedder for OpenAiEmbedder {
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
                warn!(attempt, wait, "openai 429; backing off");
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
    fn dimensions_per_model() {
        let e = OpenAiEmbedder::new("sk-x", "text-embedding-3-small").expect("ok");
        assert_eq!(e.dimensions(), 1536);
        let e = OpenAiEmbedder::new("sk-x", "text-embedding-3-large").expect("ok");
        assert_eq!(e.dimensions(), 3072);
        let e = OpenAiEmbedder::new("sk-x", "text-embedding-ada-002").expect("ok");
        assert_eq!(e.dimensions(), 1536);
    }

    #[test]
    fn unknown_model_errors() {
        let r = OpenAiEmbedder::new("sk-x", "nope");
        assert!(matches!(r, Err(EmbeddingError::ModelNotFound(_))));
    }

    #[test]
    fn request_body_shape() {
        let e = OpenAiEmbedder::new("sk-x", "text-embedding-3-small").expect("ok");
        let body = e.build_body(&["hello", "world"]);
        assert_eq!(body["model"], "text-embedding-3-small");
        assert_eq!(body["input"][0], "hello");
        assert_eq!(body["input"][1], "world");
        assert_eq!(body["encoding_format"], "float");
    }
}
