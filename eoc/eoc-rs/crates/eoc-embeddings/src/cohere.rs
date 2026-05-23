//! Cohere embeddings backend (v2).
//!
//! Implements [`Embedder`] for:
//!
//! * `embed-english-v3.0` (1024d)
//! * `embed-multilingual-v3.0` (1024d)
//! * `embed-english-light-v3.0` (384d)
//!
//! Endpoint: `POST https://api.cohere.com/v2/embed`. Cohere requires an
//! `input_type` field on every request; we default to `search_document`,
//! which is the right choice for indexing into the KV stage. Override via
//! [`CohereEmbedder::with_input_type`] when embedding queries.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tracing::warn;

use eoc_core::JouleCost;

use crate::embedder::Embedder;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::joule_estimator::JouleEstimator;

/// Default endpoint for Cohere v2.
pub const DEFAULT_ENDPOINT: &str = "https://api.cohere.com/v2/embed";

/// Cohere embeddings client.
pub struct CohereEmbedder {
    api_key: String,
    model: String,
    dimensions: usize,
    endpoint: String,
    input_type: String,
    http: reqwest::Client,
    estimator: JouleEstimator,
    max_retries: u32,
}

impl CohereEmbedder {
    /// Build a `CohereEmbedder` for `model`.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> EmbeddingResult<Self> {
        let model = model.into();
        let dimensions = match model.as_str() {
            "embed-english-v3.0" => 1024,
            "embed-multilingual-v3.0" => 1024,
            "embed-english-light-v3.0" => 384,
            other => return Err(EmbeddingError::ModelNotFound(other.to_string())),
        };
        Ok(Self {
            api_key: api_key.into(),
            model,
            dimensions,
            endpoint: DEFAULT_ENDPOINT.to_string(),
            input_type: "search_document".to_string(),
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

    /// Override the Cohere `input_type` field (`search_document`,
    /// `search_query`, `classification`, `clustering`).
    pub fn with_input_type(mut self, t: impl Into<String>) -> Self {
        self.input_type = t.into();
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
            "texts": texts,
            "input_type": self.input_type,
            "embedding_types": ["float"],
        })
    }
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    embeddings: ApiEmbeddings,
}

#[derive(Debug, Deserialize)]
struct ApiEmbeddings {
    float: Vec<Vec<f32>>,
}

#[async_trait]
impl Embedder for CohereEmbedder {
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
                return Ok(parsed.embeddings.float);
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
                warn!(attempt, wait, "cohere 429; backing off");
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
        assert_eq!(
            CohereEmbedder::new("k", "embed-english-v3.0").expect("ok").dimensions(),
            1024
        );
        assert_eq!(
            CohereEmbedder::new("k", "embed-multilingual-v3.0").expect("ok").dimensions(),
            1024
        );
        assert_eq!(
            CohereEmbedder::new("k", "embed-english-light-v3.0").expect("ok").dimensions(),
            384
        );
    }

    #[test]
    fn body_includes_required_fields() {
        let e = CohereEmbedder::new("k", "embed-english-v3.0").expect("ok");
        let body = e.build_body(&["hi"]);
        assert_eq!(body["model"], "embed-english-v3.0");
        assert_eq!(body["texts"][0], "hi");
        assert_eq!(body["input_type"], "search_document");
        assert_eq!(body["embedding_types"][0], "float");
    }
}
