//! Voyage AI re-rank API client.
//!
//! Implements [`Reranker`] for:
//!
//! * `rerank-2`
//! * `rerank-2-lite`
//!
//! Endpoint: `POST https://api.voyageai.com/v1/rerank`.
//!
//! Voyage's response shape mirrors Cohere's: `data: [{ index, relevance_score }]`.

use std::time::Duration;

use async_trait::async_trait;
use eoc_core::JouleCost;
use serde::Deserialize;
use serde_json::json;
use tracing::warn;

use crate::error::{RerankError, RerankResult};
use crate::reranker::{Candidate, Reranker, ScoredCandidate};

/// Default endpoint for Voyage rerank v1.
pub const DEFAULT_ENDPOINT: &str = "https://api.voyageai.com/v1/rerank";

/// Voyage rerank client.
pub struct VoyageReranker {
    api_key: String,
    model: String,
    endpoint: String,
    http: reqwest::Client,
    max_retries: u32,
    microjoules_per_pair: u64,
}

impl VoyageReranker {
    /// Build a new client.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> RerankResult<Self> {
        let model = model.into();
        let microjoules_per_pair = match model.as_str() {
            "rerank-2" => 100,
            "rerank-2-lite" => 45,
            other => return Err(RerankError::ModelNotFound(other.to_string())),
        };
        Ok(Self {
            api_key: api_key.into(),
            model,
            endpoint: DEFAULT_ENDPOINT.to_string(),
            http: reqwest::Client::new(),
            max_retries: 3,
            microjoules_per_pair,
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

    /// Build the JSON request body.
    pub fn build_body(&self, query: &str, candidates: &[Candidate]) -> serde_json::Value {
        let docs: Vec<&str> = candidates.iter().map(|c| c.text.as_str()).collect();
        json!({
            "model": self.model,
            "query": query,
            "documents": docs,
            "top_k": docs.len(),
        })
    }

    /// Estimate energy for re-ranking `pairs` candidates.
    pub fn joule_estimate(&self, pairs: usize) -> JouleCost {
        JouleCost::estimated((pairs as u64).saturating_mul(self.microjoules_per_pair))
    }
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    data: Vec<ApiResult>,
}

#[derive(Debug, Deserialize)]
struct ApiResult {
    index: usize,
    relevance_score: f32,
}

#[async_trait]
impl Reranker for VoyageReranker {
    async fn rerank(
        &self,
        query: &str,
        candidates: &[Candidate],
    ) -> RerankResult<Vec<ScoredCandidate>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        if candidates.len() > self.max_pairs() {
            return Err(RerankError::BatchTooLarge(format!(
                "{} candidates exceeds Voyage v1 limit {}",
                candidates.len(),
                self.max_pairs()
            )));
        }
        let body = self.build_body(query, candidates);
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
                let mut out: Vec<ScoredCandidate> = parsed
                    .data
                    .into_iter()
                    .filter_map(|r| {
                        candidates.get(r.index).map(|c| ScoredCandidate {
                            candidate: c.clone(),
                            score: r.relevance_score,
                            rank: 0,
                        })
                    })
                    .collect();
                out.sort_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                });
                for (i, c) in out.iter_mut().enumerate() {
                    c.rank = i + 1;
                }
                return Ok(out);
            }
            if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(RerankError::InvalidApiKey);
            }
            if status.as_u16() == 429 {
                let retry_after = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok());
                if attempt >= self.max_retries {
                    return Err(RerankError::RateLimited {
                        retry_after_secs: retry_after,
                    });
                }
                let wait = retry_after.unwrap_or(1u64 << attempt.min(4));
                warn!(attempt, wait, "voyage rerank 429; backing off");
                tokio::time::sleep(Duration::from_secs(wait)).await;
                attempt += 1;
                continue;
            }
            let body = resp.text().await.unwrap_or_default();
            let truncated: String = body.chars().take(256).collect();
            return Err(RerankError::Unexpected {
                status: status.as_u16(),
                body: truncated,
            });
        }
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn max_pairs(&self) -> usize {
        // Voyage v1 caps at 1000.
        1000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models() {
        assert!(VoyageReranker::new("k", "rerank-2").is_ok());
        assert!(VoyageReranker::new("k", "rerank-2-lite").is_ok());
        assert!(matches!(
            VoyageReranker::new("k", "rerank-99"),
            Err(RerankError::ModelNotFound(_))
        ));
    }

    #[test]
    fn body_shape() {
        let v = VoyageReranker::new("k", "rerank-2").expect("ok");
        let body = v.build_body("q", &[Candidate::new("a", "alpha")]);
        assert_eq!(body["model"], "rerank-2");
        assert_eq!(body["documents"][0], "alpha");
        assert_eq!(body["top_k"], 1);
    }
}
