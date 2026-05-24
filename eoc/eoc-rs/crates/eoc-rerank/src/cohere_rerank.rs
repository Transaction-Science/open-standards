//! Cohere re-rank API client.
//!
//! Implements [`Reranker`] for:
//!
//! * `rerank-english-v3.0`
//! * `rerank-multilingual-v3.0`
//! * `rerank-v3.5`
//!
//! Endpoint: `POST https://api.cohere.com/v2/rerank`.
//!
//! Cohere returns `{ results: [{ index, relevance_score }, ...] }` —
//! `index` refers back into the request's `documents` array.

use std::time::Duration;

use async_trait::async_trait;
use eoc_core::JouleCost;
use serde::Deserialize;
use serde_json::json;
use tracing::warn;

use crate::error::{RerankError, RerankResult};
use crate::reranker::{Candidate, Reranker, ScoredCandidate};

/// Default endpoint for Cohere rerank v2.
pub const DEFAULT_ENDPOINT: &str = "https://api.cohere.com/v2/rerank";

/// Cohere rerank client.
pub struct CohereReranker {
    api_key: String,
    model: String,
    endpoint: String,
    http: reqwest::Client,
    max_retries: u32,
    /// Micro-joules per `(query, document)` pair, drawn from the HF
    /// Energy Score reranker numbers.
    microjoules_per_pair: u64,
}

impl CohereReranker {
    /// Build a new client.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> RerankResult<Self> {
        let model = model.into();
        // Per-pair joule estimates roughly track the model class. The
        // English/multilingual v3 cross-encoders cost ~70 µJ/pair on the
        // HF Energy Score reference rig; v3.5 (larger) ~120 µJ/pair.
        let microjoules_per_pair = match model.as_str() {
            "rerank-english-v3.0" => 70,
            "rerank-multilingual-v3.0" => 80,
            "rerank-v3.5" => 120,
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

    /// Build the JSON request body for snapshot tests.
    pub fn build_body(&self, query: &str, candidates: &[Candidate]) -> serde_json::Value {
        let docs: Vec<&str> = candidates.iter().map(|c| c.text.as_str()).collect();
        json!({
            "model": self.model,
            "query": query,
            "documents": docs,
            "top_n": docs.len(),
        })
    }

    /// Estimate energy for re-ranking `pairs` candidates.
    pub fn joule_estimate(&self, pairs: usize) -> JouleCost {
        JouleCost::estimated((pairs as u64).saturating_mul(self.microjoules_per_pair))
    }
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    results: Vec<ApiResult>,
}

#[derive(Debug, Deserialize)]
struct ApiResult {
    index: usize,
    relevance_score: f32,
}

#[async_trait]
impl Reranker for CohereReranker {
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
                "{} candidates exceeds Cohere v2 limit {}",
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
                    .results
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
                warn!(attempt, wait, "cohere rerank 429; backing off");
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
        // Cohere v2 caps `documents` at 1000.
        1000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models_resolve() {
        assert!(CohereReranker::new("k", "rerank-english-v3.0").is_ok());
        assert!(CohereReranker::new("k", "rerank-multilingual-v3.0").is_ok());
        assert!(CohereReranker::new("k", "rerank-v3.5").is_ok());
    }

    #[test]
    fn unknown_model_errors() {
        assert!(matches!(
            CohereReranker::new("k", "nope"),
            Err(RerankError::ModelNotFound(_))
        ));
    }

    #[test]
    fn body_shape() {
        let c = CohereReranker::new("k", "rerank-english-v3.0").expect("ok");
        let body = c.build_body(
            "q",
            &[Candidate::new("a", "alpha"), Candidate::new("b", "beta")],
        );
        assert_eq!(body["model"], "rerank-english-v3.0");
        assert_eq!(body["query"], "q");
        assert_eq!(body["documents"][0], "alpha");
        assert_eq!(body["documents"][1], "beta");
        assert_eq!(body["top_n"], 2);
    }

    #[test]
    fn joule_estimate_scales_with_pairs() {
        let c = CohereReranker::new("k", "rerank-v3.5").expect("ok");
        let one = c.joule_estimate(1).microjoules;
        let hundred = c.joule_estimate(100).microjoules;
        assert_eq!(hundred, one * 100);
    }
}
