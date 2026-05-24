//! Integration tests for the vendor re-rank HTTP backends.

use eoc_rerank::reranker::{Candidate, Reranker};
use eoc_rerank::{CohereReranker, VoyageReranker};
use eoc_rerank::error::RerankError;
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn cohere_body() -> serde_json::Value {
    json!({
        "id": "x",
        "results": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.20},
        ],
        "meta": {}
    })
}

fn voyage_body() -> serde_json::Value {
    json!({
        "object": "list",
        "data": [
            {"index": 1, "relevance_score": 0.99},
            {"index": 0, "relevance_score": 0.10},
        ],
        "model": "rerank-2",
        "usage": {"total_tokens": 1}
    })
}

#[tokio::test]
async fn cohere_happy_path_reorders_candidates() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/rerank"))
        .and(header("authorization", "Bearer sk-test"))
        .and(body_partial_json(json!({
            "model": "rerank-english-v3.0",
            "query": "what is EOC?",
            "documents": ["irrelevant filler", "Energy-optimized compute"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(cohere_body()))
        .mount(&server)
        .await;

    let r = CohereReranker::new("sk-test", "rerank-english-v3.0")
        .expect("ok")
        .with_endpoint(format!("{}/v2/rerank", server.uri()));

    let scored = r
        .rerank(
            "what is EOC?",
            &[
                Candidate::new("a", "irrelevant filler"),
                Candidate::new("b", "Energy-optimized compute"),
            ],
        )
        .await
        .expect("rerank ok");

    assert_eq!(scored.len(), 2);
    assert_eq!(scored[0].candidate.id, "b");
    assert_eq!(scored[0].rank, 1);
    assert!(scored[0].score > scored[1].score);
}

#[tokio::test]
async fn cohere_401_no_retry() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/rerank"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&server)
        .await;
    let r = CohereReranker::new("sk-bad", "rerank-english-v3.0")
        .expect("ok")
        .with_endpoint(format!("{}/v2/rerank", server.uri()));
    let res = r.rerank("q", &[Candidate::new("a", "x")]).await;
    assert!(matches!(res, Err(RerankError::InvalidApiKey)));
}

#[tokio::test]
async fn cohere_429_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/rerank"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v2/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(cohere_body()))
        .mount(&server)
        .await;
    let r = CohereReranker::new("k", "rerank-english-v3.0")
        .expect("ok")
        .with_endpoint(format!("{}/v2/rerank", server.uri()))
        .with_max_retries(3);
    let scored = r
        .rerank(
            "q",
            &[Candidate::new("a", "x"), Candidate::new("b", "y")],
        )
        .await
        .expect("eventually ok");
    assert_eq!(scored.len(), 2);
}

#[tokio::test]
async fn voyage_happy_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(voyage_body()))
        .mount(&server)
        .await;
    let r = VoyageReranker::new("k", "rerank-2")
        .expect("ok")
        .with_endpoint(format!("{}/v1/rerank", server.uri()));
    let scored = r
        .rerank(
            "q",
            &[Candidate::new("a", "x"), Candidate::new("b", "y")],
        )
        .await
        .expect("ok");
    assert_eq!(scored[0].candidate.id, "b");
    assert_eq!(scored[0].rank, 1);
}

#[tokio::test]
async fn voyage_429_exhausts_budget() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
        .mount(&server)
        .await;
    let r = VoyageReranker::new("k", "rerank-2")
        .expect("ok")
        .with_endpoint(format!("{}/v1/rerank", server.uri()))
        .with_max_retries(1);
    let res = r.rerank("q", &[Candidate::new("a", "x")]).await;
    assert!(matches!(res, Err(RerankError::RateLimited { .. })));
}
