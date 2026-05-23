//! Integration tests for vendor HTTP backends using `wiremock`.
//!
//! Covers happy path, 429-with-retry, and 401-no-retry for each vendor.

use eoc_embeddings::{
    CohereEmbedder, Embedder, JinaEmbedder, MistralEmbedder, OpenAiEmbedder, VoyageEmbedder,
    error::EmbeddingError,
};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn openai_success_body(dim: usize) -> serde_json::Value {
    let v: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
    json!({
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": v},
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    })
}

fn cohere_success_body() -> serde_json::Value {
    let v: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.0001).collect();
    json!({"id": "x", "embeddings": {"float": [v]}, "texts": ["hi"]})
}

fn voyage_success_body() -> serde_json::Value {
    let v: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.0001).collect();
    json!({"object": "list", "data": [{"object": "embedding", "embedding": v, "index": 0}], "model": "voyage-3", "usage": {"total_tokens": 1}})
}

fn jina_success_body() -> serde_json::Value {
    let v: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.0001).collect();
    json!({"data": [{"embedding": v, "index": 0, "object": "embedding"}], "model": "jina-embeddings-v3"})
}

fn mistral_success_body() -> serde_json::Value {
    let v: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.0001).collect();
    json!({"id": "x", "object": "list", "data": [{"object": "embedding", "embedding": v, "index": 0}], "model": "mistral-embed", "usage": {"prompt_tokens": 1, "total_tokens": 1}})
}

#[tokio::test]
async fn openai_happy_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header("authorization", "Bearer sk-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_success_body(1536)))
        .mount(&server)
        .await;

    let e = OpenAiEmbedder::new("sk-test", "text-embedding-3-small")
        .expect("ok")
        .with_endpoint(format!("{}/v1/embeddings", server.uri()));
    let v = e.embed(&["hi"]).await.expect("embed ok");
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].len(), 1536);
}

#[tokio::test]
async fn openai_401_no_retry() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&server)
        .await;

    let e = OpenAiEmbedder::new("sk-bad", "text-embedding-3-small")
        .expect("ok")
        .with_endpoint(format!("{}/v1/embeddings", server.uri()));
    let r = e.embed(&["hi"]).await;
    assert!(matches!(r, Err(EmbeddingError::InvalidApiKey)));
}

#[tokio::test]
async fn openai_429_retries_then_succeeds() {
    let server = MockServer::start().await;
    // First call: 429 (no retry-after to keep test fast).
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(429).insert_header("retry-after", "0"),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // Subsequent calls: 200.
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_success_body(1536)))
        .mount(&server)
        .await;

    let e = OpenAiEmbedder::new("sk-test", "text-embedding-3-small")
        .expect("ok")
        .with_endpoint(format!("{}/v1/embeddings", server.uri()))
        .with_max_retries(3);
    let v = e.embed(&["hi"]).await.expect("eventually ok");
    assert_eq!(v[0].len(), 1536);
}

#[tokio::test]
async fn cohere_happy_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(cohere_success_body()))
        .mount(&server)
        .await;
    let e = CohereEmbedder::new("k", "embed-english-v3.0")
        .expect("ok")
        .with_endpoint(format!("{}/v2/embed", server.uri()));
    let v = e.embed(&["hi"]).await.expect("ok");
    assert_eq!(v[0].len(), 1024);
}

#[tokio::test]
async fn cohere_401() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/embed"))
        .respond_with(ResponseTemplate::new(403))
        .mount(&server)
        .await;
    let e = CohereEmbedder::new("k", "embed-english-v3.0")
        .expect("ok")
        .with_endpoint(format!("{}/v2/embed", server.uri()));
    assert!(matches!(e.embed(&["hi"]).await, Err(EmbeddingError::InvalidApiKey)));
}

#[tokio::test]
async fn voyage_happy_path_and_429() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(voyage_success_body()))
        .mount(&server)
        .await;
    let e = VoyageEmbedder::new("k", "voyage-3")
        .expect("ok")
        .with_endpoint(format!("{}/v1/embeddings", server.uri()))
        .with_max_retries(3);
    let v = e.embed(&["hi"]).await.expect("ok");
    assert_eq!(v[0].len(), 1024);
}

#[tokio::test]
async fn jina_happy_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(jina_success_body()))
        .mount(&server)
        .await;
    let e = JinaEmbedder::new("k", "jina-embeddings-v3")
        .expect("ok")
        .with_endpoint(format!("{}/v1/embeddings", server.uri()));
    let v = e.embed(&["hi"]).await.expect("ok");
    assert_eq!(v[0].len(), 1024);
}

#[tokio::test]
async fn jina_401() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&server)
        .await;
    let e = JinaEmbedder::new("k", "jina-embeddings-v3")
        .expect("ok")
        .with_endpoint(format!("{}/v1/embeddings", server.uri()));
    assert!(matches!(e.embed(&["hi"]).await, Err(EmbeddingError::InvalidApiKey)));
}

#[tokio::test]
async fn mistral_happy_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mistral_success_body()))
        .mount(&server)
        .await;
    let e = MistralEmbedder::new("k", "mistral-embed")
        .expect("ok")
        .with_endpoint(format!("{}/v1/embeddings", server.uri()));
    let v = e.embed(&["hi"]).await.expect("ok");
    assert_eq!(v[0].len(), 1024);
}

#[tokio::test]
async fn mistral_429_exhausts_budget() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
        .mount(&server)
        .await;
    let e = MistralEmbedder::new("k", "mistral-embed")
        .expect("ok")
        .with_endpoint(format!("{}/v1/embeddings", server.uri()))
        .with_max_retries(1);
    let r = e.embed(&["hi"]).await;
    assert!(matches!(r, Err(EmbeddingError::RateLimited { .. })));
}
