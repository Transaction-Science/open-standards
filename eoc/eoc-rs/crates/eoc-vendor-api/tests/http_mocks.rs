//! End-to-end HTTP behaviour tests using `wiremock`.
//!
//! Covers the OpenAI-compatible path (which exercises four of the eight
//! backends: OpenAI, Mistral, Groq, Together, Fireworks). Each backend
//! shares the same retry / error-classification code in
//! `openai_compat::execute`, so testing the common path here gives full
//! coverage with one fixture set.

use std::time::Duration;

use eoc_core::{JouleSource, Query, Stage};
use eoc_neural::NeuralBackend;
use eoc_vendor_api::{
    config::{RetryPolicy, VendorConfig},
    error::VendorError,
    openai::OpenAiBackend,
    openai_compat,
};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn small_config() -> VendorConfig {
    VendorConfig::new()
        .with_timeout(Duration::from_secs(2))
        .with_retry_policy(RetryPolicy {
            max_retries: 2,
            initial_backoff: Duration::from_millis(1),
            backoff_factor: 1.0,
        })
}

#[tokio::test]
async fn openai_happy_path_200() {
    let server = MockServer::start().await;
    let body = json!({
        "choices": [{
            "message": {"role": "assistant", "content": "hi there"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2}
    });
    Mock::given(method("POST"))
        .and(path("/"))
        .and(header("Authorization", "Bearer sk-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let backend = OpenAiBackend::new("sk-test", "gpt-4o")
        .without_stream()
        .with_config(small_config().with_endpoint(server.uri()));

    let q = Query::new("hello");
    let r = backend.infer(&q).await;
    assert_eq!(r.payload, "hi there");
    assert_eq!(r.stage, Stage::Neural);
    assert_eq!(r.joule_cost.source, JouleSource::Estimated);
    // gpt-4o: 5*0.10 + 2*0.40 = 0.5 + 0.8 = 1.3 J = 1_300_000 µJ.
    assert_eq!(r.joule_cost.microjoules, 1_300_000);
}

#[tokio::test]
async fn openai_429_then_200_via_retry() {
    let server = MockServer::start().await;
    // First call: 429. Second call: 200.
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(429))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        })))
        .mount(&server)
        .await;

    let backend = OpenAiBackend::new("sk-test", "gpt-4o")
        .without_stream()
        .with_config(small_config().with_endpoint(server.uri()));

    let r = backend.infer(&Query::new("hi")).await;
    assert_eq!(r.payload, "ok");
}

#[tokio::test]
async fn openai_401_no_retry() {
    let server = MockServer::start().await;
    // 401 must NOT be retried. We expect exactly one inbound call.
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&server)
        .await;

    let cfg = small_config().with_endpoint(server.uri());
    let client = reqwest::Client::new();
    let result = openai_compat::execute(
        &client,
        &format!("{}/", server.uri()),
        &eoc_vendor_api::auth::Auth::Bearer("sk-bad".to_string()),
        "gpt-4o",
        &Query::new("hi"),
        false,
        &cfg,
    )
    .await;

    match result {
        Err(VendorError::InvalidApiKey) => {}
        other => panic!("expected InvalidApiKey, got {other:?}"),
    }
    // wiremock's `.expect(1)` verifies count on drop.
}

#[tokio::test]
async fn openai_429_exhausts_retries() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&server)
        .await;

    let cfg = small_config().with_endpoint(server.uri());
    let client = reqwest::Client::new();
    let result = openai_compat::execute(
        &client,
        &format!("{}/", server.uri()),
        &eoc_vendor_api::auth::Auth::Bearer("sk-test".to_string()),
        "gpt-4o",
        &Query::new("hi"),
        false,
        &cfg,
    )
    .await;
    match result {
        Err(VendorError::RateLimited { .. }) => {}
        other => panic!("expected RateLimited, got {other:?}"),
    }
}
