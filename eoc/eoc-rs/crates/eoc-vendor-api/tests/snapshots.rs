//! Snapshot tests verifying that each vendor request body matches the
//! shape we promise to send. Uses `insta` inline snapshots.

use eoc_core::Query;
use eoc_vendor_api::{
    anthropic::AnthropicBackend, cohere::CohereBackend, google::GoogleBackend,
    openai_compat::build_request,
};

#[test]
fn anthropic_request_body_shape() {
    let backend = AnthropicBackend::new("test-key", "claude-3-5-sonnet-20241022")
        .with_max_tokens(2048)
        .without_stream();
    let q = Query::new("hello, world");
    let body = backend.build_request_body(&q);
    insta::assert_json_snapshot!("anthropic_basic", body);
}

#[test]
fn anthropic_cached_system_prompt() {
    let backend = AnthropicBackend::new("test-key", "claude-3-5-sonnet-20241022")
        .with_max_tokens(2048)
        .with_system("You are helpful.")
        .with_system_cached(true)
        .without_stream();
    let q = Query::new("hello, world");
    let body = backend.build_request_body(&q);
    insta::assert_json_snapshot!("anthropic_cached_system", body);
}

#[test]
fn openai_request_body_shape() {
    let q = Query::new("hello, world");
    let body = build_request("gpt-4o", &q, false);
    insta::assert_json_snapshot!("openai_basic", body);
}

#[test]
fn openai_streaming_flag() {
    let q = Query::new("hello, world");
    let body = build_request("gpt-4o", &q, true);
    insta::assert_json_snapshot!("openai_streaming", body);
}

#[test]
fn google_request_body_shape() {
    let backend = GoogleBackend::new("test-key", "gemini-1.5-pro").without_stream();
    let q = Query::new("hello, world");
    let body = backend.build_request_body(&q);
    insta::assert_json_snapshot!("google_basic", body);
}

#[test]
fn cohere_request_body_shape() {
    let backend = CohereBackend::new("test-key", "command-r-plus");
    let q = Query::new("hello, world");
    let body = backend.build_request_body(&q);
    insta::assert_json_snapshot!("cohere_basic", body);
}
