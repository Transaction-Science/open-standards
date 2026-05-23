//! Snapshot tests for request body shapes. Vendor APIs are picky; locking
//! the body format down prevents accidental drift.

use eoc_embeddings::{CohereEmbedder, JinaEmbedder, MistralEmbedder, OpenAiEmbedder, VoyageEmbedder};

#[test]
fn openai_body_snapshot() {
    let e = OpenAiEmbedder::new("sk-x", "text-embedding-3-small").expect("ok");
    let body = e.build_body(&["alpha", "beta"]);
    insta::assert_json_snapshot!("openai_body", body);
}

#[test]
fn cohere_body_snapshot() {
    let e = CohereEmbedder::new("k", "embed-english-v3.0").expect("ok");
    let body = e.build_body(&["alpha"]);
    insta::assert_json_snapshot!("cohere_body", body);
}

#[test]
fn voyage_body_snapshot() {
    let e = VoyageEmbedder::new("k", "voyage-3").expect("ok");
    let body = e.build_body(&["alpha"]);
    insta::assert_json_snapshot!("voyage_body", body);
}

#[test]
fn jina_body_snapshot() {
    let e = JinaEmbedder::new("k", "jina-embeddings-v3").expect("ok");
    let body = e.build_body(&["alpha"]);
    insta::assert_json_snapshot!("jina_body", body);
}

#[test]
fn mistral_body_snapshot() {
    let e = MistralEmbedder::new("k", "mistral-embed").expect("ok");
    let body = e.build_body(&["alpha"]);
    insta::assert_json_snapshot!("mistral_body", body);
}
