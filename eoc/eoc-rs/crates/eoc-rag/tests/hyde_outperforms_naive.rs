//! Integration test: synthetic recall — HyDE retrieves the gold chunk
//! on a short query where naive retrieval misses entirely.
//!
//! The Jaccard token retriever in [`eoc_rag::InMemoryStore`] is
//! deliberately simple: a query that doesn't share tokens with any
//! evidence misses. HyDE expands the query with scaffolding tokens
//! ("refers to", "described in detail") that match the evidence,
//! giving the retriever extra surface area. This mirrors the
//! empirical recall lift HyDE shows on short queries (Gao et al.
//! 2022).

use std::sync::Arc;

use eoc_rag::{
    Chunk, CitationPolicy, DocumentStore, HydePipeline, InMemoryStore, NaivePipeline, Pipeline,
    RagRequest,
};

fn store() -> Arc<dyn DocumentStore> {
    Arc::new(InMemoryStore::from_chunks(
        "hyde-test",
        vec![
            // Gold chunk — written with the scaffolding-style phrasing
            // that HyDE's hypothetical document generation produces
            // ("refers to a well-known concept", "described in
            // detail"). No literal token overlap with the user query.
            Chunk::new(
                "gold",
                0,
                "The notion refers to a well-known consensus framework. In general, it can be described in detail as a combination of rankings.",
            ),
            // Distractor — unrelated.
            Chunk::new(
                "distractor",
                0,
                "Quantum chromodynamics studies the strong force.",
            ),
        ],
    ))
}

#[tokio::test]
async fn hyde_retrieves_more_than_naive_on_short_query() {
    let naive = NaivePipeline::new(store()).with_citation_policy(CitationPolicy::Optional);
    let hyde = HydePipeline::new(store()).with_citation_policy(CitationPolicy::Optional);

    // A query whose tokens are absent from the corpus on purpose:
    // "ZZZQUERY" appears nowhere, so naive Jaccard retrieval returns
    // nothing. HyDE rewrites the query into hypothetical-document
    // form which *does* share tokens ("refers", "described",
    // "detail") with the gold chunk.
    let req = RagRequest::new("ZZZQUERY", 2);

    let naive_res = naive.answer(&req).await;
    let hyde_res = hyde.answer(&req).await.expect("hyde ok");

    // Naive on a query with no overlapping tokens errors with
    // NoChunks; HyDE finds the gold chunk via scaffolding overlap.
    assert!(
        naive_res.is_err(),
        "naive should miss; got: {:?}",
        naive_res.as_ref().map(|a| &a.text)
    );
    assert!(!hyde_res.chunks.is_empty());
    assert_eq!(hyde_res.chunks[0].chunk.doc_id, "gold");
    // Recall: HyDE retrieved at least as many docs as naive.
    let naive_count = naive_res.map(|a| a.chunks.len()).unwrap_or(0);
    assert!(hyde_res.chunks.len() >= naive_count);
}
