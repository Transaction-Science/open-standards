//! Integration test: naive RAG returns the most overlap-rich chunk
//! and records joule accounting in its trace.

use std::sync::Arc;

use eoc_rag::{
    Chunk, CitationPolicy, DocumentStore, InMemoryStore, NaivePipeline, Pipeline, RagRequest,
    Stage,
};

fn corpus() -> Arc<dyn DocumentStore> {
    Arc::new(InMemoryStore::from_chunks(
        "naive-test",
        vec![
            Chunk::new(
                "rfc-eoc",
                0,
                "Energy-optimized compute measures joules per byte across the cascade.",
            ),
            Chunk::new(
                "rfc-misc",
                0,
                "Bicycles have wheels and frames, often made of aluminium.",
            ),
            Chunk::new(
                "rfc-rag",
                0,
                "Retrieval augmented generation embeds the query and stuffs top-k chunks.",
            ),
        ],
    ))
}

#[tokio::test]
async fn naive_pipeline_finds_eoc_chunk() {
    let p = NaivePipeline::new(corpus()).with_citation_policy(CitationPolicy::Optional);
    let req = RagRequest::new("joules per byte energy-optimized compute", 3);
    let ans = p.answer(&req).await.expect("pipeline ok");
    assert!(ans.text.contains("joules"));
    assert!(ans.trace.events.iter().any(|e| e.stage == Stage::Retrieve));
    assert!(ans.trace.events.iter().any(|e| e.stage == Stage::Generate));
    assert!(ans.trace.total_microjoules() > 0);
}

#[tokio::test]
async fn naive_pipeline_no_match_returns_err() {
    let p = NaivePipeline::new(corpus()).with_citation_policy(CitationPolicy::Optional);
    let req = RagRequest::new("quantum chromodynamics gluon strong force", 3);
    let res = p.answer(&req).await;
    // The corpus has no overlap; jaccard yields zero -> NoChunks.
    assert!(res.is_err());
}
