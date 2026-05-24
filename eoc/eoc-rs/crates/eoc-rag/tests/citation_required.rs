//! Integration test: under `CitationPolicy::Required`, the
//! enforcement gate rejects answers with no citations.

use eoc_rag::{
    CitationEnforcement, CitationPolicy, Chunk, Cite, RagError, RetrievedChunk,
};

#[test]
fn required_policy_rejects_empty_citations() {
    let gate = CitationEnforcement::new(CitationPolicy::Required);
    let chunks = vec![RetrievedChunk {
        chunk: Chunk::new("d1", 0, "evidence text"),
        score: 1.0,
        rank: 1,
    }];
    let err = gate
        .enforce("an answer with no cites", &chunks, &[])
        .expect_err("must reject");
    matches!(err, RagError::CitationRequired);
}

#[test]
fn required_policy_accepts_with_cite() {
    let gate = CitationEnforcement::new(CitationPolicy::Required);
    let chunk = Chunk::new("d1", 0, "evidence text");
    let cites = vec![Cite::whole(&chunk)];
    let chunks = vec![RetrievedChunk {
        chunk,
        score: 1.0,
        rank: 1,
    }];
    assert!(
        gate.enforce("an answer that cites evidence", &chunks, &cites)
            .is_ok()
    );
}

#[test]
fn optional_policy_accepts_empty_citations() {
    let gate = CitationEnforcement::new(CitationPolicy::Optional);
    let chunks: Vec<RetrievedChunk> = Vec::new();
    assert!(gate.enforce("no cites no problem", &chunks, &[]).is_ok());
}
