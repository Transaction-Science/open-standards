//! Integration test: RAG-Fusion's RRF promotes chunks that appear
//! across multiple rewrites.

use eoc_rag::{Chunk, RetrievedChunk, reciprocal_rank_fusion};

fn rc(doc: &str, text: &str, score: f32, rank: usize) -> RetrievedChunk {
    RetrievedChunk {
        chunk: Chunk::new(doc, 0, text),
        score,
        rank,
    }
}

#[test]
fn rrf_consensus_wins() {
    // Three rewrite rankings. `consensus` is top-3 in all three;
    // `solo` is top-1 in only one. RRF should promote `consensus`.
    let r1 = vec![
        rc("solo", "x", 0.99, 1),
        rc("consensus", "x", 0.50, 2),
        rc("filler1", "x", 0.10, 3),
    ];
    let r2 = vec![
        rc("consensus", "x", 0.90, 1),
        rc("filler2", "x", 0.40, 2),
        rc("solo", "x", 0.05, 3),
    ];
    let r3 = vec![
        rc("consensus", "x", 0.80, 1),
        rc("filler3", "x", 0.30, 2),
        rc("filler1", "x", 0.20, 3),
    ];
    let fused = reciprocal_rank_fusion(&[r1, r2, r3], 60.0, 5);
    assert_eq!(fused[0].chunk.doc_id, "consensus");
    // 1-based rank populated.
    assert_eq!(fused[0].rank, 1);
}

#[test]
fn rrf_empty_input_safe() {
    let fused: Vec<RetrievedChunk> = reciprocal_rank_fusion(&[], 60.0, 5);
    assert!(fused.is_empty());
}
