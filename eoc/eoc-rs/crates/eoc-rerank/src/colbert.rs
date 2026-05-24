//! ColBERT v2 late-interaction retrieval (multi-vector).
//!
//! ColBERT (Khattab & Zaharia 2020, ColBERTv2 Santhanam et al. 2022)
//! represents each document as a *bag of token-level vectors* — one
//! vector per BERT WordPiece position. Query-document relevance is the
//! sum over query-token MaxSim:
//!
//! ```text
//! score(q, d) = Σ_i  max_j  q_i · d_j
//! ```
//!
//! This trades index size (≈ one vector per token vs one per document)
//! for ranking quality — ColBERT v2 routinely outperforms single-vector
//! dense retrieval on MS MARCO, BEIR, and LoTTE.
//!
//! ## Caveats
//!
//! * The index is heavier — typically 100-150 bytes per document token
//!   after PLAID compression. Plan for it.
//! * The ONNX inference path lives in `eoc-local` (the actual ColBERT
//!   model load). This module holds the *index data structure* and the
//!   MaxSim scorer.
//! * Gated behind the `local` feature.

use std::collections::BTreeMap;

use crate::DocId;

/// A multi-vector document representation — one vector per token.
#[derive(Debug, Clone)]
pub struct MultiVecDoc {
    /// Document id.
    pub id: DocId,
    /// `(token_count, dim)` matrix flattened row-major as `Vec<Vec<f32>>`.
    pub token_vectors: Vec<Vec<f32>>,
}

/// In-memory ColBERT index.
#[derive(Default)]
pub struct ColBertIndex {
    docs: Vec<MultiVecDoc>,
    dim: usize,
    text: BTreeMap<DocId, String>,
}

impl ColBertIndex {
    /// Construct an empty index.
    pub fn new(dim: usize) -> Self {
        Self {
            docs: Vec::new(),
            dim,
            text: BTreeMap::new(),
        }
    }

    /// Insert a document and its multi-vector encoding.
    pub fn insert(&mut self, doc: MultiVecDoc, text: String) {
        self.text.insert(doc.id.clone(), text);
        self.docs.push(doc);
    }

    /// Number of documents.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Is the index empty?
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Compute MaxSim scores for `query_tokens` (one vector per query
    /// WordPiece) against every indexed document. Returns the top `top_k`
    /// `(DocId, score)` pairs sorted descending by score.
    pub fn search(&self, query_tokens: &[Vec<f32>], top_k: usize) -> Vec<(DocId, f32)> {
        if self.docs.is_empty() || query_tokens.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(DocId, f32)> = self
            .docs
            .iter()
            .map(|d| (d.id.clone(), max_sim(query_tokens, &d.token_vectors)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Look up document text.
    pub fn document_text(&self, id: &DocId) -> Option<String> {
        self.text.get(id).cloned()
    }
}

/// MaxSim: for each query vector, take the max dot-product against any
/// doc vector, sum across query positions.
pub fn max_sim(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
    let mut total = 0.0f32;
    for q in query_tokens {
        let mut best = f32::NEG_INFINITY;
        for d in doc_tokens {
            let n = q.len().min(d.len());
            let mut dot = 0.0f32;
            for i in 0..n {
                dot += q[i] * d[i];
            }
            if dot > best {
                best = dot;
            }
        }
        if best.is_finite() {
            total += best;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_sim_picks_best_match_per_query_token() {
        // 2D query: one token along x, one along y.
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // Doc with one token along x and one along y.
        let d_match = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // Doc with two tokens along x — second query token has no good match.
        let d_partial = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let s_match = max_sim(&q, &d_match);
        let s_partial = max_sim(&q, &d_partial);
        assert!(s_match > s_partial);
    }

    #[test]
    fn index_round_trip() {
        let mut idx = ColBertIndex::new(2);
        idx.insert(
            MultiVecDoc {
                id: "d1".into(),
                token_vectors: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            },
            "first document".into(),
        );
        idx.insert(
            MultiVecDoc {
                id: "d2".into(),
                token_vectors: vec![vec![1.0, 0.0], vec![1.0, 0.0]],
            },
            "second document".into(),
        );
        let hits = idx.search(&[vec![0.0, 1.0]], 2);
        assert_eq!(hits[0].0, "d1");
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.dim(), 2);
        assert_eq!(idx.document_text(&"d1".to_string()).as_deref(), Some("first document"));
    }
}
