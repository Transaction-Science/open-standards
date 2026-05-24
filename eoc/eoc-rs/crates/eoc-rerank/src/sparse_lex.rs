//! SPLADE-style learned-sparse retrieval.
//!
//! SPLADE (Sparse Lexical AnD Expansion) is a BERT-based model that emits
//! a sparse vector over the WordPiece vocabulary for each input — terms
//! get expanded into related vocabulary tokens with learned weights. This
//! gives the lexical-match strength of BM25 with the semantic-expansion
//! strength of dense embeddings.
//!
//! This module ships the *index data structure* (an inverted index over
//! `(vocab_id, weight)` postings) plus a [`Retriever`] adapter. The actual
//! inference path is a stub — production deployments wire it into
//! `eoc-local` for the ONNX-runtime side. The trait surface is identical
//! to BM25.
//!
//! Recommended SPLADE checkpoints:
//!
//! * `naver/splade-cocondenser-ensembledistil`
//! * `naver/splade-v3`
//! * `naver/splade-v3-distilbert`

use std::collections::HashMap;

use async_trait::async_trait;

use crate::DocId;
use crate::error::{RerankError, RerankResult};
use crate::reranker::Retriever;

/// A sparse vector — `(vocab_id, weight)` pairs.
#[derive(Debug, Clone, Default)]
pub struct SparseVector {
    /// `(vocab_id, weight)` entries; weights must be non-negative.
    pub entries: Vec<(u32, f32)>,
}

impl SparseVector {
    /// Construct from entries. Discards entries with `weight <= 0`.
    pub fn new(entries: Vec<(u32, f32)>) -> Self {
        let entries = entries.into_iter().filter(|(_, w)| *w > 0.0).collect();
        Self { entries }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }
}

/// Posting for the sparse inverted index.
#[derive(Debug, Clone, Copy)]
struct SparsePosting {
    doc_idx: usize,
    weight: f32,
}

/// SPLADE-style sparse inverted index.
pub struct SparseLexIndex {
    docs: Vec<(DocId, String)>,
    inverted: HashMap<u32, Vec<SparsePosting>>,
    model_name: String,
}

impl SparseLexIndex {
    /// Construct an empty index for `model_name` — caller will populate
    /// via [`Self::add`].
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            docs: Vec::new(),
            inverted: HashMap::new(),
            model_name: model_name.into(),
        }
    }

    /// Insert a document with its pre-computed sparse vector.
    pub fn add(&mut self, id: DocId, text: String, vec: SparseVector) {
        let idx = self.docs.len();
        self.docs.push((id, text));
        for (vocab_id, weight) in vec.entries {
            self.inverted
                .entry(vocab_id)
                .or_default()
                .push(SparsePosting {
                    doc_idx: idx,
                    weight,
                });
        }
    }

    /// Score documents against a pre-computed query sparse vector.
    pub fn search_sparse(&self, query: &SparseVector, top_k: usize) -> Vec<(DocId, f32)> {
        if self.docs.is_empty() {
            return Vec::new();
        }
        let mut scores: HashMap<usize, f32> = HashMap::new();
        for (vocab_id, q_w) in &query.entries {
            let Some(postings) = self.inverted.get(vocab_id) else {
                continue;
            };
            for p in postings {
                *scores.entry(p.doc_idx).or_insert(0.0) += q_w * p.weight;
            }
        }
        let mut out: Vec<(usize, f32)> = scores.into_iter().collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(top_k);
        out.into_iter()
            .map(|(i, s)| (self.docs[i].0.clone(), s))
            .collect()
    }

    /// Encode `text` into a [`SparseVector`].
    ///
    /// This is the inference path — implemented by `eoc-local` when the
    /// SPLADE ONNX bundle is loaded. The stub here returns an error so
    /// callers compose with the real model explicitly.
    pub fn encode(&self, _text: &str) -> RerankResult<SparseVector> {
        Err(RerankError::Local(
            "SPLADE encode requires the eoc-local feature path with a loaded model".into(),
        ))
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Is the index empty?
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// SPLADE model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[async_trait]
impl Retriever for SparseLexIndex {
    async fn retrieve(&self, query: &str, top_k: usize) -> RerankResult<Vec<(DocId, f32)>> {
        let qv = self.encode(query)?;
        Ok(self.search_sparse(&qv, top_k))
    }

    fn document_text(&self, id: &DocId) -> Option<String> {
        self.docs
            .iter()
            .find(|(d, _)| d == id)
            .map(|(_, t)| t.clone())
    }

    fn name(&self) -> &str {
        "sparse-lex"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_search_known_vectors() {
        let mut idx = SparseLexIndex::new("naver/splade-v3");
        idx.add(
            "d1".into(),
            "first".into(),
            SparseVector::new(vec![(1, 1.0), (5, 0.5)]),
        );
        idx.add(
            "d2".into(),
            "second".into(),
            SparseVector::new(vec![(2, 1.0)]),
        );
        idx.add(
            "d3".into(),
            "third".into(),
            SparseVector::new(vec![(1, 0.2), (5, 0.9)]),
        );

        let q = SparseVector::new(vec![(1, 1.0), (5, 1.0)]);
        let hits = idx.search_sparse(&q, 3);
        assert!(!hits.is_empty());
        // d3 has both terms with sum 1.1; d1 has 1.5 — d1 should top.
        assert_eq!(hits[0].0, "d1");
    }

    #[test]
    fn encode_stub_errors() {
        let idx = SparseLexIndex::new("naver/splade-v3");
        assert!(idx.encode("anything").is_err());
    }

    #[test]
    fn nnz_filters_zero_weights() {
        let v = SparseVector::new(vec![(1, 0.0), (2, 0.5), (3, -1.0)]);
        assert_eq!(v.nnz(), 1);
    }
}
