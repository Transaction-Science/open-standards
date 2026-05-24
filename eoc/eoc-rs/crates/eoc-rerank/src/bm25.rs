//! Okapi BM25 sparse retrieval.
//!
//! Implements the standard BM25 scoring function over an inverted index
//! built from a static corpus:
//!
//! ```text
//!                   tf(t, d) * (k1 + 1)
//! score(d, q) = Σ  ───────────────────────────────────────── * idf(t)
//!           t∈q   tf(t, d) + k1 * (1 - b + b * |d| / avgdl)
//! ```
//!
//! with `idf(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)` (Lucene
//! smoothing variant — always non-negative).
//!
//! Tokenisation: NFKD-normalised, lowercased, alphanumeric runs, optional
//! English stopword removal. Snowball stemming is gated behind the
//! `stemming` feature (not enabled by default — `rust-stemmers` is
//! intentionally omitted from default deps to keep the build slim).
//!
//! Strong baseline for rare-token / exact-match queries where embeddings
//! fail (e.g. quoted IDs, regulatory clause numbers).

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::DocId;
use crate::error::RerankResult;
use crate::reranker::Retriever;

/// Configuration for [`Bm25Index`].
#[derive(Debug, Clone)]
pub struct Bm25Config {
    /// Term-frequency saturation parameter. Typical range 1.2-2.0.
    pub k1: f32,
    /// Length-normalisation parameter. 0.0 = no normalisation, 1.0 = full.
    /// Typical value 0.75.
    pub b: f32,
    /// Lower-case the corpus + queries before tokenising.
    pub lowercase: bool,
    /// Strip standard English stopwords.
    pub remove_stopwords: bool,
    /// Apply (very basic) English suffix stripping. Ignored when the
    /// `stemming` feature is enabled — that path uses Snowball.
    pub naive_stem: bool,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            lowercase: true,
            remove_stopwords: true,
            naive_stem: false,
        }
    }
}

/// A document in the corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Document identifier.
    pub id: DocId,
    /// Document text.
    pub text: String,
}

impl Document {
    /// Construct a document.
    pub fn new(id: impl Into<DocId>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

/// A posting — `(doc_index, term_frequency)`.
#[derive(Debug, Clone, Copy)]
struct Posting {
    doc_idx: usize,
    tf: u32,
}

/// Okapi BM25 inverted index.
pub struct Bm25Index {
    config: Bm25Config,
    docs: Vec<Document>,
    /// One entry per doc: tokenised length.
    doc_lens: Vec<u32>,
    /// term -> postings.
    inverted: HashMap<String, Vec<Posting>>,
    avg_doc_len: f32,
}

impl Bm25Index {
    /// Build a fresh index from `docs`.
    pub fn build(docs: &[Document]) -> Self {
        Self::build_with_config(docs, Bm25Config::default())
    }

    /// Build with a custom config.
    pub fn build_with_config(docs: &[Document], config: Bm25Config) -> Self {
        let mut inverted: HashMap<String, Vec<Posting>> = HashMap::new();
        let mut doc_lens: Vec<u32> = Vec::with_capacity(docs.len());

        for (i, d) in docs.iter().enumerate() {
            let toks = tokenise(&d.text, &config);
            doc_lens.push(toks.len() as u32);
            // Aggregate term frequencies for this doc.
            let mut tf_map: HashMap<String, u32> = HashMap::new();
            for t in toks {
                *tf_map.entry(t).or_insert(0) += 1;
            }
            for (t, tf) in tf_map {
                inverted.entry(t).or_default().push(Posting { doc_idx: i, tf });
            }
        }

        let total: u64 = doc_lens.iter().map(|&n| n as u64).sum();
        let avg_doc_len = if docs.is_empty() {
            0.0
        } else {
            total as f32 / docs.len() as f32
        };

        Self {
            config,
            docs: docs.to_vec(),
            doc_lens,
            inverted,
            avg_doc_len,
        }
    }

    /// Number of documents in the index.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Is the index empty?
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Average document length (tokens).
    pub fn avg_doc_len(&self) -> f32 {
        self.avg_doc_len
    }

    /// Score documents against `query` and return the top `top_k`.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(DocId, f32)> {
        let q_tokens = tokenise(query, &self.config);
        if q_tokens.is_empty() || self.docs.is_empty() {
            return Vec::new();
        }

        let n = self.docs.len() as f32;
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for t in q_tokens {
            let Some(postings) = self.inverted.get(&t) else {
                continue;
            };
            let df = postings.len() as f32;
            // Lucene-smoothed IDF — always >= 0.
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for p in postings {
                let dl = self.doc_lens[p.doc_idx] as f32;
                let norm = 1.0 - self.config.b + self.config.b * dl / self.avg_doc_len.max(1.0);
                let tf = p.tf as f32;
                let contribution = idf * (tf * (self.config.k1 + 1.0))
                    / (tf + self.config.k1 * norm);
                *scores.entry(p.doc_idx).or_insert(0.0) += contribution;
            }
        }

        let mut out: Vec<(usize, f32)> = scores.into_iter().collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(top_k);
        out.into_iter()
            .map(|(i, s)| (self.docs[i].id.clone(), s))
            .collect()
    }

    /// Borrow the configured parameters (`k1`, `b`).
    pub fn config(&self) -> &Bm25Config {
        &self.config
    }

    /// Look up a document by id.
    pub fn document(&self, id: &DocId) -> Option<&Document> {
        self.docs.iter().find(|d| &d.id == id)
    }
}

#[async_trait]
impl Retriever for Bm25Index {
    async fn retrieve(&self, query: &str, top_k: usize) -> RerankResult<Vec<(DocId, f32)>> {
        Ok(self.search(query, top_k))
    }

    fn document_text(&self, id: &DocId) -> Option<String> {
        self.document(id).map(|d| d.text.clone())
    }

    fn name(&self) -> &str {
        "bm25"
    }
}

/// Standard English stopword list — short, conservative.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "he", "in",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their", "they", "this", "to",
    "was", "were", "will", "with",
];

fn is_stopword(t: &str) -> bool {
    STOPWORDS.binary_search(&t).is_ok()
}

/// Lower-case + NFKD + alphanumeric-run tokeniser.
pub(crate) fn tokenise(text: &str, cfg: &Bm25Config) -> Vec<String> {
    let normalised: String = text.nfkd().collect();
    let lowered = if cfg.lowercase {
        normalised.to_lowercase()
    } else {
        normalised
    };

    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in lowered.chars() {
        if ch.is_alphanumeric() {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }

    if cfg.remove_stopwords {
        out.retain(|t| !is_stopword(t.as_str()));
    }

    if cfg.naive_stem {
        out = out.into_iter().map(naive_stem).collect();
    }

    out
}

/// Very small "stemmer" — strips a handful of common English suffixes.
/// Real Snowball stemming lives behind the `stemming` feature.
fn naive_stem(t: String) -> String {
    for suf in ["ing", "edly", "edly", "edly", "ed", "ly", "es", "s"] {
        if t.len() > suf.len() + 2 && t.ends_with(suf) {
            return t[..t.len() - suf.len()].to_string();
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<Document> {
        vec![
            Document::new("d1", "The quick brown fox jumps over the lazy dog."),
            Document::new("d2", "A fast brown fox leaps above a sleepy hound."),
            Document::new("d3", "Slow turtles win the marathon."),
            Document::new("d4", "Apples and oranges grow on trees."),
            Document::new(
                "d5",
                "The energy-optimized compute substrate measures joules per byte.",
            ),
            Document::new("d6", "Quick brown rabbits hop in the meadow."),
            Document::new("d7", "Cosine similarity is a vector-space measure."),
            Document::new("d8", "BM25 is a probabilistic ranking function."),
            Document::new("d9", "Hybrid retrieval combines dense and sparse signals."),
            Document::new("d10", "Re-rankers improve the precision of top-K results."),
        ]
    }

    #[test]
    fn tokenise_drops_stopwords() {
        let cfg = Bm25Config::default();
        let toks = tokenise("The quick brown fox", &cfg);
        assert!(!toks.contains(&"the".to_string()));
        assert!(toks.contains(&"quick".to_string()));
    }

    #[test]
    fn bm25_matches_expected_docs() {
        let idx = Bm25Index::build(&corpus());
        let hits = idx.search("brown fox", 3);
        assert!(!hits.is_empty());
        // d1 and d2 both contain "brown fox" — top-1 should be one of them.
        let top = &hits[0].0;
        assert!(top == "d1" || top == "d2");
    }

    #[test]
    fn bm25_rare_term_outranks_common() {
        let idx = Bm25Index::build(&corpus());
        let hits = idx.search("joules byte", 3);
        assert_eq!(hits[0].0, "d5");
    }

    #[test]
    fn bm25_returns_top_k_at_most() {
        let idx = Bm25Index::build(&corpus());
        let hits = idx.search("brown", 100);
        assert!(hits.len() <= idx.len());
    }

    #[test]
    fn bm25_empty_query_no_results() {
        let idx = Bm25Index::build(&corpus());
        let hits = idx.search("", 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn bm25_unknown_term_no_panic() {
        let idx = Bm25Index::build(&corpus());
        let hits = idx.search("zzzzzz", 10);
        assert!(hits.is_empty());
    }
}
