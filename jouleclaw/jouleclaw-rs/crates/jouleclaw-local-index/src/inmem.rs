//! [`InMemoryIndex`] — the deterministic, zero-dep reference
//! implementation of [`crate::LocalIndex`].
//!
//! Holds a `Vec<Document>` in process memory and scores queries with
//! a BM25-shaped function:
//!
//! ```text
//! score(q, d) = Σ idf(t) · ((f(t,d) · (k1 + 1)) / (f(t,d) + k1 · (1 - b + b · |d|/avg|d|)))
//! ```
//!
//! It is fast, deterministic, has no training step, and runs in
//! microjoule-class energy on a Pi-zero. Production deployments that
//! need durable storage or a persistent index should swap in tantivy /
//! sled / SQLite-FTS5 via the [`crate::LocalIndex`] trait; this
//! implementation is intentionally minimal.

use crate::index::{IndexError, IndexHit, LocalIndex};

// ─── Document ────────────────────────────────────────────────────

/// A single document held by [`InMemoryIndex`]. `id` is opaque to the
/// index; `text` is what gets tokenised and scored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    /// Opaque caller-side identifier.
    pub id: String,
    /// Document text fed to the scorer. The index tokenises this on
    /// every search; consumers with millions of documents should swap
    /// to a backend with pre-built inverted indexes.
    pub text: String,
}

impl Document {
    /// Construct a new document.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

// ─── Parameters ──────────────────────────────────────────────────

/// Tunable parameters for [`InMemoryIndex`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InMemoryParams {
    /// Term-frequency saturation. Robertson's "okapi" default is 1.2.
    pub k1: f32,
    /// Length-normalisation. 0 = none, 1 = fully normalised.
    /// Robertson's default is 0.75.
    pub b: f32,
}

impl Default for InMemoryParams {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
        }
    }
}

// ─── The index ───────────────────────────────────────────────────

/// Pure-Rust in-memory BM25-shaped index. Implements
/// [`crate::LocalIndex`].
#[derive(Debug, Clone, Default)]
pub struct InMemoryIndex {
    docs: Vec<Document>,
    params: InMemoryParams,
}

impl InMemoryIndex {
    /// Construct an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct an empty index with custom parameters.
    pub fn with_params(params: InMemoryParams) -> Self {
        Self {
            docs: Vec::new(),
            params,
        }
    }

    /// Borrow the active parameters.
    pub fn params(&self) -> &InMemoryParams {
        &self.params
    }

    /// Insert a single document. Duplicate IDs are allowed — the
    /// reference impl does not dedupe.
    pub fn insert(&mut self, doc: Document) {
        self.docs.push(doc);
    }

    /// Bulk-insert documents.
    pub fn extend<I: IntoIterator<Item = Document>>(&mut self, docs: I) {
        self.docs.extend(docs);
    }

    /// Borrow the corpus.
    pub fn docs(&self) -> &[Document] {
        &self.docs
    }

    /// Tokenise a string into lowercase alphanumeric word tokens.
    /// Punctuation and whitespace are skipped; non-ASCII letters pass
    /// through via `char::is_alphanumeric`.
    pub fn tokenize(text: &str) -> Vec<String> {
        let mut tokens: Vec<String> = Vec::new();
        let mut cur = String::new();
        for c in text.chars() {
            if c.is_alphanumeric() {
                for lc in c.to_lowercase() {
                    cur.push(lc);
                }
            } else if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            tokens.push(cur);
        }
        tokens
    }

    /// Score every doc against the query and return all non-zero hits
    /// in descending order.
    fn score_all(&self, query: &str) -> Vec<IndexHit> {
        let query_terms = Self::tokenize(query);
        if query_terms.is_empty() || self.docs.is_empty() {
            return Vec::new();
        }

        let tokenised: Vec<Vec<String>> = self
            .docs
            .iter()
            .map(|d| Self::tokenize(&d.text))
            .collect();
        let n_docs = self.docs.len() as f32;
        let total_len: usize = tokenised.iter().map(|t| t.len()).sum();
        let avg_len_raw = if n_docs > 0.0 {
            total_len as f32 / n_docs
        } else {
            1.0
        };
        let avg_len = if avg_len_raw <= 0.0 { 1.0 } else { avg_len_raw };

        // Document-frequency per query term.
        let mut df: std::collections::HashMap<&str, u32> =
            std::collections::HashMap::new();
        for term in &query_terms {
            let mut count = 0u32;
            for toks in &tokenised {
                if toks.iter().any(|t| t == term) {
                    count += 1;
                }
            }
            df.insert(term.as_str(), count);
        }

        let k1 = self.params.k1;
        let b = self.params.b;

        let mut hits: Vec<IndexHit> = Vec::with_capacity(self.docs.len());
        for (i, toks) in tokenised.iter().enumerate() {
            let dl = toks.len() as f32;
            let mut score = 0.0f32;
            for term in &query_terms {
                let tf = toks.iter().filter(|t| *t == term).count() as f32;
                if tf == 0.0 {
                    continue;
                }
                let n_qi = *df.get(term.as_str()).unwrap_or(&0) as f32;
                // Smoothed IDF: ln(1 + (N - n + 0.5) / (n + 0.5)).
                let idf =
                    ((n_docs - n_qi + 0.5) / (n_qi + 0.5) + 1.0).ln();
                let denom = tf + k1 * (1.0 - b + b * (dl / avg_len));
                let numer = tf * (k1 + 1.0);
                score += idf * (numer / denom);
            }
            if score > 0.0 {
                let doc = &self.docs[i];
                hits.push(IndexHit::new(&doc.id, &doc.text, score));
            }
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits
    }
}

impl LocalIndex for InMemoryIndex {
    fn search(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<IndexHit>, IndexError> {
        if query.is_empty() {
            return Err(IndexError::Input("query is empty".into()));
        }
        let mut hits = self.score_all(query);
        if k < hits.len() {
            hits.truncate(k);
        }
        Ok(hits)
    }

    fn doc_count(&self) -> usize {
        self.docs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn five_doc_corpus() -> InMemoryIndex {
        let mut idx = InMemoryIndex::new();
        idx.extend(vec![
            Document::new("d1", "Paris is the capital of France."),
            Document::new("d2", "Berlin is the capital of Germany."),
            Document::new("d3", "Madrid is the capital of Spain."),
            Document::new("d4", "Rome is the capital of Italy."),
            Document::new("d5", "Lisbon is the capital of Portugal."),
        ]);
        idx
    }

    #[test]
    fn default_params_match_robertson() {
        let p = InMemoryParams::default();
        assert!((p.k1 - 1.2).abs() < 1e-6);
        assert!((p.b - 0.75).abs() < 1e-6);
    }

    #[test]
    fn new_is_empty() {
        let idx = InMemoryIndex::new();
        assert_eq!(idx.doc_count(), 0);
        assert!(idx.docs().is_empty());
    }

    #[test]
    fn insert_grows_corpus() {
        let mut idx = InMemoryIndex::new();
        idx.insert(Document::new("a", "alpha"));
        idx.insert(Document::new("b", "beta"));
        assert_eq!(idx.doc_count(), 2);
    }

    #[test]
    fn tokenize_lowercases_and_splits() {
        let toks = InMemoryIndex::tokenize("Hello, WORLD! 123");
        assert_eq!(toks, vec!["hello", "world", "123"]);
    }

    #[test]
    fn tokenize_empty_string_is_empty() {
        assert!(InMemoryIndex::tokenize("").is_empty());
    }

    #[test]
    fn search_empty_query_errors() {
        let idx = five_doc_corpus();
        let err = idx.search("", 5).expect_err("empty query must error");
        assert!(matches!(err, IndexError::Input(_)));
    }

    #[test]
    fn search_empty_corpus_returns_empty() {
        let idx = InMemoryIndex::new();
        let hits = idx.search("anything", 5).expect("ok");
        assert!(hits.is_empty());
    }

    #[test]
    fn search_returns_descending_by_score() {
        let idx = five_doc_corpus();
        let hits = idx.search("capital", 5).expect("ok");
        assert!(!hits.is_empty());
        for w in hits.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "not descending: {} then {}",
                w[0].score,
                w[1].score,
            );
        }
    }

    #[test]
    fn search_top_hit_is_most_relevant() {
        let idx = five_doc_corpus();
        let hits = idx.search("Paris France", 5).expect("ok");
        assert!(!hits.is_empty());
        assert_eq!(hits[0].doc_id, "d1");
    }

    #[test]
    fn search_truncates_to_k() {
        let idx = five_doc_corpus();
        let hits = idx.search("capital", 2).expect("ok");
        assert!(hits.len() <= 2);
    }

    #[test]
    fn search_filters_zero_score_docs() {
        let mut idx = InMemoryIndex::new();
        idx.insert(Document::new("hit", "alpha beta gamma"));
        idx.insert(Document::new("miss", "completely unrelated"));
        let hits = idx.search("alpha", 5).expect("ok");
        // The "miss" doc has zero overlap with the query, so it must
        // not appear in the result set.
        assert!(hits.iter().all(|h| h.doc_id != "miss"));
        assert!(hits.iter().any(|h| h.doc_id == "hit"));
    }

    #[test]
    fn search_deterministic_on_repeat() {
        let idx = five_doc_corpus();
        let first = idx.search("capital France", 3).expect("ok");
        let second = idx.search("capital France", 3).expect("ok");
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.doc_id, b.doc_id);
            assert!((a.score - b.score).abs() < 1e-6);
        }
    }

    #[test]
    fn search_scores_are_nonnegative() {
        let idx = five_doc_corpus();
        let hits = idx.search("capital", 5).expect("ok");
        for h in &hits {
            assert!(h.score >= 0.0, "negative score: {}", h.score);
        }
    }

    #[test]
    fn with_params_overrides_defaults() {
        let idx = InMemoryIndex::with_params(InMemoryParams {
            k1: 2.0,
            b: 0.5,
        });
        assert!((idx.params().k1 - 2.0).abs() < 1e-6);
        assert!((idx.params().b - 0.5).abs() < 1e-6);
    }
}
