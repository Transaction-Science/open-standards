//! [`Bm25Reranker`] — the deterministic, zero-neural reference
//! implementation of [`crate::Reranker`].
//!
//! BM25 is the classical sparse-lexical retrieval/ranking function:
//!
//! ```text
//! score(q, d) = Σ idf(t) · ((f(t,d) · (k1 + 1)) / (f(t,d) + k1 · (1 - b + b · |d|/avg|d|)))
//! ```
//!
//! It is fast, deterministic, has no training step, runs on a Pi-zero
//! at a few microjoules per document, and provides a useful floor
//! against which to benchmark neural rerankers. Production deployments
//! that need state-of-the-art relevance ordering should swap in a
//! ColBERT/SPLADE/cross-encoder reranker; this implementation is
//! intentionally minimal.

use crate::reranker::{Doc, RerankError, RerankScore, Reranker};

// ─── BM25 parameters ─────────────────────────────────────────────

/// Tunable parameters for [`Bm25Reranker`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Params {
    /// Term-frequency saturation. Higher = TF matters more per
    /// occurrence. Robertson's "okapi" default is 1.2.
    pub k1: f32,
    /// Length-normalisation. 0 = no normalisation, 1 = fully
    /// normalised. Robertson's default is 0.75.
    pub b: f32,
    /// Self-reported joule cost per scored document. Default ~10 µJ
    /// reflects a Pi-class CPU doing ~10 µs of work per doc.
    pub joules_per_doc: f64,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            joules_per_doc: 10e-6,
        }
    }
}

// ─── The reranker ────────────────────────────────────────────────

/// Pure-Rust BM25 reranker. Constructed with default parameters or
/// custom [`Bm25Params`]. Implements [`Reranker`].
#[derive(Debug, Clone)]
pub struct Bm25Reranker {
    params: Bm25Params,
    name: String,
}

impl Bm25Reranker {
    /// Construct with Robertson's default parameters (`k1=1.2`,
    /// `b=0.75`).
    pub fn new() -> Self {
        Self {
            params: Bm25Params::default(),
            name: "bm25".to_string(),
        }
    }

    /// Construct with custom parameters.
    pub fn with_params(params: Bm25Params) -> Self {
        Self {
            params,
            name: "bm25".to_string(),
        }
    }

    /// Override the reported name (useful for differentiating tuned
    /// variants in calibration reports).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Borrow the active parameters.
    pub fn params(&self) -> &Bm25Params {
        &self.params
    }

    /// Tokenise a string into lowercase alphanumeric word tokens.
    /// Punctuation and whitespace are skipped; CJK / non-ASCII letters
    /// pass through via `char::is_alphanumeric`.
    fn tokenize(text: &str) -> Vec<String> {
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

    /// Score every doc against the query using the standard BM25
    /// formula. IDF uses the +0.5 smoothed variant ("BM25+ smoothed
    /// IDF"); the additive 1.0 keeps every term's contribution
    /// non-negative.
    fn score_docs(&self, query: &str, docs: &[Doc]) -> Vec<f32> {
        let query_terms = Self::tokenize(query);
        if query_terms.is_empty() || docs.is_empty() {
            return vec![0.0; docs.len()];
        }

        // Per-doc tokenisation and lengths.
        let tokenised: Vec<Vec<String>> =
            docs.iter().map(|d| Self::tokenize(&d.text)).collect();
        let n_docs = docs.len() as f32;
        let total_len: usize = tokenised.iter().map(|t| t.len()).sum();
        let avg_len = if n_docs > 0.0 {
            total_len as f32 / n_docs
        } else {
            1.0
        };
        let avg_len = if avg_len <= 0.0 { 1.0 } else { avg_len };

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

        let mut scores = vec![0.0f32; docs.len()];
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
                // Always non-negative; ln(1 + …) keeps zero-floor.
                let idf =
                    ((n_docs - n_qi + 0.5) / (n_qi + 0.5) + 1.0).ln();
                let denom = tf + k1 * (1.0 - b + b * (dl / avg_len));
                let numer = tf * (k1 + 1.0);
                score += idf * (numer / denom);
            }
            scores[i] = score;
        }
        scores
    }
}

impl Default for Bm25Reranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker for Bm25Reranker {
    fn name(&self) -> &str {
        &self.name
    }

    fn rerank(
        &self,
        query: &str,
        docs: &[Doc],
    ) -> Result<Vec<RerankScore>, RerankError> {
        if query.is_empty() {
            return Err(RerankError::Input("query is empty".into()));
        }
        let scores = self.score_docs(query, docs);
        Ok(docs
            .iter()
            .zip(scores.iter())
            .map(|(d, s)| RerankScore::new(&d.id, *s))
            .collect())
    }

    fn typical_joules_per_doc(&self) -> f64 {
        self.params.joules_per_doc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params_match_robertson() {
        let p = Bm25Params::default();
        assert!((p.k1 - 1.2).abs() < 1e-6);
        assert!((p.b - 0.75).abs() < 1e-6);
        assert!((p.joules_per_doc - 10e-6).abs() < 1e-12);
    }

    #[test]
    fn name_defaults_to_bm25() {
        let r = Bm25Reranker::new();
        assert_eq!(r.name(), "bm25");
    }

    #[test]
    fn name_override_round_trips() {
        let r = Bm25Reranker::new().with_name("bm25-tuned");
        assert_eq!(r.name(), "bm25-tuned");
    }

    #[test]
    fn tokenize_lowercases_and_splits() {
        let toks = Bm25Reranker::tokenize("Hello, World! 123");
        assert_eq!(toks, vec!["hello", "world", "123"]);
    }

    #[test]
    fn tokenize_handles_empty_string() {
        assert!(Bm25Reranker::tokenize("").is_empty());
    }

    #[test]
    fn rerank_empty_query_errors() {
        let r = Bm25Reranker::new();
        let err = r.rerank("", &[Doc::new("a", "x")]).unwrap_err();
        assert!(matches!(err, RerankError::Input(_)));
    }

    #[test]
    fn rerank_empty_docs_returns_empty() {
        let r = Bm25Reranker::new();
        let out = r.rerank("hello", &[]).expect("ok");
        assert!(out.is_empty());
    }

    #[test]
    fn rerank_ranks_relevant_doc_above_irrelevant() {
        let r = Bm25Reranker::new();
        let docs = vec![
            Doc::new("d1", "the cat sat on the mat"),
            Doc::new("d2", "completely unrelated text about dogs"),
            Doc::new("d3", "the cat ate the cat food"),
        ];
        let scored = r.rerank("cat", &docs).expect("ok");
        assert_eq!(scored.len(), 3);
        // d3 has 2 cat occurrences vs d1's 1; d2 has zero.
        let d2 = scored.iter().find(|s| s.doc_id == "d2").unwrap();
        let d1 = scored.iter().find(|s| s.doc_id == "d1").unwrap();
        let d3 = scored.iter().find(|s| s.doc_id == "d3").unwrap();
        assert!(d3.score >= d1.score, "d3={} d1={}", d3.score, d1.score);
        assert!(d1.score > d2.score, "d1={} d2={}", d1.score, d2.score);
        // The completely irrelevant doc should score zero.
        assert!(d2.score.abs() < 1e-6);
    }

    #[test]
    fn rerank_score_is_nonnegative() {
        let r = Bm25Reranker::new();
        let docs = vec![
            Doc::new("a", "alpha beta gamma"),
            Doc::new("b", "delta epsilon"),
            Doc::new("c", "alpha alpha alpha"),
        ];
        let scored = r.rerank("alpha beta", &docs).expect("ok");
        for s in &scored {
            assert!(s.score >= 0.0, "negative score: {}", s.score);
        }
    }

    #[test]
    fn typical_joules_per_doc_is_microjoule_class() {
        let r = Bm25Reranker::new();
        assert!(r.typical_joules_per_doc() < 1e-3);
        assert!(r.typical_joules_per_doc() > 0.0);
    }

    #[test]
    fn with_params_overrides_defaults() {
        let r = Bm25Reranker::with_params(Bm25Params {
            k1: 2.0,
            b: 0.5,
            joules_per_doc: 1e-3,
        });
        assert!((r.params().k1 - 2.0).abs() < 1e-6);
        assert!((r.params().b - 0.5).abs() < 1e-6);
        assert!((r.typical_joules_per_doc() - 1e-3).abs() < 1e-12);
    }

    #[test]
    fn deterministic_repeat_runs_match() {
        let r = Bm25Reranker::new();
        let docs = vec![
            Doc::new("a", "foo bar baz"),
            Doc::new("b", "bar bar quux"),
            Doc::new("c", "foo foo foo"),
        ];
        let first = r.rerank("foo bar", &docs).expect("ok");
        let second = r.rerank("foo bar", &docs).expect("ok");
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.doc_id, b.doc_id);
            assert!((a.score - b.score).abs() < 1e-6);
        }
    }
}
