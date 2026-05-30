//! The end-to-end recall composition: BM25-cheap top-N (from
//! [`jouleclaw_memory`]) → consumer-supplied [`Reranker`] reorders → top-K.
//!
//! The pattern every production memory framework converges on
//! (Hindsight, Mem0, Zep, …) is: cheap lexical recall pulls a wide
//! candidate set; a more expensive reranker (cross-encoder, ColBERT,
//! SPLADE, embedding-cosine) reorders the small set; the top-K survives.
//! The lexical stage is L1-cheap by construction; the reranker stage is
//! L2.5-cheap because it touches only N≪|store| documents.
//!
//! This crate is the bridge. It doesn't ship a reranker — [`jouleclaw-rerank`]
//! already defines the [`Reranker`] trait and ships a BM25 reference;
//! consumers plug in whatever neural backend they prefer. Here we wire
//! the two together so a [`MemoryStore`] hit becomes a reranker `Doc`,
//! the reranker's score replaces the BM25 score, and the result comes
//! back as a [`RecallHit`] with the new ordering.
//!
//! ```text
//! query → MemoryStore::recall (k=top_n) → bridge to Doc[] → Reranker::rerank
//!                                                    ↓
//!                                          stable-sort + truncate top_k
//!                                                    ↓
//!                                                RecallHit[] (reranked)
//! ```
//!
//! ## Honest scope
//!
//! No new trait; no new reranker. The [`Reranker`] trait already covers
//! every variant the field uses (cross-encoder, ColBERT, SPLADE,
//! embedding-cosine). This crate is the **composition** — small, pure,
//! deterministic given a deterministic reranker.

#![forbid(unsafe_code)]

use jouleclaw_memory::{MemoryStore, RecallHit, RecallOptions};
use jouleclaw_rerank::{Doc, RerankError, Reranker};

/// Errors from the composed recall path.
#[derive(Debug, thiserror::Error)]
pub enum RecallError {
    #[error("reranker failed: {0}")]
    Rerank(#[from] RerankError),
}

/// Options for [`recall_reranked`].
#[derive(Debug, Clone)]
pub struct RecallRerankOptions {
    /// Wide candidate set pulled from the memory store before reranking.
    /// Should be ≥ `top_k`. Default 50.
    pub top_n: usize,
    /// Final result count. Default 10.
    pub top_k: usize,
    /// Recall filter passed through to the memory store (kind, trust,
    /// temporal window, …). `k` is overridden by `top_n`; other fields
    /// pass through.
    pub recall: RecallOptions,
}

impl Default for RecallRerankOptions {
    fn default() -> Self {
        Self {
            top_n: 50,
            top_k: 10,
            recall: RecallOptions::default(),
        }
    }
}

/// Pull the top-N memory hits for `query`, hand them to `reranker`, and
/// return the top-K reordered hits. The reranker's score replaces the
/// BM25 score in the returned [`RecallHit`]; the [`RecallHit::fact`] is
/// untouched.
pub fn recall_reranked<S, R>(
    store: &S,
    reranker: &R,
    query: &str,
    opts: RecallRerankOptions,
) -> Result<Vec<RecallHit>, RecallError>
where
    S: MemoryStore + ?Sized,
    R: Reranker + ?Sized,
{
    let top_n = opts.top_n.max(opts.top_k).max(1);
    let mut recall_opts = opts.recall;
    recall_opts.k = Some(top_n);
    let hits = store.recall(query, recall_opts);
    if hits.is_empty() {
        return Ok(Vec::new());
    }
    // Bridge to the reranker's Doc shape. The fact id is content-addressed,
    // so it doubles cleanly as the reranker doc_id; the fact text is what
    // the reranker scores.
    let docs: Vec<Doc> = hits
        .iter()
        .map(|h| Doc::new(h.fact.id.clone(), h.fact.text.clone()))
        .collect();
    let scores = reranker.rerank(query, &docs)?;
    // Map scores back onto the original hits by id. A reranker that
    // omits an id is treated as score = -infinity (excluded).
    let by_id: std::collections::HashMap<&str, f32> = scores
        .iter()
        .map(|s| (s.doc_id.as_str(), s.score))
        .collect();
    let mut reranked: Vec<RecallHit> = hits
        .into_iter()
        .filter_map(|h| by_id.get(h.fact.id.as_str()).map(|s| RecallHit { fact: h.fact, score: *s }))
        .collect();
    // Deterministic ordering: score desc, then content-address asc.
    reranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.fact.id.cmp(&b.fact.id))
    });
    reranked.truncate(opts.top_k);
    Ok(reranked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_memory::{CaptureOptions, InMemoryStore, MemoryStore, MemoryType};
    use jouleclaw_rerank::{Bm25Reranker, RerankScore};

    /// A reranker that inverts the natural BM25 ordering — useful for
    /// proving the composed result reflects the reranker's order, not the
    /// store's.
    struct InvertedLengthReranker;
    impl Reranker for InvertedLengthReranker {
        fn name(&self) -> &str {
            "inverted-length"
        }
        fn rerank(
            &self,
            _query: &str,
            docs: &[Doc],
        ) -> Result<Vec<RerankScore>, RerankError> {
            // Shortest doc wins. Pure determinism, no randomness.
            Ok(docs
                .iter()
                .map(|d| RerankScore::new(d.id.clone(), 1.0 / (d.text.len() as f32 + 1.0)))
                .collect())
        }
        fn typical_joules_per_doc(&self) -> f64 {
            1e-9
        }
    }

    /// A reranker that omits some doc ids — exercises the
    /// reranker-may-drop-docs path.
    struct DropFirstReranker;
    impl Reranker for DropFirstReranker {
        fn name(&self) -> &str {
            "drop-first"
        }
        fn rerank(
            &self,
            _query: &str,
            docs: &[Doc],
        ) -> Result<Vec<RerankScore>, RerankError> {
            Ok(docs
                .iter()
                .skip(1)
                .map(|d| RerankScore::new(d.id.clone(), 1.0))
                .collect())
        }
        fn typical_joules_per_doc(&self) -> f64 {
            0.0
        }
    }

    fn store_with_three() -> InMemoryStore {
        let mut s = InMemoryStore::new();
        s.capture(
            "Sarah is considering leaving consulting",
            CaptureOptions {
                kind: Some(MemoryType::Episodic),
                ..Default::default()
            },
            1,
        );
        s.capture(
            "Sarah said she has been unhappy since the reorg",
            CaptureOptions {
                kind: Some(MemoryType::Episodic),
                ..Default::default()
            },
            2,
        );
        s.capture(
            "Sarah",
            CaptureOptions {
                kind: Some(MemoryType::Episodic),
                ..Default::default()
            },
            3,
        );
        s
    }

    #[test]
    fn empty_store_returns_empty() {
        let store = InMemoryStore::new();
        let r = Bm25Reranker::default();
        let out = recall_reranked(&store, &r, "anything", Default::default()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn reranker_order_replaces_bm25_order() {
        let store = store_with_three();
        let r = InvertedLengthReranker;
        // top_n big enough to pull all three; top_k = 3 to see ordering.
        let out = recall_reranked(
            &store,
            &r,
            "Sarah",
            RecallRerankOptions {
                top_n: 10,
                top_k: 3,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(out.len(), 3);
        // Inverted-length puts the shortest doc ("Sarah") first.
        assert_eq!(out[0].fact.text, "Sarah");
        // Scores monotonically decrease — the composed sort is correct.
        for w in out.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn top_k_truncates_after_rerank() {
        let store = store_with_three();
        let r = InvertedLengthReranker;
        let out = recall_reranked(
            &store,
            &r,
            "Sarah",
            RecallRerankOptions {
                top_n: 10,
                top_k: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].fact.text, "Sarah");
    }

    #[test]
    fn reranker_dropping_a_doc_excludes_it_from_results() {
        let store = store_with_three();
        let r = DropFirstReranker;
        let out = recall_reranked(
            &store,
            &r,
            "Sarah",
            RecallRerankOptions {
                top_n: 10,
                top_k: 10,
                ..Default::default()
            },
        )
        .unwrap();
        // DropFirstReranker emits N-1 scores; the bridge filters out the
        // unscored doc.
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn bm25_reranker_passes_through_with_consistent_ordering() {
        let store = store_with_three();
        let r = Bm25Reranker::default();
        let out = recall_reranked(
            &store,
            &r,
            "Sarah leaving reorg",
            RecallRerankOptions {
                top_n: 10,
                top_k: 3,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!out.is_empty());
        for w in out.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn passes_through_kinds_filter_to_memory() {
        let mut store = InMemoryStore::new();
        store.capture(
            "episodic note",
            CaptureOptions {
                kind: Some(MemoryType::Episodic),
                ..Default::default()
            },
            1,
        );
        store.capture(
            "semantic fact",
            CaptureOptions {
                kind: Some(MemoryType::Semantic),
                ..Default::default()
            },
            2,
        );
        let r = Bm25Reranker::default();
        let out = recall_reranked(
            &store,
            &r,
            "note fact",
            RecallRerankOptions {
                recall: RecallOptions {
                    kinds: vec![MemoryType::Semantic],
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].fact.kind, MemoryType::Semantic);
    }
}
