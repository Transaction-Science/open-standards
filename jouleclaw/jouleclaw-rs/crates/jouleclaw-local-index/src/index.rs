//! The [`LocalIndex`] trait — the consumer-supplied interface for
//! retrieving documents from a local corpus.
//!
//! Implementations cover the full local-retrieval taxonomy:
//!
//! - **Full-text inverted index**: tantivy, Bleve, Whoosh-class
//! - **FTS-extended SQL**: SQLite FTS5, Postgres tsvector
//! - **Embedded KV with secondary index**: sled + a hand-rolled BM25
//! - **Memory-mapped FST / suffix array**: fst crate, sucds
//! - **In-process reference**: [`crate::InMemoryIndex`] (this crate)
//!
//! The trait is intentionally narrow — `search` and `doc_count` — so
//! any backend can be wired into the L1 tier without the trait
//! churning. Energy ownership stays inside the JouleClaw cascade: the
//! tier reports a fixed-envelope [`crate::tier::LOCAL_INDEX_JOULES`]
//! per dispatch, sized for client-only retrieval. Implementations that
//! draw materially more energy (large indexes, network-attached
//! storage, GPU-accelerated scoring) SHOULD report through their own
//! receipt path and refuse to register as an L1 tier.

use serde::{Deserialize, Serialize};

// ─── Wire types ──────────────────────────────────────────────────

/// One scored document returned by [`LocalIndex::search`]. `doc_id` is
/// opaque to the tier — it round-trips untouched into the L1 output so
/// the caller can re-correlate hits with their store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexHit {
    /// Opaque caller-side identifier (URL, UUID, surrogate key, …).
    pub doc_id: String,
    /// The matching passage text. Implementations decide snippet vs.
    /// full-document policy; the tier passes it through unchanged.
    pub text: String,
    /// Relevance score. Scale is index-defined — callers should treat
    /// scores as ordinal, not as a probability.
    pub score: f32,
}

impl IndexHit {
    /// Construct a new hit.
    pub fn new(
        doc_id: impl Into<String>,
        text: impl Into<String>,
        score: f32,
    ) -> Self {
        Self {
            doc_id: doc_id.into(),
            text: text.into(),
            score,
        }
    }
}

// ─── Errors ──────────────────────────────────────────────────────

/// Errors a local index can surface back to the tier.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    /// The index received malformed input (empty query, malformed
    /// filter, oversized batch, …).
    #[error("local-index input error: {0}")]
    Input(String),
    /// The index backend failed (corrupt segment, I/O, missing
    /// schema, …).
    #[error("local-index backend error: {0}")]
    Backend(String),
}

// ─── The trait ───────────────────────────────────────────────────

/// Consumer-supplied local index. One implementation per backend.
///
/// Implementations MUST:
///
/// - Be `Send + Sync` (the tier may live in a multi-threaded runtime).
/// - Return at most `k` [`IndexHit`]s in descending-score order. The
///   tier re-sorts defensively, but well-behaved implementations save
///   the cost.
/// - Be deterministic for a fixed (corpus, query, k) — JouleClaw's
///   calibration loop assumes repeatable dispatch costs.
/// - Run inside the L1 energy envelope (~890 µJ). Backends that
///   exceed this should refuse to register as an L1 tier and surface
///   themselves at L2 instead.
pub trait LocalIndex: Send + Sync {
    /// Score the local corpus against `query`, returning up to `k`
    /// hits in descending-score order.
    fn search(&self, query: &str, k: usize) -> Result<Vec<IndexHit>, IndexError>;

    /// Number of indexed documents. Used by the tier to short-circuit
    /// `estimate_cost` when the corpus is empty.
    fn doc_count(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_hit_new_round_trips() {
        let h = IndexHit::new("d1", "hello world", 1.5);
        assert_eq!(h.doc_id, "d1");
        assert_eq!(h.text, "hello world");
        assert!((h.score - 1.5).abs() < 1e-6);
    }

    #[test]
    fn index_hit_serde_round_trip() {
        let h = IndexHit::new("d1", "hello", 0.75);
        let bytes = serde_json::to_vec(&h).expect("ser");
        let back: IndexHit =
            serde_json::from_slice(&bytes).expect("deser");
        assert_eq!(back, h);
    }

    #[test]
    fn index_error_display_includes_cause() {
        let e = IndexError::Input("empty query".into());
        assert!(e.to_string().contains("empty query"));
        let e = IndexError::Backend("segment corrupt".into());
        assert!(e.to_string().contains("segment corrupt"));
    }
}
