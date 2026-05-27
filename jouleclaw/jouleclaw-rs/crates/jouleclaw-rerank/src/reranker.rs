//! The [`Reranker`] trait — the consumer-supplied interface for
//! reordering a candidate document set by query relevance.
//!
//! Implementations cover the full taxonomy:
//!
//! - **Late-interaction**: ColBERT / ColBERTv2 (per-token max-sim)
//! - **Sparse-neural**: SPLADE / uniCOIL (sparse projection)
//! - **Cross-encoder**: MiniLM / DeBERTa / monoT5
//! - **Lexical reference**: [`crate::Bm25Reranker`] (this crate)
//!
//! The trait is intentionally narrow — `name`, `rerank`, and an energy
//! self-report — so any backend can be wired into the L2.5 tier without
//! the trait churning. Energy ownership stays with the implementation:
//! the consumer knows whether the reranker runs on CPU, on an NVIDIA
//! GPU sampled through NVML, or on an Apple GPU sampled through IOReport,
//! and reports its honest per-document spend through
//! [`Reranker::typical_joules_per_doc`].

use serde::{Deserialize, Serialize};

// ─── Wire types ──────────────────────────────────────────────────

/// A document to rerank. `id` is opaque to the tier — it round-trips
/// untouched into the output so the caller can re-correlate scores
/// with their store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Doc {
    /// Opaque caller-side identifier (URL, UUID, surrogate key, …).
    pub id: String,
    /// Document text fed to the reranker. Implementations decide their
    /// own truncation policy.
    pub text: String,
}

impl Doc {
    /// Construct a new document.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

/// One scored document in the reranked output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankScore {
    /// Echoes [`Doc::id`] from the input set.
    pub doc_id: String,
    /// Relevance score. Scale is reranker-defined — callers should
    /// treat scores as ordinal, not as a probability.
    pub score: f32,
    /// Optional human-readable explanation (e.g. matched terms,
    /// attention heatmap summary). Defaults to `None`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub explanation: Option<String>,
}

impl RerankScore {
    /// Construct a score with no explanation.
    pub fn new(doc_id: impl Into<String>, score: f32) -> Self {
        Self {
            doc_id: doc_id.into(),
            score,
            explanation: None,
        }
    }

    /// Attach an explanation string.
    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = Some(explanation.into());
        self
    }
}

// ─── Errors ──────────────────────────────────────────────────────

/// Errors a reranker can surface back to the tier.
#[derive(Debug, thiserror::Error)]
pub enum RerankError {
    /// The reranker received malformed input (empty query, malformed
    /// doc, oversized batch, …).
    #[error("reranker input error: {0}")]
    Input(String),
    /// The reranker's backend (model load, GPU init, …) failed.
    #[error("reranker backend error: {0}")]
    Backend(String),
}

// ─── The trait ───────────────────────────────────────────────────

/// Consumer-supplied reranker. One implementation per backend.
///
/// Implementations MUST:
///
/// - Be `Send + Sync` (the tier may live in a multi-threaded runtime).
/// - Return one [`RerankScore`] per input [`Doc`], preserving the
///   `doc_id`. The tier is responsible for sorting and truncating —
///   implementations SHOULD return in input order, though the tier
///   re-sorts defensively.
/// - Report an honest joule-per-doc estimate. The tier uses this to
///   propagate joule spend into the cascade calibration loop.
pub trait Reranker: Send + Sync {
    /// Human-readable name (e.g. `"colbert-v2"`, `"splade-pp"`).
    fn name(&self) -> &str;

    /// Score every `doc` against `query`. Returns one score per input.
    fn rerank(
        &self,
        query: &str,
        docs: &[Doc],
    ) -> Result<Vec<RerankScore>, RerankError>;

    /// Self-reported energy cost per document. The tier multiplies this
    /// by the input doc count to compute `joules_spent`. Implementations
    /// SHOULD report the measured average from their telemetry, not a
    /// vendor datasheet number.
    fn typical_joules_per_doc(&self) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doc_new_round_trips() {
        let d = Doc::new("id-1", "hello world");
        assert_eq!(d.id, "id-1");
        assert_eq!(d.text, "hello world");
    }

    #[test]
    fn rerank_score_new_has_no_explanation() {
        let s = RerankScore::new("d", 1.5);
        assert_eq!(s.doc_id, "d");
        assert!((s.score - 1.5).abs() < 1e-6);
        assert!(s.explanation.is_none());
    }

    #[test]
    fn rerank_score_with_explanation_attaches() {
        let s = RerankScore::new("d", 0.5).with_explanation("matched: foo");
        assert_eq!(s.explanation.as_deref(), Some("matched: foo"));
    }

    #[test]
    fn doc_serde_roundtrip() {
        let d = Doc::new("x", "y");
        let bytes = serde_json::to_vec(&d).expect("ser");
        let back: Doc = serde_json::from_slice(&bytes).expect("deser");
        assert_eq!(back, d);
    }

    #[test]
    fn rerank_score_serde_skips_none_explanation() {
        let s = RerankScore::new("d", 1.0);
        let v = serde_json::to_value(&s).expect("ser");
        // `explanation` should be omitted when None.
        assert!(v.get("explanation").is_none());
    }

    #[test]
    fn rerank_error_display() {
        let e = RerankError::Input("empty query".into());
        assert!(e.to_string().contains("empty query"));
        let e = RerankError::Backend("cuda init failed".into());
        assert!(e.to_string().contains("cuda init failed"));
    }
}
