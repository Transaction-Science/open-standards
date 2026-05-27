//! L1 — local-index tier adapter.
//!
//! [`LocalIndexTier`] is generic over a [`LocalIndex`] implementation:
//! any consumer-supplied backend plugs in via the trait. The tier
//! handles the cascade-side concerns:
//!
//! - reading a [`QueryInput::Text`] payload,
//! - dispatching to the index's `search`,
//! - sorting and truncating the hits to `k`,
//! - reporting the L1 envelope joule spend,
//! - shaping the output as `AnswerOutput::Structured(json)`.

use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, L1Primitive, Query,
    QueryInput, RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;
use jouleclaw_energy::Provenance;
use serde::{Deserialize, Serialize};

use crate::index::{IndexError, IndexHit, LocalIndex};

// ─── Cost model ──────────────────────────────────────────────────

/// Donor envelope: ~890 µJ per local-index search. Inherited from
/// `verity-cascade::layers::l1_index`'s measured tantivy reference
/// path. Client-energy-only — no wire spend.
pub const LOCAL_INDEX_JOULES: f64 = 890e-6;
/// Wall-clock latency envelope advertised in [`TierEstimate`].
pub const LOCAL_INDEX_LATENCY: Duration = Duration::from_millis(5);
/// Confidence floor advertised to the runtime. Mirrors the donor's
/// 0.6 baseline — a local-index hit is suggestive, not oracle-certain.
pub const LOCAL_INDEX_CONFIDENCE_FLOOR: f32 = 0.6;
/// Default `k` when the call site does not specify one.
pub const DEFAULT_K: usize = 10;
/// Hard ceiling on hits returned. Prevents a malicious or buggy
/// caller from forcing a runaway page.
pub const MAX_K: usize = 1_000;

// ─── Output shape ────────────────────────────────────────────────

/// Structured response shape returned by the L1 tier. Wire JSON:
///
/// ```json
/// {
///   "hits": [
///     { "doc_id": "doc-1", "text": "…", "score": 4.213 }
///   ],
///   "k": 10
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalIndexOutput {
    /// Top hits in descending score order, truncated to at most `k`.
    pub hits: Vec<IndexHit>,
    /// The `k` requested at dispatch time. Echoed for caller-side
    /// diagnostics.
    pub k: usize,
}

// ─── Errors ──────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
enum LocalIndexTierError {
    #[error("local index failed: {0}")]
    Index(#[from] IndexError),
    #[error("output serialization failed: {0}")]
    Serialise(#[from] serde_json::Error),
}

// ─── Tier impl ───────────────────────────────────────────────────

/// L1 LocalIndex tier. Wires a [`LocalIndex`] implementation into the
/// cascade.
pub struct LocalIndexTier<I: LocalIndex> {
    index: I,
    /// `k` to request from the index when the call site does not
    /// override it through a per-call mechanism.
    default_k: usize,
}

impl<I: LocalIndex> LocalIndexTier<I> {
    /// Construct a tier wrapping `index`, defaulting `k` to
    /// [`DEFAULT_K`].
    pub fn new(index: I) -> Self {
        Self {
            index,
            default_k: DEFAULT_K,
        }
    }

    /// Construct a tier with a custom default `k`. Values above
    /// [`MAX_K`] are clamped.
    pub fn with_default_k(mut self, k: usize) -> Self {
        self.default_k = k.clamp(1, MAX_K);
        self
    }

    /// The `k` this tier will request unless overridden.
    pub fn default_k(&self) -> usize {
        self.default_k
    }

    /// Borrow the underlying index.
    pub fn index(&self) -> &I {
        &self.index
    }

    /// Provenance tag for this tier's energy spend.
    ///
    /// The L1 envelope is a JouleClaw static cost model — the donor
    /// measured ~890 µJ on its tantivy reference path but no hardware
    /// shunt is sampled here, so [`Provenance::Estimator`] is the
    /// honest label. Consumers whose backend exposes a real shunt
    /// (NVML, IOReport) SHOULD surface that separately through the
    /// `jouleclaw-prov` receipt path.
    pub const fn provenance() -> Provenance {
        Provenance::Estimator
    }

    /// End-to-end pipeline: search → sort → truncate → wrap.
    fn dispatch(
        &self,
        query: &str,
    ) -> Result<Answer, LocalIndexTierError> {
        let k = self.default_k;
        let mut hits = self.index.search(query, k)?;
        // Defensive sort — the trait demands descending order, but the
        // tier must not depend on caller correctness.
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if k < hits.len() {
            hits.truncate(k);
        }

        // No hits → refuse `Inapplicable`. The cascade walker moves on
        // to L2 federation / L3 model.
        if hits.is_empty() {
            return Ok(refused_inapplicable(LOCAL_INDEX_JOULES));
        }

        let confidence = normalise_top_confidence(&hits);
        let payload = LocalIndexOutput { hits, k };
        let bytes = serde_json::to_vec(&payload)?;

        Ok(Answer {
            output: AnswerOutput::Structured(bytes),
            tier_used: TierId::L1(L1Primitive::Retrieve),
            joules_spent: LOCAL_INDEX_JOULES,
            confidence,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        })
    }
}

// ─── Confidence ──────────────────────────────────────────────────

/// Map the top hit's score into a `[0, 1]` confidence.
///
/// BM25 / tantivy scores are open-scale relevance numbers rather than
/// probabilities. We map "top score >= 5.0" to 0.9 confidence (a strong
/// match), scaling linearly below, with a floor at the advertised
/// tier floor of 0.6 (we found *something*) and a cap at 0.9 so the
/// runtime never treats a local-index hit as oracle-certain.
fn normalise_top_confidence(hits: &[IndexHit]) -> f32 {
    let Some(top) = hits.first() else {
        return 0.0;
    };
    if !top.score.is_finite() || top.score <= 0.0 {
        return 0.0;
    }
    let clamped = (top.score / 5.0).clamp(0.0, 1.0);
    let floor = LOCAL_INDEX_CONFIDENCE_FLOOR;
    (floor + (0.9 - floor) * clamped).clamp(0.0, 0.9)
}

// ─── Answer helpers ──────────────────────────────────────────────

fn refused_inapplicable(joules: f64) -> Answer {
    Answer {
        output: AnswerOutput::Refused(RefusalReason::Inapplicable),
        tier_used: TierId::L1(L1Primitive::Retrieve),
        joules_spent: joules,
        confidence: 0.0,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

// ─── Tier trait wiring ───────────────────────────────────────────

impl<I: LocalIndex + 'static> Tier for LocalIndexTier<I> {
    fn id(&self) -> TierId {
        TierId::L1(L1Primitive::Retrieve)
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        // Only text queries against a non-empty corpus are applicable.
        match &q.input {
            QueryInput::Text(s) if !s.is_empty() && self.index.doc_count() > 0 => {
                Some(TierEstimate {
                    joules: LOCAL_INDEX_JOULES,
                    latency: LOCAL_INDEX_LATENCY,
                    confidence_floor: LOCAL_INDEX_CONFIDENCE_FLOOR,
                })
            }
            _ => None,
        }
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget_remaining: f64,
    ) -> Result<Answer, AnswerError> {
        let query = match &q.input {
            QueryInput::Text(s) if !s.is_empty() => s.clone(),
            _ => return Ok(refused_inapplicable(0.0)),
        };

        if self.index.doc_count() == 0 {
            return Ok(refused_inapplicable(0.0));
        }

        match self.dispatch(&query) {
            Ok(a) => Ok(a),
            // An IndexError::Input is treated as "this query is not
            // applicable to this index" — refuse rather than failing
            // the cascade. Backend errors are honest tier failures.
            Err(LocalIndexTierError::Index(IndexError::Input(_))) => {
                Ok(refused_inapplicable(0.0))
            }
            Err(e) => Err(AnswerError::TierFailed {
                tier: TierId::L1(L1Primitive::Retrieve),
                cause: e.to_string(),
            }),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{IndexError, IndexHit, LocalIndex};
    use crate::inmem::{Document, InMemoryIndex};
    use jouleclaw_cascade::tier::Cascade;
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, QualityFloor, Query, QueryInput,
    };

    fn five_doc_index() -> InMemoryIndex {
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

    fn text_query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.into()),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn binary_query(b: Vec<u8>) -> Query {
        Query {
            input: QueryInput::Binary(b),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn structured_query(b: Vec<u8>) -> Query {
        Query {
            input: QueryInput::Structured(b),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn tier_from_corpus() -> LocalIndexTier<InMemoryIndex> {
        LocalIndexTier::new(five_doc_index())
    }

    // ─── Identity / cost / estimate ──────────────────────────────

    #[test]
    fn tier_id_is_l1_retrieve() {
        let t = tier_from_corpus();
        assert_eq!(t.id(), TierId::L1(L1Primitive::Retrieve));
    }

    #[test]
    fn estimate_cost_for_text_query_returns_envelope() {
        let t = tier_from_corpus();
        let q = text_query("capital France");
        let est = t.estimate_cost(&q).expect("applicable");
        assert!((est.joules - LOCAL_INDEX_JOULES).abs() < 1e-12);
        assert_eq!(est.latency, LOCAL_INDEX_LATENCY);
        assert!((est.confidence_floor - LOCAL_INDEX_CONFIDENCE_FLOOR).abs() < 1e-6);
    }

    #[test]
    fn estimate_cost_for_empty_corpus_is_none() {
        let t = LocalIndexTier::new(InMemoryIndex::new());
        let q = text_query("anything");
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_cost_for_empty_text_is_none() {
        let t = tier_from_corpus();
        let q = text_query("");
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_cost_for_binary_is_none() {
        let t = tier_from_corpus();
        let q = binary_query(vec![0, 1, 2]);
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_cost_for_structured_is_none() {
        let t = tier_from_corpus();
        let q = structured_query(b"{}".to_vec());
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn provenance_is_estimator() {
        assert_eq!(
            LocalIndexTier::<InMemoryIndex>::provenance(),
            Provenance::Estimator,
        );
    }

    #[test]
    fn default_k_matches_constant() {
        let t = tier_from_corpus();
        assert_eq!(t.default_k(), DEFAULT_K);
    }

    #[test]
    fn with_default_k_clamps_to_max() {
        let t = tier_from_corpus().with_default_k(MAX_K + 50);
        assert_eq!(t.default_k(), MAX_K);
    }

    #[test]
    fn with_default_k_clamps_zero_to_one() {
        let t = tier_from_corpus().with_default_k(0);
        assert_eq!(t.default_k(), 1);
    }

    // ─── End-to-end dispatch (5 docs → query → top hit) ──────────

    #[test]
    fn end_to_end_five_docs_top_hit_is_relevant() {
        let mut t = tier_from_corpus();
        let q = text_query("Paris France");
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert_eq!(a.tier_used, TierId::L1(L1Primitive::Retrieve));
        assert!((a.joules_spent - LOCAL_INDEX_JOULES).abs() < 1e-9);

        let bytes = match a.output {
            AnswerOutput::Structured(b) => b,
            other => panic!("expected structured, got {other:?}"),
        };
        let payload: LocalIndexOutput =
            serde_json::from_slice(&bytes).expect("deser");
        assert!(!payload.hits.is_empty());
        assert_eq!(payload.hits[0].doc_id, "d1");
        assert_eq!(payload.k, DEFAULT_K);
    }

    #[test]
    fn end_to_end_truncates_to_default_k() {
        let mut idx = InMemoryIndex::new();
        for i in 0..(DEFAULT_K + 5) {
            idx.insert(Document::new(
                format!("d{i}"),
                format!("alpha beta gamma {i}"),
            ));
        }
        let mut t = LocalIndexTier::new(idx);
        let q = text_query("alpha");
        let a = t.try_answer(&q, 1.0).expect("ok");
        let bytes = match a.output {
            AnswerOutput::Structured(b) => b,
            _ => panic!("expected structured"),
        };
        let payload: LocalIndexOutput =
            serde_json::from_slice(&bytes).expect("deser");
        assert!(payload.hits.len() <= DEFAULT_K);
    }

    // ─── Refusal paths ───────────────────────────────────────────

    #[test]
    fn empty_corpus_refuses() {
        let mut t = LocalIndexTier::new(InMemoryIndex::new());
        let q = text_query("anything");
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn empty_text_refuses() {
        let mut t = tier_from_corpus();
        let q = text_query("");
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn non_text_input_refuses() {
        let mut t = tier_from_corpus();
        let q = binary_query(vec![1, 2, 3]);
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn no_hits_refuses() {
        // A corpus that contains nothing matching the query.
        let mut idx = InMemoryIndex::new();
        idx.insert(Document::new("a", "completely unrelated zzz"));
        let mut t = LocalIndexTier::new(idx);
        let q = text_query("xyzzy");
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    // ─── Confidence ──────────────────────────────────────────────

    #[test]
    fn confidence_bounded_between_zero_and_oracle_cap() {
        let mut t = tier_from_corpus();
        let q = text_query("Paris");
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(a.confidence >= 0.0);
        // Never report oracle-certain confidence — we cap at 0.9.
        assert!(a.confidence <= 0.9 + 1e-6);
    }

    #[test]
    fn confidence_zero_for_empty_hits() {
        assert!(normalise_top_confidence(&[]).abs() < 1e-6);
    }

    #[test]
    fn confidence_zero_for_nonpositive_top_score() {
        let s = vec![IndexHit::new("a", "x", 0.0)];
        assert!(normalise_top_confidence(&s).abs() < 1e-6);
        let s = vec![IndexHit::new("a", "x", -0.5)];
        assert!(normalise_top_confidence(&s).abs() < 1e-6);
    }

    // ─── Cascade registration ────────────────────────────────────

    #[test]
    fn registers_in_a_cascade() {
        let mut c = Cascade::new();
        c.register(Box::new(tier_from_corpus()));
        assert!(c.tier_ids().contains(&TierId::L1(L1Primitive::Retrieve)));
    }

    // ─── Output serde ────────────────────────────────────────────

    #[test]
    fn output_serialises_round_trip() {
        let out = LocalIndexOutput {
            hits: vec![
                IndexHit::new("a", "alpha", 1.5),
                IndexHit::new("b", "beta", 0.5),
            ],
            k: 5,
        };
        let bytes = serde_json::to_vec(&out).expect("ser");
        let back: LocalIndexOutput =
            serde_json::from_slice(&bytes).expect("deser");
        assert_eq!(back.hits.len(), 2);
        assert_eq!(back.hits[0].doc_id, "a");
        assert_eq!(back.k, 5);
    }

    // ─── Custom LocalIndex fixture ───────────────────────────────

    /// A trivial index that returns one canned hit per query. Used to
    /// exercise the tier in isolation from the in-memory BM25.
    struct CannedIndex {
        canned: Vec<IndexHit>,
        count: usize,
    }

    impl LocalIndex for CannedIndex {
        fn search(
            &self,
            _query: &str,
            k: usize,
        ) -> Result<Vec<IndexHit>, IndexError> {
            let mut out = self.canned.clone();
            if k < out.len() {
                out.truncate(k);
            }
            Ok(out)
        }
        fn doc_count(&self) -> usize {
            self.count
        }
    }

    #[test]
    fn custom_local_index_plugs_into_tier() {
        let idx = CannedIndex {
            canned: vec![
                IndexHit::new("only", "the canned hit", 3.0),
            ],
            count: 1,
        };
        let mut t = LocalIndexTier::new(idx);
        let q = text_query("anything");
        let a = t.try_answer(&q, 1.0).expect("ok");
        let bytes = match a.output {
            AnswerOutput::Structured(b) => b,
            _ => panic!("expected structured"),
        };
        let payload: LocalIndexOutput =
            serde_json::from_slice(&bytes).expect("deser");
        assert_eq!(payload.hits.len(), 1);
        assert_eq!(payload.hits[0].doc_id, "only");
    }

    /// A backend that always errors on `search`. Tier should bubble as
    /// `TierFailed`.
    struct FailingIndex;

    impl LocalIndex for FailingIndex {
        fn search(
            &self,
            _query: &str,
            _k: usize,
        ) -> Result<Vec<IndexHit>, IndexError> {
            Err(IndexError::Backend("simulated".into()))
        }
        fn doc_count(&self) -> usize {
            1
        }
    }

    #[test]
    fn backend_failure_bubbles_to_tier_failed() {
        let mut t = LocalIndexTier::new(FailingIndex);
        let q = text_query("hi");
        let err = t.try_answer(&q, 1.0).expect_err("must fail");
        assert!(matches!(
            err,
            AnswerError::TierFailed {
                tier: TierId::L1(L1Primitive::Retrieve),
                ..
            }
        ));
    }

    /// A backend whose `search` reports an `Input` error. The tier
    /// should treat that as an inapplicable refusal, not a tier
    /// failure.
    struct InputErrorIndex;

    impl LocalIndex for InputErrorIndex {
        fn search(
            &self,
            _query: &str,
            _k: usize,
        ) -> Result<Vec<IndexHit>, IndexError> {
            Err(IndexError::Input("doesn't apply".into()))
        }
        fn doc_count(&self) -> usize {
            1
        }
    }

    #[test]
    fn input_error_refuses_rather_than_fails() {
        let mut t = LocalIndexTier::new(InputErrorIndex);
        let q = text_query("hi");
        let a = t.try_answer(&q, 1.0).expect("input error should not fail");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }
}
