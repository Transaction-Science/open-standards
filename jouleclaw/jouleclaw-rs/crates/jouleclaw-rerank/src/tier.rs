//! L2.5 — neural reranking tier adapter.
//!
//! [`RerankTier`] is generic over a [`Reranker`] implementation: any
//! consumer-supplied backend plugs in via the trait. The tier handles
//! the cascade-side concerns:
//!
//! - parsing the [`QueryInput::Structured`] envelope,
//! - sorting and truncating the reranker's scores to `top_k`,
//! - reporting honest joule spend via the reranker's self-report,
//! - shaping the output as `AnswerOutput::Structured`.

use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, Query, QueryInput,
    RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;
use jouleclaw_energy::Provenance;
use serde::{Deserialize, Serialize};

use crate::reranker::{Doc, RerankScore, Reranker};

// ─── Cost model ──────────────────────────────────────────────────

/// Donor envelope: ~500 µJ per rerank invocation on GPU. The actual
/// per-query cost is computed from `doc_count * reranker.typical_joules_per_doc()`
/// at dispatch time; this constant is the *budget envelope* the tier
/// advertises in [`Tier::estimate_cost`].
pub const RERANK_JOULES: f64 = 500e-6;
/// Wall-clock latency envelope advertised in [`TierEstimate`].
pub const RERANK_LATENCY: Duration = Duration::from_millis(100);
/// Confidence floor advertised to the runtime. Mirrors the donor's
/// 0.7 baseline.
pub const RERANK_CONFIDENCE_FLOOR: f32 = 0.7;
/// Default `top_k` when the envelope omits it.
pub const DEFAULT_TOP_K: usize = 10;
/// Hard ceiling on docs accepted from the envelope. Prevents a
/// malicious or buggy caller from forcing a multi-thousand-doc rerank.
pub const MAX_DOCS_IN: usize = 1_000;

// ─── Envelope + output ───────────────────────────────────────────

/// Wire envelope consumed by the L2.5 tier through
/// [`QueryInput::Structured`]. JSON example:
///
/// ```json
/// {
///   "query": "what is the capital of france",
///   "docs": [
///     { "id": "doc-1", "text": "Paris is the capital of France." },
///     { "id": "doc-2", "text": "Berlin is the capital of Germany." }
///   ],
///   "top_k": 10
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankEnvelope {
    /// Free-text query the reranker scores against.
    pub query: String,
    /// Federated candidate set.
    pub docs: Vec<Doc>,
    /// Truncate the output to this many top results. Defaults to
    /// [`DEFAULT_TOP_K`] when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
}

/// Structured response shape returned by the L2.5 tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankOutput {
    /// Reranked scores in descending order, truncated to `top_k`.
    pub reranked: Vec<RerankScore>,
    /// Reranker self-reported name (e.g. `"bm25"`, `"colbert-v2"`).
    pub reranker: String,
}

// ─── Errors ──────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
enum RerankTierError {
    #[error("invalid rerank envelope: {0}")]
    BadEnvelope(String),
    #[error("reranker failed: {0}")]
    Reranker(#[from] crate::reranker::RerankError),
    #[error("output serialization failed: {0}")]
    Serialise(#[from] serde_json::Error),
}

// ─── Tier impl ───────────────────────────────────────────────────

/// L2.5 NeuralRerank tier. Wires a [`Reranker`] implementation into the
/// cascade.
pub struct RerankTier<R: Reranker> {
    reranker: R,
}

impl<R: Reranker> RerankTier<R> {
    /// Construct a tier wrapping `reranker`.
    pub fn new(reranker: R) -> Self {
        Self { reranker }
    }

    /// Borrow the underlying reranker.
    pub fn reranker(&self) -> &R {
        &self.reranker
    }

    /// Provenance tag for this tier's energy spend.
    ///
    /// The L2.5 envelope is derived from the reranker's self-report
    /// (`typical_joules_per_doc * doc_count`), not from a hardware
    /// shunt sampled by JouleClaw, so [`Provenance::Estimator`] is the
    /// honest label. Consumers whose backend exposes a real shunt
    /// (NVML, IOReport) SHOULD surface that separately through the
    /// `jouleclaw-prov` receipt path.
    pub const fn provenance() -> Provenance {
        Provenance::Estimator
    }

    /// Decode the canonical envelope from a `Structured` query payload.
    fn parse_envelope(bytes: &[u8]) -> Result<RerankEnvelope, RerankTierError> {
        let env: RerankEnvelope = serde_json::from_slice(bytes)
            .map_err(|e| RerankTierError::BadEnvelope(e.to_string()))?;
        if env.query.is_empty() {
            return Err(RerankTierError::BadEnvelope(
                "query is empty".into(),
            ));
        }
        if env.docs.is_empty() {
            return Err(RerankTierError::BadEnvelope("docs is empty".into()));
        }
        if env.docs.len() > MAX_DOCS_IN {
            return Err(RerankTierError::BadEnvelope(format!(
                "too many docs: {} > {}",
                env.docs.len(),
                MAX_DOCS_IN,
            )));
        }
        Ok(env)
    }

    /// End-to-end pipeline: parse → rerank → sort → truncate → wrap.
    fn dispatch(&self, bytes: &[u8]) -> Result<Answer, RerankTierError> {
        let envelope = Self::parse_envelope(bytes)?;
        let top_k = envelope.top_k.unwrap_or(DEFAULT_TOP_K).max(1);
        let doc_count = envelope.docs.len();

        let mut scored = self
            .reranker
            .rerank(&envelope.query, &envelope.docs)?;
        // Defensive sort regardless of what the reranker returned.
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);

        let confidence = normalise_top_confidence(&scored);
        let joules_spent =
            (doc_count as f64) * self.reranker.typical_joules_per_doc();

        let payload = RerankOutput {
            reranked: scored,
            reranker: self.reranker.name().to_string(),
        };
        let bytes = serde_json::to_vec(&payload)?;

        Ok(Answer {
            output: AnswerOutput::Structured(bytes),
            tier_used: TierId::L2_5NeuralRerank,
            joules_spent,
            confidence,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        })
    }
}

// ─── Confidence ──────────────────────────────────────────────────

/// Map the top reranked score into a `[0, 1]` confidence.
///
/// BM25 / ColBERT / SPLADE all produce open-scale relevance scores
/// rather than probabilities. We map "top score >= 1.0" to 0.9
/// confidence (saturating below 1.0 so the runtime never treats a
/// reranker as oracle-certain), scaling linearly below.
fn normalise_top_confidence(scored: &[RerankScore]) -> f32 {
    let Some(top) = scored.first() else {
        return 0.0;
    };
    if !top.score.is_finite() || top.score <= 0.0 {
        return 0.0;
    }
    let clamped = (top.score / 1.0).clamp(0.0, 1.0);
    // Floor at the advertised tier floor when we have a positive score
    // — the reranker did find *something*. Cap at 0.9 to never
    // pretend to be oracle-certain.
    let floor = RERANK_CONFIDENCE_FLOOR;
    (floor + (0.9 - floor) * clamped).clamp(0.0, 0.9)
}

// ─── Answer helpers ──────────────────────────────────────────────

fn refused_inapplicable(joules: f64) -> Answer {
    Answer {
        output: AnswerOutput::Refused(RefusalReason::Inapplicable),
        tier_used: TierId::L2_5NeuralRerank,
        joules_spent: joules,
        confidence: 0.0,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

// ─── Tier trait wiring ───────────────────────────────────────────

impl<R: Reranker + 'static> Tier for RerankTier<R> {
    fn id(&self) -> TierId {
        TierId::L2_5NeuralRerank
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        // Only structured envelopes are accepted.
        match &q.input {
            QueryInput::Structured(_) => Some(TierEstimate {
                joules: RERANK_JOULES,
                latency: RERANK_LATENCY,
                confidence_floor: RERANK_CONFIDENCE_FLOOR,
            }),
            _ => None,
        }
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget_remaining: f64,
    ) -> Result<Answer, AnswerError> {
        let bytes = match &q.input {
            QueryInput::Structured(b) => b.clone(),
            _ => return Ok(refused_inapplicable(0.0)),
        };
        match self.dispatch(&bytes) {
            Ok(a) => Ok(a),
            Err(RerankTierError::BadEnvelope(_)) => {
                Ok(refused_inapplicable(0.0))
            }
            Err(e) => Err(AnswerError::TierFailed {
                tier: TierId::L2_5NeuralRerank,
                cause: e.to_string(),
            }),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::Bm25Reranker;
    use crate::reranker::RerankError;
    use jouleclaw_cascade::tier::Cascade;
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, QualityFloor, Query, QueryInput,
    };

    fn envelope(query: &str, docs: &[(&str, &str)], top_k: Option<usize>) -> Vec<u8> {
        let env = RerankEnvelope {
            query: query.to_string(),
            docs: docs
                .iter()
                .map(|(id, text)| Doc::new(*id, *text))
                .collect(),
            top_k,
        };
        serde_json::to_vec(&env).expect("encode envelope")
    }

    fn structured_query(bytes: Vec<u8>) -> Query {
        Query {
            input: QueryInput::Structured(bytes),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    // ─── Bm25 fixture for trait tests ────────────────────────────

    fn bm25_tier() -> RerankTier<Bm25Reranker> {
        RerankTier::new(Bm25Reranker::new())
    }

    // ─── Identity / cost / estimate ──────────────────────────────

    #[test]
    fn tier_id_is_l2_5() {
        let t = bm25_tier();
        assert_eq!(t.id(), TierId::L2_5NeuralRerank);
    }

    #[test]
    fn estimate_cost_for_structured_returns_envelope() {
        let t = bm25_tier();
        let q = structured_query(envelope("hello", &[("a", "hello world")], None));
        let est = t.estimate_cost(&q).expect("structured input applicable");
        assert!((est.joules - RERANK_JOULES).abs() < 1e-12);
        assert_eq!(est.latency, RERANK_LATENCY);
        assert!((est.confidence_floor - RERANK_CONFIDENCE_FLOOR).abs() < 1e-6);
    }

    #[test]
    fn estimate_cost_for_text_is_none() {
        let t = bm25_tier();
        let q = Query {
            input: QueryInput::Text("hello".into()),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_cost_for_binary_is_none() {
        let t = bm25_tier();
        let q = Query {
            input: QueryInput::Binary(vec![0, 1, 2]),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn provenance_is_estimator() {
        assert_eq!(
            RerankTier::<Bm25Reranker>::provenance(),
            Provenance::Estimator
        );
    }

    // ─── End-to-end BM25 dispatch ────────────────────────────────

    #[test]
    fn bm25_ranks_three_docs_top_one_is_most_relevant() {
        let mut t = bm25_tier();
        let bytes = envelope(
            "capital of france",
            &[
                ("d1", "Berlin is the capital of Germany."),
                ("d2", "Paris is the capital of France."),
                ("d3", "Madrid is the capital of Spain."),
            ],
            Some(1),
        );
        let q = structured_query(bytes);
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert_eq!(a.tier_used, TierId::L2_5NeuralRerank);

        let out = match a.output {
            AnswerOutput::Structured(b) => b,
            other => panic!("expected structured, got {other:?}"),
        };
        let payload: RerankOutput =
            serde_json::from_slice(&out).expect("deser");
        assert_eq!(payload.reranked.len(), 1);
        assert_eq!(payload.reranked[0].doc_id, "d2");
        assert!(payload.reranked[0].score > 0.0);
        assert_eq!(payload.reranker, "bm25");
    }

    #[test]
    fn bm25_default_top_k_truncates_to_ten() {
        let mut t = bm25_tier();
        let docs: Vec<(String, String)> = (0..15)
            .map(|i| (format!("d{i}"), format!("doc number {i} with foo")))
            .collect();
        let docs_ref: Vec<(&str, &str)> = docs
            .iter()
            .map(|(id, text)| (id.as_str(), text.as_str()))
            .collect();
        let bytes = envelope("foo", &docs_ref, None);
        let q = structured_query(bytes);
        let a = t.try_answer(&q, 1.0).expect("ok");
        let out = match a.output {
            AnswerOutput::Structured(b) => b,
            _ => panic!("expected structured"),
        };
        let payload: RerankOutput =
            serde_json::from_slice(&out).expect("deser");
        assert_eq!(payload.reranked.len(), DEFAULT_TOP_K);
    }

    #[test]
    fn bm25_joules_spent_scales_with_doc_count() {
        let mut t = bm25_tier();
        let bytes = envelope(
            "alpha",
            &[
                ("a", "alpha beta"),
                ("b", "gamma delta"),
                ("c", "alpha alpha alpha"),
            ],
            None,
        );
        let q = structured_query(bytes);
        let a = t.try_answer(&q, 1.0).expect("ok");
        // 3 docs * 10 µJ = 30 µJ. Allow some float wiggle.
        let expected = 3.0 * 10e-6;
        assert!(
            (a.joules_spent - expected).abs() < 1e-9,
            "joules_spent={} expected={}",
            a.joules_spent,
            expected,
        );
    }

    #[test]
    fn empty_docs_envelope_refuses() {
        let mut t = bm25_tier();
        let bytes = envelope("hello", &[], None);
        let q = structured_query(bytes);
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn malformed_envelope_refuses() {
        let mut t = bm25_tier();
        let q = structured_query(b"not json at all".to_vec());
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn non_structured_input_refuses() {
        let mut t = bm25_tier();
        let q = Query {
            input: QueryInput::Text("hi".into()),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn confidence_is_bounded_between_zero_and_oracle_cap() {
        let mut t = bm25_tier();
        let bytes = envelope("alpha", &[("a", "alpha beta")], None);
        let q = structured_query(bytes);
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(a.confidence >= 0.0);
        // Never report oracle-certain confidence — we cap at 0.9.
        assert!(a.confidence <= 0.9 + 1e-6);
    }

    #[test]
    fn registers_in_a_cascade() {
        let mut c = Cascade::new();
        c.register(Box::new(bm25_tier()));
        assert!(c.tier_ids().contains(&TierId::L2_5NeuralRerank));
    }

    #[test]
    fn output_serialises_round_trip() {
        let out = RerankOutput {
            reranked: vec![
                RerankScore::new("a", 1.5),
                RerankScore::new("b", 0.5).with_explanation("matched: 1 term"),
            ],
            reranker: "bm25".into(),
        };
        let bytes = serde_json::to_vec(&out).expect("ser");
        let back: RerankOutput =
            serde_json::from_slice(&bytes).expect("deser");
        assert_eq!(back.reranked.len(), 2);
        assert_eq!(back.reranked[0].doc_id, "a");
        assert_eq!(back.reranker, "bm25");
        assert_eq!(
            back.reranked[1].explanation.as_deref(),
            Some("matched: 1 term"),
        );
    }

    #[test]
    fn confidence_zero_for_empty_reranked() {
        assert!(normalise_top_confidence(&[]).abs() < 1e-6);
    }

    #[test]
    fn confidence_zero_for_nonpositive_top_score() {
        let s = vec![RerankScore::new("a", 0.0)];
        assert!(normalise_top_confidence(&s).abs() < 1e-6);
        let s = vec![RerankScore::new("a", -0.5)];
        assert!(normalise_top_confidence(&s).abs() < 1e-6);
    }

    // ─── Custom Reranker fixture ─────────────────────────────────

    /// A trivial reranker that returns scores equal to the length of
    /// the doc text. Used to exercise the tier's plumbing in isolation
    /// from BM25 details.
    struct LengthReranker {
        joules_per_doc: f64,
    }

    impl Reranker for LengthReranker {
        fn name(&self) -> &str {
            "length"
        }
        fn rerank(
            &self,
            _query: &str,
            docs: &[Doc],
        ) -> Result<Vec<RerankScore>, RerankError> {
            Ok(docs
                .iter()
                .map(|d| RerankScore::new(&d.id, d.text.len() as f32))
                .collect())
        }
        fn typical_joules_per_doc(&self) -> f64 {
            self.joules_per_doc
        }
    }

    #[test]
    fn custom_reranker_plugs_into_tier() {
        let mut t = RerankTier::new(LengthReranker {
            joules_per_doc: 1e-4,
        });
        let bytes = envelope(
            "anything",
            &[
                ("short", "abc"),
                ("longer", "abcdefghij"),
                ("middle", "abcdef"),
            ],
            Some(2),
        );
        let q = structured_query(bytes);
        let a = t.try_answer(&q, 1.0).expect("ok");
        let out = match a.output {
            AnswerOutput::Structured(b) => b,
            _ => panic!("expected structured"),
        };
        let payload: RerankOutput =
            serde_json::from_slice(&out).expect("deser");
        assert_eq!(payload.reranked.len(), 2);
        // Length-reranker returns descending by text length.
        assert_eq!(payload.reranked[0].doc_id, "longer");
        assert_eq!(payload.reranked[1].doc_id, "middle");
        assert_eq!(payload.reranker, "length");
        // 3 docs * 100 µJ.
        assert!((a.joules_spent - 3.0 * 1e-4).abs() < 1e-9);
    }

    /// A reranker that always errors. Tier should bubble as TierFailed.
    struct FailingReranker;

    impl Reranker for FailingReranker {
        fn name(&self) -> &str {
            "fail"
        }
        fn rerank(
            &self,
            _query: &str,
            _docs: &[Doc],
        ) -> Result<Vec<RerankScore>, RerankError> {
            Err(RerankError::Backend("simulated".into()))
        }
        fn typical_joules_per_doc(&self) -> f64 {
            0.0
        }
    }

    #[test]
    fn backend_failure_bubbles_to_tier_failed() {
        let mut t = RerankTier::new(FailingReranker);
        let bytes = envelope("q", &[("a", "x")], None);
        let q = structured_query(bytes);
        let err = t.try_answer(&q, 1.0).expect_err("must fail");
        assert!(matches!(
            err,
            AnswerError::TierFailed {
                tier: TierId::L2_5NeuralRerank,
                ..
            }
        ));
    }

    #[test]
    fn too_many_docs_refuses() {
        let mut t = bm25_tier();
        // Construct an envelope just over the cap.
        let docs: Vec<(String, String)> = (0..(MAX_DOCS_IN + 1))
            .map(|i| (format!("d{i}"), format!("text {i}")))
            .collect();
        let docs_ref: Vec<(&str, &str)> = docs
            .iter()
            .map(|(id, text)| (id.as_str(), text.as_str()))
            .collect();
        let bytes = envelope("hi", &docs_ref, None);
        let q = structured_query(bytes);
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }
}
