//! [`jouleclaw_cascade::Tier`] implementation for the L0.75 SSM router.
//!
//! The tier owns an [`IntentClassifier`] (the deterministic
//! [`KeywordClassifier`] by default) and emits a JSON [`RouteHint`] on
//! every text query. It never refuses on a successful classification —
//! the route is the answer — but it surfaces [`AnswerOutput::Refused`]
//! for non-text inputs so the cascade keeps walking.

use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, Query, QueryInput, RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;
use serde_json::json;

use crate::classifier::{IntentClassifier, KeywordClassifier, RouteHint};

/// Flat dispatch energy. ~100 µJ — class-typical for a small local SSM
/// inference pass (Mamba-3 64M params at int8). The default
/// [`KeywordClassifier`] is cheaper still, but we report the SSM-class
/// figure so production swaps (Liquid / Mamba backends) stay honest to
/// the cascade's calibration ledger.
pub const SSM_ROUTER_JOULES: f64 = 100e-6;

/// Flat dispatch latency. ~100 µs.
pub const SSM_ROUTER_LATENCY: Duration = Duration::from_micros(100);

/// Confidence floor reported by `estimate_cost`. The router classifies
/// at >0.8 confidence on the canonical conformance vectors; the floor is
/// the lower-bound the runtime can rely on for quality-floor gating.
pub const SSM_ROUTER_CONFIDENCE_FLOOR: f32 = 0.8;

/// Errors surfaced by the SSM-router tier. JSON serialisation of the
/// route hint is the only failure mode — kept as a typed error so
/// downstream consumers can wrap it.
#[derive(Debug, thiserror::Error)]
pub enum SsmRouterError {
    /// JSON serialisation of the route hint failed.
    #[error("route hint serialization failed: {0}")]
    SerializeFailed(String),
}

/// The L0.75 SSM-router tier.
///
/// Construct with [`SsmRouterTier::new`] for the deterministic
/// [`KeywordClassifier`] default, or [`SsmRouterTier::with_classifier`]
/// to plug in a Liquid / Mamba / Hyena backend.
pub struct SsmRouterTier {
    classifier: Box<dyn IntentClassifier>,
}

impl SsmRouterTier {
    /// Construct a tier with the deterministic v0.1 [`KeywordClassifier`].
    pub fn new() -> Self {
        Self {
            classifier: Box::new(KeywordClassifier::new()),
        }
    }

    /// Construct a tier with a caller-supplied [`IntentClassifier`].
    pub fn with_classifier(classifier: Box<dyn IntentClassifier>) -> Self {
        Self { classifier }
    }

    /// Borrow the classifier (for diagnostics — e.g. `name()`).
    pub fn classifier(&self) -> &dyn IntentClassifier {
        &*self.classifier
    }
}

impl Default for SsmRouterTier {
    fn default() -> Self {
        Self::new()
    }
}

impl Tier for SsmRouterTier {
    fn id(&self) -> TierId {
        TierId::L0_75SsmRouter
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        // Text-only. The donor falls back to keyword heuristics for
        // empty queries; we treat empty text as inapplicable so the
        // runtime moves on cleanly.
        match &q.input {
            QueryInput::Text(s) if !s.trim().is_empty() => Some(TierEstimate {
                joules: SSM_ROUTER_JOULES,
                latency: SSM_ROUTER_LATENCY,
                confidence_floor: SSM_ROUTER_CONFIDENCE_FLOOR,
            }),
            _ => None,
        }
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget_remaining: f64,
    ) -> Result<Answer, AnswerError> {
        let text = match &q.input {
            QueryInput::Text(s) if !s.trim().is_empty() => s.as_str(),
            _ => return Ok(refused(RefusalReason::Inapplicable, 0.0)),
        };

        let hint = self.classifier.classify(text);
        let bytes = serialize_hint(&hint).map_err(|e| AnswerError::TierFailed {
            tier: TierId::L0_75SsmRouter,
            cause: format!("serialize: {e}"),
        })?;

        Ok(Answer {
            output: AnswerOutput::Structured(bytes),
            tier_used: TierId::L0_75SsmRouter,
            joules_spent: SSM_ROUTER_JOULES,
            confidence: hint.confidence,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        })
    }
}

fn refused(reason: RefusalReason, joules: f64) -> Answer {
    Answer {
        output: AnswerOutput::Refused(reason),
        tier_used: TierId::L0_75SsmRouter,
        joules_spent: joules,
        confidence: 0.0,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

fn serialize_hint(hint: &RouteHint) -> Result<Vec<u8>, SsmRouterError> {
    let routed_to: Vec<&'static str> = hint.routed_to.iter().map(|t| t.wire_tag()).collect();
    let value = json!({
        "intent": hint.intent.wire_tag(),
        "routed_to": routed_to,
        "confidence": hint.confidence,
    });
    serde_json::to_vec(&value).map_err(|e| SsmRouterError::SerializeFailed(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{ContextRef, JouleBudget, QualityFloor, Query, QueryInput};

    use crate::classifier::Intent;

    fn text_query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn id_is_l0_75_ssm_router() {
        let t = SsmRouterTier::new();
        assert_eq!(t.id(), TierId::L0_75SsmRouter);
        assert_eq!(t.id().wire_tag(), "L0.75");
        assert_eq!(t.id().name(), "SsmRouter");
    }

    #[test]
    fn estimate_text_returns_some() {
        let t = SsmRouterTier::new();
        let q = text_query("what is the capital of France");
        let est = t.estimate_cost(&q).expect("text → estimate");
        assert_eq!(est.joules, SSM_ROUTER_JOULES);
        assert_eq!(est.latency, SSM_ROUTER_LATENCY);
        assert!((est.confidence_floor - SSM_ROUTER_CONFIDENCE_FLOOR).abs() < f32::EPSILON);
    }

    #[test]
    fn estimate_empty_text_returns_none() {
        let t = SsmRouterTier::new();
        let q = text_query("");
        assert!(t.estimate_cost(&q).is_none());
        let q = text_query("   ");
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_non_text_returns_none() {
        let t = SsmRouterTier::new();
        let q = Query {
            input: QueryInput::Binary(vec![1, 2, 3]),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(t.estimate_cost(&q).is_none());

        let q = Query {
            input: QueryInput::Image(vec![0xff, 0xd8]),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn try_answer_emits_structured_route() {
        let mut t = SsmRouterTier::new();
        let q = text_query("what is the capital of France");
        let ans = t.try_answer(&q, 1.0).expect("text classifies");
        match ans.output {
            AnswerOutput::Structured(bytes) => {
                let v: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("structured JSON");
                assert_eq!(v["intent"], "factual");
                let routed = v["routed_to"].as_array().expect("routed_to array");
                assert!(routed.iter().any(|s| s == "L0.1"));
            }
            other => panic!("expected Structured, got {other:?}"),
        }
        assert_eq!(ans.tier_used, TierId::L0_75SsmRouter);
        assert_eq!(ans.joules_spent, SSM_ROUTER_JOULES);
        assert!(ans.confidence >= SSM_ROUTER_CONFIDENCE_FLOOR);
    }

    #[test]
    fn try_answer_tool_query_routes_to_tool_tier() {
        let mut t = SsmRouterTier::new();
        let q = text_query("convert 5 miles to km");
        let ans = t.try_answer(&q, 1.0).expect("classifies");
        match ans.output {
            AnswerOutput::Structured(bytes) => {
                let v: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("json");
                assert_eq!(v["intent"], "computation");
                let routed = v["routed_to"].as_array().expect("array");
                assert!(routed.iter().any(|s| s == "L0.5"));
            }
            other => panic!("expected Structured, got {other:?}"),
        }
    }

    #[test]
    fn try_answer_greeting_routes_conversational() {
        let mut t = SsmRouterTier::new();
        let q = text_query("hello");
        let ans = t.try_answer(&q, 1.0).expect("classifies");
        match ans.output {
            AnswerOutput::Structured(bytes) => {
                let v: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("json");
                assert_eq!(v["intent"], "conversational");
            }
            other => panic!("expected Structured, got {other:?}"),
        }
        assert!(ans.confidence > 0.9);
    }

    #[test]
    fn try_answer_non_text_refuses() {
        let mut t = SsmRouterTier::new();
        let q = Query {
            input: QueryInput::Binary(vec![1, 2, 3]),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let ans = t.try_answer(&q, 1.0).expect("refuses cleanly");
        assert!(matches!(
            ans.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
        assert_eq!(ans.confidence, 0.0);
        assert_eq!(ans.tier_used, TierId::L0_75SsmRouter);
    }

    #[test]
    fn try_answer_empty_text_refuses() {
        let mut t = SsmRouterTier::new();
        let q = text_query("   ");
        let ans = t.try_answer(&q, 1.0).expect("refuses");
        assert!(matches!(
            ans.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn end_to_end_via_cascade_runtime() {
        use jouleclaw_cascade::tier::{Cascade, Runtime};

        let mut cascade = Cascade::new();
        cascade.register(Box::new(SsmRouterTier::new()));
        let mut rt = Runtime::new_without_l0(cascade);

        let q = text_query("what is the capital of France");
        let ans = rt.answer(q).expect("runtime answer");
        assert_eq!(ans.tier_used, TierId::L0_75SsmRouter);
        match ans.output {
            AnswerOutput::Structured(bytes) => {
                let v: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("json");
                assert_eq!(v["intent"], "factual");
            }
            other => panic!("expected Structured, got {other:?}"),
        }
    }

    // ── Custom-classifier swap ──────────────────────────────────────

    /// Test-only deterministic classifier that always says "reasoning".
    struct AlwaysReasoning;
    impl IntentClassifier for AlwaysReasoning {
        fn classify(&self, _text: &str) -> RouteHint {
            RouteHint {
                intent: Intent::Reasoning,
                routed_to: Intent::Reasoning.routed_to(),
                confidence: 0.99,
            }
        }
        fn name(&self) -> &'static str {
            "always-reasoning"
        }
    }

    #[test]
    fn with_classifier_swaps_backend() {
        let mut t = SsmRouterTier::with_classifier(Box::new(AlwaysReasoning));
        assert_eq!(t.classifier().name(), "always-reasoning");
        let q = text_query("hello");
        let ans = t.try_answer(&q, 1.0).expect("classifies");
        match ans.output {
            AnswerOutput::Structured(bytes) => {
                let v: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("json");
                // Even on a greeting, the swapped classifier overrides.
                assert_eq!(v["intent"], "reasoning");
            }
            other => panic!("expected Structured, got {other:?}"),
        }
        assert!((ans.confidence - 0.99).abs() < 1e-6);
    }

    #[test]
    fn classifier_default_is_keyword_v0() {
        let t = SsmRouterTier::new();
        assert_eq!(t.classifier().name(), "keyword-v0");
    }
}
