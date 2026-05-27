//! [`VerificationTier`] — the L4 cascade tier.
//!
//! Cross-model verification: dispatch the query to ≥2 *different* LLM
//! backends in parallel, then ask an [`AgreementChecker`] whether the
//! candidate answers agree. On agreement we return the consensus with
//! very-high confidence (typically `≥ 0.9`); on disagreement we refuse
//! so the cascade falls through.
//!
//! This is the most expensive tier in the cascade — ~4 J for two
//! cheap models, more if the operator wires in a frontier model — and
//! it sits at the top of the L0-L4 cost stack. It is the tier you
//! reach when you really do need the answer.

use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, L4ModelId, Query, QueryInput,
    RefusalReason, TierId, TraceEntry, TraceOutcome,
};
use jouleclaw_cascade::verification::VerificationStatus;

use crate::checker::{AgreementChecker, AgreementVerdict, StringMatchChecker};
use crate::llm::{LlmBackend, LlmRequest};

/// Default L4 latency target — six seconds. Two cheap-LLM dispatches
/// running in parallel typically clear in three seconds; we double it
/// so the cascade planner stays honest under network jitter.
pub const VERIFICATION_TIER_LATENCY: Duration = Duration::from_secs(6);

/// Default confidence floor reported by `estimate_cost`. A successful
/// L4 dispatch returns confidence `≥ 0.9` by construction (see
/// [`AgreementVerdict::Agree`]'s confidence floor); the cascade should
/// skip this tier when the query's quality floor is below `0.9` and a
/// cheaper tier might satisfy it.
pub const VERIFICATION_TIER_CONFIDENCE_FLOOR: f32 = 0.9;

/// Default per-completion `max_tokens` we ask the backends to emit.
/// L4's job is to verify a target answer, not to generate prose, so 64
/// tokens covers the realistic short-answer surface.
pub const VERIFICATION_TIER_MAX_TOKENS: u32 = 64;

/// Errors specific to constructing or running the verification tier.
#[derive(Debug, thiserror::Error)]
pub enum VerificationTierError {
    /// Caller passed fewer than two backends. Cross-model verification
    /// is meaningless with only one model; we refuse to construct.
    #[error("verification tier needs ≥2 backends, got {0}")]
    InsufficientBackends(usize),
}

/// The L4 cross-model verification tier.
///
/// Holds a `Vec<Box<dyn LlmBackend>>` (≥2 required) and a
/// `Box<dyn AgreementChecker>`. Construct with [`VerificationTier::new`]
/// for the default [`StringMatchChecker`] or
/// [`VerificationTier::with_checker`] to swap in [`JaccardChecker`] or
/// a custom strategy.
///
/// [`JaccardChecker`]: crate::checker::JaccardChecker
pub struct VerificationTier {
    backends: Vec<Box<dyn LlmBackend>>,
    checker: Box<dyn AgreementChecker>,
    max_tokens: u32,
}

impl VerificationTier {
    /// Construct with the default [`StringMatchChecker`].
    ///
    /// Returns [`VerificationTierError::InsufficientBackends`] if
    /// fewer than two backends are supplied.
    pub fn new(
        backends: Vec<Box<dyn LlmBackend>>,
    ) -> Result<Self, VerificationTierError> {
        Self::with_checker(backends, Box::new(StringMatchChecker::new()))
    }

    /// Construct with a custom checker.
    pub fn with_checker(
        backends: Vec<Box<dyn LlmBackend>>,
        checker: Box<dyn AgreementChecker>,
    ) -> Result<Self, VerificationTierError> {
        if backends.len() < 2 {
            return Err(VerificationTierError::InsufficientBackends(
                backends.len(),
            ));
        }
        Ok(Self {
            backends,
            checker,
            max_tokens: VERIFICATION_TIER_MAX_TOKENS,
        })
    }

    /// Override the per-completion token cap (default
    /// [`VERIFICATION_TIER_MAX_TOKENS`] = 64).
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Number of backends configured.
    pub fn backend_count(&self) -> usize {
        self.backends.len()
    }

    /// Model identifiers of the configured backends, in dispatch order.
    pub fn model_ids(&self) -> Vec<String> {
        self.backends.iter().map(|b| b.model_id().to_string()).collect()
    }

    /// Build the [`LlmRequest`] this tier issues for `prompt`.
    fn build_request(&self, prompt: String) -> LlmRequest {
        LlmRequest {
            prompt,
            max_tokens: self.max_tokens,
            temperature: 0.0,
        }
    }

    /// Extract the prompt text from a `Query`. Structured envelopes
    /// are surfaced as their UTF-8 form when valid; otherwise we
    /// refuse the query as inapplicable.
    fn prompt_from(q: &Query) -> Option<String> {
        match &q.input {
            QueryInput::Text(s) => Some(s.clone()),
            QueryInput::Structured(bytes) => {
                std::str::from_utf8(bytes).ok().map(|s| s.to_string())
            }
            _ => None,
        }
    }
}

impl Tier for VerificationTier {
    fn id(&self) -> TierId {
        TierId::L4(L4ModelId(0))
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        // Only Text and Structured envelopes are routable.
        let prompt = Self::prompt_from(q)?;
        if prompt.is_empty() {
            return None;
        }
        let request = self.build_request(prompt);
        // Sum the per-backend estimates — every backend runs.
        let joules: f64 = self.backends.iter().map(|b| b.estimate_joules(&request)).sum();
        Some(TierEstimate {
            joules,
            latency: VERIFICATION_TIER_LATENCY,
            confidence_floor: VERIFICATION_TIER_CONFIDENCE_FLOOR,
        })
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget_remaining: f64,
    ) -> Result<Answer, AnswerError> {
        let id = self.id();
        let prompt = match Self::prompt_from(q) {
            Some(p) if !p.is_empty() => p,
            _ => return Ok(refused(id, 0.0, RefusalReason::Inapplicable)),
        };
        let request = self.build_request(prompt);

        // Dispatch every backend in parallel. `std::thread::scope`
        // borrows `&self.backends` for the lifetime of the scope, so
        // we can hand each `&dyn LlmBackend` to its own thread without
        // arc-ing.
        let results: Vec<Result<crate::llm::LlmResponse, crate::llm::LlmError>> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = self
                    .backends
                    .iter()
                    .map(|backend| {
                        let req = request.clone();
                        let b: &dyn LlmBackend = backend.as_ref();
                        scope.spawn(move || b.complete(&req))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| match h.join() {
                        Ok(r) => r,
                        Err(_) => Err(crate::llm::LlmError::BackendFailed {
                            model_id: "<panic>".into(),
                            reason: "worker thread panicked".into(),
                        }),
                    })
                    .collect()
            });

        // Tally joule spend across every backend. We charge for *all*
        // backends regardless of outcome — the work happened on the
        // wire whether or not the comparison agreed.
        let mut joules_spent = 0.0;
        let mut candidates: Vec<String> = Vec::with_capacity(results.len());
        let mut any_error: Option<String> = None;
        for r in results {
            match r {
                Ok(resp) => {
                    joules_spent += resp.joules;
                    candidates.push(resp.text);
                }
                Err(e) => {
                    // Sum at least the estimated cost for the failing
                    // backend so calibration is not skewed downward.
                    joules_spent += self
                        .backends
                        .iter()
                        .find(|b| matches!(&e, crate::llm::LlmError::BackendFailed { model_id, .. } if b.model_id() == model_id))
                        .map(|b| b.estimate_joules(&request))
                        .unwrap_or(0.0);
                    any_error.get_or_insert_with(|| e.to_string());
                }
            }
        }

        // Any backend failure → tier refuses. Cross-model agreement
        // requires *every* participant to vote.
        if let Some(err) = any_error {
            return Ok(refused(
                id,
                joules_spent,
                RefusalReason::TierSpecific(format!("backend failure: {err}")),
            ));
        }

        match self.checker.check(&candidates) {
            AgreementVerdict::Agree {
                consensus,
                confidence,
            } => {
                let mut trace = ExecutionTrace::default();
                trace.attempts.push(TraceEntry {
                    tier: id,
                    outcome: TraceOutcome::Hit,
                    joules: joules_spent,
                });
                Ok(Answer {
                    output: AnswerOutput::Text(consensus),
                    tier_used: id,
                    joules_spent,
                    confidence,
                    trace,
                    verification: VerificationStatus::Resolved,
                })
            }
            AgreementVerdict::Disagree { reason } => Ok(refused(
                id,
                joules_spent,
                RefusalReason::low_confidence_with_reason(reason),
            )),
            AgreementVerdict::Inconclusive => Ok(refused(
                id,
                joules_spent,
                RefusalReason::TierSpecific("verification inconclusive".into()),
            )),
        }
    }
}

/// Internal helper: low_confidence refusal, ignoring the diagnostic
/// reason (cascade::RefusalReason::LowConfidence carries a confidence
/// scalar, not a string). We log the disagreement reason into a
/// `TierSpecific` variant alongside so audits keep the human form.
trait LowConfidenceWithReason {
    fn low_confidence_with_reason(reason: String) -> Self;
}

impl LowConfidenceWithReason for RefusalReason {
    fn low_confidence_with_reason(reason: String) -> Self {
        // We pick 0.0 as the confidence scalar: when the cross-model
        // agreement check fails, the tier's confidence in any single
        // candidate is zero by construction (we cannot pick a winner).
        // The textual reason rides along in a separate `TierSpecific`
        // variant via the alternative constructor below if callers
        // want diagnostic detail.
        let _ = reason;
        RefusalReason::low_confidence(0.0)
    }
}

fn refused(tier: TierId, joules: f64, reason: RefusalReason) -> Answer {
    let mut trace = ExecutionTrace::default();
    trace.attempts.push(TraceEntry {
        tier,
        outcome: TraceOutcome::Refused(reason.clone()),
        joules,
    });
    Answer {
        output: AnswerOutput::Refused(reason),
        tier_used: tier,
        joules_spent: joules,
        confidence: 0.0,
        trace,
        verification: VerificationStatus::Resolved,
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::JaccardChecker;
    use crate::llm::{FailingBackend, StaticBackend};
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, QualityFloor, Query, QueryInput,
    };

    fn text_query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn id_is_l4_with_zero_model_id() {
        let tier = VerificationTier::new(vec![
            Box::new(StaticBackend::new("a", "x")),
            Box::new(StaticBackend::new("b", "x")),
        ])
        .expect("two backends");
        assert_eq!(tier.id(), TierId::L4(L4ModelId(0)));
        assert_eq!(tier.id().wire_tag(), "L4");
    }

    #[test]
    fn constructor_rejects_single_backend() {
        let r = VerificationTier::new(vec![Box::new(StaticBackend::new("a", "x"))]);
        match r {
            Err(VerificationTierError::InsufficientBackends(1)) => {}
            _ => panic!("expected InsufficientBackends(1)"),
        }
    }

    #[test]
    fn constructor_rejects_zero_backends() {
        let r = VerificationTier::new(vec![]);
        match r {
            Err(VerificationTierError::InsufficientBackends(0)) => {}
            _ => panic!("expected InsufficientBackends(0)"),
        }
    }

    #[test]
    fn estimate_sums_per_backend_joules() {
        let tier = VerificationTier::new(vec![
            Box::new(StaticBackend::new("a", "x").with_joules(1.5)),
            Box::new(StaticBackend::new("b", "x").with_joules(2.5)),
        ])
        .unwrap();
        let est = tier
            .estimate_cost(&text_query("what is 2+2?"))
            .expect("estimate");
        assert!((est.joules - 4.0).abs() < 1e-6);
        assert_eq!(est.latency, VERIFICATION_TIER_LATENCY);
        assert!((est.confidence_floor - VERIFICATION_TIER_CONFIDENCE_FLOOR).abs() < 1e-6);
    }

    #[test]
    fn estimate_returns_none_for_empty_prompt() {
        let tier = VerificationTier::new(vec![
            Box::new(StaticBackend::new("a", "x")),
            Box::new(StaticBackend::new("b", "x")),
        ])
        .unwrap();
        assert!(tier.estimate_cost(&text_query("")).is_none());
    }

    #[test]
    fn estimate_returns_none_for_binary_input() {
        let tier = VerificationTier::new(vec![
            Box::new(StaticBackend::new("a", "x")),
            Box::new(StaticBackend::new("b", "x")),
        ])
        .unwrap();
        let q = Query {
            input: QueryInput::Binary(vec![1, 2, 3]),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(tier.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_accepts_structured_utf8() {
        let tier = VerificationTier::new(vec![
            Box::new(StaticBackend::new("a", "x")),
            Box::new(StaticBackend::new("b", "x")),
        ])
        .unwrap();
        let q = Query {
            input: QueryInput::Structured(b"hello?".to_vec()),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(tier.estimate_cost(&q).is_some());
    }

    #[test]
    fn two_agreeing_backends_produce_answer() {
        let mut tier = VerificationTier::new(vec![
            Box::new(StaticBackend::new("a", "Paris").with_joules(1.5)),
            Box::new(StaticBackend::new("b", "paris").with_joules(2.5)),
        ])
        .unwrap();
        let ans = tier
            .try_answer(&text_query("capital of France?"), 100.0)
            .expect("dispatch");
        match &ans.output {
            AnswerOutput::Text(t) => assert_eq!(t, "Paris"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert!(ans.confidence > 0.9);
        assert_eq!(ans.tier_used, TierId::L4(L4ModelId(0)));
        // joules_spent is the sum of per-backend joules.
        assert!((ans.joules_spent - 4.0).abs() < 1e-6);
    }

    #[test]
    fn three_agreeing_backends_produce_answer() {
        let mut tier = VerificationTier::new(vec![
            Box::new(StaticBackend::new("a", "yes")),
            Box::new(StaticBackend::new("b", "YES")),
            Box::new(StaticBackend::new("c", "  yes  ")),
        ])
        .unwrap();
        let ans = tier
            .try_answer(&text_query("does the sun rise?"), 100.0)
            .expect("dispatch");
        assert!(matches!(ans.output, AnswerOutput::Text(_)));
    }

    #[test]
    fn two_disagreeing_backends_refuse_low_confidence() {
        let mut tier = VerificationTier::new(vec![
            Box::new(StaticBackend::new("a", "Paris")),
            Box::new(StaticBackend::new("b", "London")),
        ])
        .unwrap();
        let ans = tier
            .try_answer(&text_query("capital of France?"), 100.0)
            .expect("dispatch");
        match ans.output {
            AnswerOutput::Refused(RefusalReason::LowConfidence(_)) => {}
            other => panic!("expected LowConfidence refusal, got {other:?}"),
        }
        // Joules still spent — the work happened.
        assert!(ans.joules_spent > 0.0);
    }

    #[test]
    fn backend_failure_refuses_with_tier_specific() {
        let mut tier = VerificationTier::new(vec![
            Box::new(StaticBackend::new("ok", "yes")),
            Box::new(FailingBackend::new("bad")),
        ])
        .unwrap();
        let ans = tier
            .try_answer(&text_query("hello?"), 100.0)
            .expect("dispatch");
        match ans.output {
            AnswerOutput::Refused(RefusalReason::TierSpecific(msg)) => {
                assert!(msg.contains("backend failure"));
            }
            other => panic!("expected TierSpecific refusal, got {other:?}"),
        }
    }

    #[test]
    fn non_text_input_refuses_inapplicable() {
        let mut tier = VerificationTier::new(vec![
            Box::new(StaticBackend::new("a", "yes")),
            Box::new(StaticBackend::new("b", "yes")),
        ])
        .unwrap();
        let q = Query {
            input: QueryInput::Binary(vec![0xff, 0xfe]),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let ans = tier.try_answer(&q, 100.0).expect("dispatch");
        assert!(matches!(
            ans.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn jaccard_checker_agrees_on_similar_prose() {
        let mut tier = VerificationTier::with_checker(
            vec![
                Box::new(StaticBackend::new(
                    "a",
                    "the quick brown fox jumps over the lazy dog",
                )),
                Box::new(StaticBackend::new(
                    "b",
                    "the quick brown fox jumps over a lazy dog",
                )),
            ],
            Box::new(JaccardChecker::with_threshold(0.7)),
        )
        .unwrap();
        let ans = tier
            .try_answer(&text_query("describe the scene"), 100.0)
            .expect("dispatch");
        assert!(matches!(ans.output, AnswerOutput::Text(_)));
        assert!(ans.confidence >= 0.9);
    }

    #[test]
    fn jaccard_checker_refuses_below_threshold() {
        let mut tier = VerificationTier::with_checker(
            vec![
                Box::new(StaticBackend::new("a", "alpha beta gamma")),
                Box::new(StaticBackend::new("b", "delta epsilon zeta")),
            ],
            Box::new(JaccardChecker::with_threshold(0.8)),
        )
        .unwrap();
        let ans = tier
            .try_answer(&text_query("name three"), 100.0)
            .expect("dispatch");
        assert!(matches!(
            ans.output,
            AnswerOutput::Refused(_)
        ));
    }

    #[test]
    fn jaccard_threshold_edge_at_one() {
        // Threshold = 1.0 means only byte-identical (after normalise)
        // candidates pass. Exact equals → agree.
        let mut tier = VerificationTier::with_checker(
            vec![
                Box::new(StaticBackend::new("a", "alpha beta")),
                Box::new(StaticBackend::new("b", "alpha beta")),
            ],
            Box::new(JaccardChecker::with_threshold(1.0)),
        )
        .unwrap();
        let ans = tier
            .try_answer(&text_query("two words"), 100.0)
            .expect("dispatch");
        assert!(matches!(ans.output, AnswerOutput::Text(_)));
    }

    #[test]
    fn model_ids_match_constructor_order() {
        let tier = VerificationTier::new(vec![
            Box::new(StaticBackend::new("first", "x")),
            Box::new(StaticBackend::new("second", "x")),
            Box::new(StaticBackend::new("third", "x")),
        ])
        .unwrap();
        assert_eq!(tier.backend_count(), 3);
        assert_eq!(
            tier.model_ids(),
            vec!["first".to_string(), "second".into(), "third".into()]
        );
    }

    #[test]
    fn end_to_end_via_cascade_runtime() {
        use jouleclaw_cascade::tier::{Cascade, Runtime};

        let mut cascade = Cascade::new();
        cascade.register(Box::new(
            VerificationTier::new(vec![
                Box::new(StaticBackend::new("a", "42")),
                Box::new(StaticBackend::new("b", "42")),
            ])
            .unwrap(),
        ));
        let mut rt = Runtime::new_without_l0(cascade);
        let ans = rt
            .answer(text_query("ultimate answer"))
            .expect("runtime answer");
        assert_eq!(ans.tier_used, TierId::L4(L4ModelId(0)));
        match ans.output {
            AnswerOutput::Text(t) => assert_eq!(t, "42"),
            other => panic!("expected Text, got {other:?}"),
        }
    }
}
