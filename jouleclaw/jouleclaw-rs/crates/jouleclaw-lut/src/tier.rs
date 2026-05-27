//! [`jouleclaw_cascade::tier::Tier`] impl for [`Lut`].
//!
//! Registering a [`Lut`] as the **first** tier in a [`Cascade`] short-
//! circuits the cost-ordered walk: hit → sub-nanojoule return,
//! miss → refuse + fall through. This matches the doctrine that the
//! LUT lives BELOW [`L0Cache`] in the conceptual stack.
//!
//! Energy model:
//! ```text
//! joules = LUT_PROBE_JOULES = 1e-9         (1 nJ, flat)
//! latency = 100 ns                          (a single hashmap probe)
//! confidence_floor = 0.9 (empty: 0.0)       (we only claim once the
//!                                            table is populated)
//! ```
//!
//! [`Lut`]: crate::lut::Lut
//! [`Cascade`]: jouleclaw_cascade::tier::Cascade
//! [`L0Cache`]: jouleclaw_cascade::l0_cache::L0Cache

use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, Query, QueryInput,
    RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;

use crate::lut::Lut;

/// Flat probe energy in joules. ~1 nJ — a single hashmap probe.
pub const LUT_PROBE_JOULES: f64 = 1e-9;

/// Flat probe latency.
pub const LUT_PROBE_LATENCY: Duration = Duration::from_nanos(100);

impl Tier for Lut {
    fn id(&self) -> TierId {
        // L0 because this is cache-class energy; it pre-empts the
        // content-addressed cache that would otherwise have evicted
        // these entries.
        TierId::L0
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        // LUT only handles Text queries — bytes/images/audio have no
        // canonical normalisation surface here.
        match &q.input {
            QueryInput::Text(_) => Some(TierEstimate {
                joules: LUT_PROBE_JOULES,
                latency: LUT_PROBE_LATENCY,
                confidence_floor: if self.is_empty() { 0.0 } else { 0.9 },
            }),
            _ => None,
        }
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget_remaining: f64,
    ) -> Result<Answer, AnswerError> {
        let input = match &q.input {
            QueryInput::Text(s) => s.as_str(),
            // Defence in depth: the runtime should have skipped us via
            // `estimate_cost` returning None, but if a caller invokes
            // `try_answer` directly we still refuse cleanly.
            _ => {
                return Ok(Answer {
                    output: AnswerOutput::Refused(RefusalReason::Inapplicable),
                    tier_used: TierId::L0,
                    joules_spent: LUT_PROBE_JOULES,
                    confidence: 0.0,
                    trace: ExecutionTrace::default(),
                    verification: VerificationStatus::Resolved,
                });
            }
        };

        match self.try_lookup(input) {
            Some(hit) => {
                // Prefer Text output when the stored bytes are valid
                // UTF-8 — most LUT payloads will be. Fall back to
                // Structured otherwise.
                let output = match std::str::from_utf8(&hit.output) {
                    Ok(s) => AnswerOutput::Text(s.to_string()),
                    Err(_) => AnswerOutput::Structured(hit.output.clone()),
                };
                let joules_spent = (hit.declared_cost_uj as f64) / 1e6;
                Ok(Answer {
                    output,
                    tier_used: TierId::L0,
                    joules_spent,
                    confidence: 1.0,
                    trace: ExecutionTrace::default(),
                    verification: VerificationStatus::Resolved,
                })
            }
            None => Ok(Answer {
                output: AnswerOutput::Refused(RefusalReason::Inapplicable),
                tier_used: TierId::L0,
                joules_spent: LUT_PROBE_JOULES,
                confidence: 0.0,
                trace: ExecutionTrace::default(),
                verification: VerificationStatus::Resolved,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, QualityFloor, Query, QueryInput,
    };

    fn text_query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn id_is_l0() {
        let lut = Lut::new();
        assert_eq!(lut.id(), TierId::L0);
    }

    #[test]
    fn estimate_cost_text_with_entries() {
        let mut lut = Lut::new();
        lut.register("hello", "world", "src");
        let q = text_query("hello");
        let est = lut.estimate_cost(&q).expect("text + entries → some");
        assert_eq!(est.joules, LUT_PROBE_JOULES);
        assert_eq!(est.latency, LUT_PROBE_LATENCY);
        assert!((est.confidence_floor - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn estimate_cost_text_when_empty_floor_is_zero() {
        let lut = Lut::new();
        let q = text_query("anything");
        let est = lut.estimate_cost(&q).expect("text → some even when empty");
        assert_eq!(est.confidence_floor, 0.0);
    }

    #[test]
    fn estimate_cost_non_text_is_none() {
        let lut = Lut::new();
        let q = Query {
            input: QueryInput::Binary(vec![1, 2, 3]),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(lut.estimate_cost(&q).is_none());
    }

    #[test]
    fn try_answer_hit_returns_text_with_declared_cost() {
        let mut lut = Lut::new();
        // 7 µJ declared cost → 7e-6 J reported.
        lut.register_with_cost("gcd 12 8", "4", 7, "lawful:gcd");
        let q = text_query("  GCD 12 8  ");
        let ans = lut
            .try_answer(&q, 1.0)
            .expect("try_answer must not error on hit");
        match ans.output {
            AnswerOutput::Text(ref s) => assert_eq!(s, "4"),
            other => panic!("expected Text(\"4\"), got {:?}", other),
        }
        assert_eq!(ans.tier_used, TierId::L0);
        assert_eq!(ans.confidence, 1.0);
        assert!((ans.joules_spent - 7e-6).abs() < 1e-12);
        assert!(matches!(ans.verification, VerificationStatus::Resolved));
    }

    #[test]
    fn try_answer_hit_with_non_utf8_returns_structured() {
        let mut lut = Lut::new();
        let bytes = vec![0xff, 0xfe, 0xfd];
        lut.register_with_cost("k", bytes.clone(), 1, "src");
        let q = text_query("k");
        let ans = lut.try_answer(&q, 1.0).expect("hit");
        match ans.output {
            AnswerOutput::Structured(ref b) => assert_eq!(b, &bytes),
            other => panic!("expected Structured, got {:?}", other),
        }
    }

    #[test]
    fn try_answer_miss_refuses() {
        let mut lut = Lut::new();
        lut.register("hello", "world", "src");
        let q = text_query("goodbye");
        let ans = lut.try_answer(&q, 1.0).expect("miss returns Ok(Refused)");
        match ans.output {
            AnswerOutput::Refused(RefusalReason::Inapplicable) => {}
            other => panic!("expected Refused(Inapplicable), got {:?}", other),
        }
        assert_eq!(ans.confidence, 0.0);
        assert_eq!(ans.tier_used, TierId::L0);
    }

    #[test]
    fn try_answer_non_text_refuses_cleanly() {
        let mut lut = Lut::new();
        lut.register("hello", "world", "src");
        let q = Query {
            input: QueryInput::Binary(vec![1, 2, 3]),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let ans = lut.try_answer(&q, 1.0).expect("non-text returns Ok(Refused)");
        match ans.output {
            AnswerOutput::Refused(RefusalReason::Inapplicable) => {}
            other => panic!("expected Refused(Inapplicable), got {:?}", other),
        }
    }
}
