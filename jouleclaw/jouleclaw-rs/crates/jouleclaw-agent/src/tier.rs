//! The L6 agent tier.

use crate::cascade_trait::AgentCascade;
use crate::composer::{Composer, Concatenator};
use crate::planner::{AgentPlanner, KeywordPlanner};
use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, Query, QueryInput, QualityFloor,
    RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;
use std::time::Duration;

/// Conservative typical energy for a multi-step agent run (~6 J). The
/// real cost is the sum of sub-dispatches and is reported exactly in the
/// returned `Answer`.
pub const AGENT_TYPICAL_JOULES: f64 = 6_000_000e-6;

/// Errors specific to agent construction/operation that aren't cascade
/// `AnswerError`s.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("agent produced no sub-queries to dispatch")]
    NoPlan,
}

/// L6 multi-step agent. Generic over the planner, the cascade shim, and
/// the composer so the same tier logic spans a keyword-split reference
/// and an LLM-driven deployment.
pub struct AgentTier<P = KeywordPlanner, C = crate::cascade_trait::MockCascade, M = Concatenator>
where
    P: AgentPlanner,
    C: AgentCascade,
    M: Composer,
{
    planner: P,
    cascade: C,
    composer: M,
    /// Minimum sub-queries for the tier to consider itself applicable.
    /// A query that doesn't decompose into ≥2 steps is better served by
    /// a single-shot tier, so the agent refuses it (returns `None` from
    /// `estimate_cost`).
    min_steps: usize,
}

impl<C: AgentCascade> AgentTier<KeywordPlanner, C, Concatenator> {
    /// Agent with the default keyword planner and newline concatenator,
    /// wired to the given cascade shim.
    pub fn new(cascade: C) -> Self {
        Self {
            planner: KeywordPlanner,
            cascade,
            composer: Concatenator::default(),
            min_steps: 2,
        }
    }
}

impl<P, C, M> AgentTier<P, C, M>
where
    P: AgentPlanner,
    C: AgentCascade,
    M: Composer,
{
    /// Fully custom agent.
    pub fn with_parts(planner: P, cascade: C, composer: M, min_steps: usize) -> Self {
        Self {
            planner,
            cascade,
            composer,
            min_steps: min_steps.max(1),
        }
    }

    /// Borrow the cascade shim (e.g. to inspect a `MockCascade`'s
    /// recorded queries in tests).
    pub fn cascade(&self) -> &C {
        &self.cascade
    }

    fn query_text(q: &Query) -> Option<&str> {
        match &q.input {
            QueryInput::Text(t) => Some(t.as_str()),
            QueryInput::Multimodal { text, .. } => Some(text.as_str()),
            _ => None,
        }
    }

    /// Build a sub-query that inherits the parent's budget/quality/
    /// context but carries the sub-query text and a fraction of the
    /// remaining budget.
    fn sub_query(parent: &Query, text: &str) -> Query {
        let mut sub = parent.clone();
        sub.input = QueryInput::Text(text.to_string());
        // Sub-queries accept partial answers; the agent composes them.
        sub.quality = QualityFloor::chat();
        sub
    }
}

impl<P, C, M> Tier for AgentTier<P, C, M>
where
    P: AgentPlanner,
    C: AgentCascade,
    M: Composer,
{
    fn id(&self) -> TierId {
        TierId::L6Agent
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        let text = Self::query_text(q)?;
        if text.trim().is_empty() {
            return None;
        }
        let steps = self.planner.plan(q);
        if steps.len() < self.min_steps {
            // Not a multi-step query — let a single-shot tier take it.
            return None;
        }
        Some(TierEstimate {
            joules: AGENT_TYPICAL_JOULES,
            latency: Duration::from_secs(10),
            confidence_floor: 0.6,
        })
    }

    fn try_answer(&mut self, q: &Query, budget_remaining: f64) -> Result<Answer, AnswerError> {
        let steps = self.planner.plan(q);
        if steps.is_empty() {
            return Ok(refusal(RefusalReason::Inapplicable, 0.0));
        }

        let mut parts: Vec<String> = Vec::with_capacity(steps.len());
        let mut spent = 0.0f64;
        let mut min_conf = 1.0f32;

        for step in &steps {
            // Budget guard: if the next sub-dispatch could blow the cap,
            // stop and refuse so the runtime's accounting stays honest.
            if spent >= budget_remaining {
                return Err(AnswerError::BudgetExhausted {
                    spent,
                    limit: budget_remaining,
                    attempted_tiers: vec![(TierId::L6Agent, spent)],
                });
            }
            let sub = Self::sub_query(q, &step.text);
            match self.cascade.dispatch(&sub) {
                Ok(ans) => {
                    spent += ans.joules_spent;
                    min_conf = min_conf.min(ans.confidence);
                    match ans.output {
                        AnswerOutput::Text(t) => parts.push(t),
                        AnswerOutput::Structured(bytes) => {
                            parts.push(String::from_utf8_lossy(&bytes).into_owned())
                        }
                        AnswerOutput::Refused(_) => {
                            // A sub-query the cascade couldn't answer means
                            // the agent can't compose a complete result.
                            return Ok(refusal(
                                RefusalReason::low_confidence(0.0),
                                spent,
                            ));
                        }
                    }
                }
                Err(e) => {
                    // Propagate budget exhaustion; otherwise the whole
                    // multi-step plan is unsatisfiable → refuse.
                    if let AnswerError::BudgetExhausted { .. } = e {
                        return Err(e);
                    }
                    return Ok(refusal(
                        RefusalReason::TierSpecific(format!("sub-dispatch failed: {e}")),
                        spent,
                    ));
                }
            }
        }

        let composed = self.composer.compose(&parts);
        Ok(Answer {
            output: AnswerOutput::Text(composed),
            tier_used: TierId::L6Agent,
            joules_spent: spent,
            // The agent is only as confident as its weakest sub-step.
            confidence: min_conf,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        })
    }
}

fn refusal(reason: RefusalReason, joules: f64) -> Answer {
    Answer {
        output: AnswerOutput::Refused(reason),
        tier_used: TierId::L6Agent,
        joules_spent: joules,
        confidence: 0.0,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade_trait::MockCascade;
    use jouleclaw_cascade::types::{ContextRef, JouleBudget};

    fn q(text: &str) -> Query {
        Query {
            input: QueryInput::Text(text.to_string()),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn id_is_l6() {
        let a = AgentTier::new(MockCascade::echo());
        assert_eq!(a.id(), TierId::L6Agent);
        assert_eq!(a.id().wire_tag(), "L6");
    }

    #[test]
    fn single_clause_not_applicable() {
        let a = AgentTier::new(MockCascade::echo());
        assert!(a.estimate_cost(&q("what is the capital of france")).is_none());
    }

    #[test]
    fn multi_clause_applicable() {
        let a = AgentTier::new(MockCascade::echo());
        assert!(a
            .estimate_cost(&q("capital of france and population of germany"))
            .is_some());
    }

    #[test]
    fn non_text_not_applicable() {
        let a = AgentTier::new(MockCascade::echo());
        let query = Query {
            input: QueryInput::Binary(vec![1]),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(a.estimate_cost(&query).is_none());
    }

    #[test]
    fn two_clauses_two_dispatches() {
        let mut a = AgentTier::new(MockCascade::echo());
        let ans = a
            .try_answer(&q("capital of france and population of germany"), 100.0)
            .unwrap();
        match ans.output {
            AnswerOutput::Text(t) => {
                assert!(t.contains("capital of france"));
                assert!(t.contains("population of germany"));
            }
            _ => panic!("expected text"),
        }
        assert_eq!(a.cascade().seen.len(), 2);
    }

    #[test]
    fn joules_are_summed() {
        let mut mock = MockCascade::echo();
        mock.joules_per_dispatch = 2.0;
        let mut a = AgentTier::new(mock);
        let ans = a.try_answer(&q("a and b and c"), 100.0).unwrap();
        assert!((ans.joules_spent - 6.0).abs() < 1e-9); // 3 × 2 J
    }

    #[test]
    fn confidence_is_weakest_link() {
        let mut mock = MockCascade::echo();
        mock.confidence = 0.55;
        let mut a = AgentTier::new(mock);
        let ans = a.try_answer(&q("x and y"), 100.0).unwrap();
        assert!((ans.confidence - 0.55).abs() < 1e-6);
    }

    #[test]
    fn sub_dispatch_failure_refuses() {
        let mut a = AgentTier::new(MockCascade::failing_on(1));
        let ans = a.try_answer(&q("first and second and third"), 100.0).unwrap();
        assert!(matches!(ans.output, AnswerOutput::Refused(_)));
    }

    #[test]
    fn refused_sub_answer_refuses_whole() {
        // A cascade that refuses the sub-query.
        struct RefusingCascade;
        impl AgentCascade for RefusingCascade {
            fn dispatch(&mut self, _q: &Query) -> Result<Answer, AnswerError> {
                Ok(Answer {
                    output: AnswerOutput::Refused(RefusalReason::Inapplicable),
                    tier_used: TierId::L0,
                    joules_spent: 0.1,
                    confidence: 0.0,
                    trace: ExecutionTrace::default(),
                    verification: VerificationStatus::Resolved,
                })
            }
        }
        let mut a = AgentTier::new(RefusingCascade);
        let ans = a.try_answer(&q("a and b"), 100.0).unwrap();
        assert!(matches!(ans.output, AnswerOutput::Refused(_)));
    }

    #[test]
    fn empty_plan_refuses() {
        let mut a = AgentTier::new(MockCascade::echo());
        let query = Query {
            input: QueryInput::Binary(vec![]),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let ans = a.try_answer(&query, 100.0).unwrap();
        assert!(matches!(ans.output, AnswerOutput::Refused(RefusalReason::Inapplicable)));
    }

    #[test]
    fn budget_exhaustion_propagates() {
        let mut mock = MockCascade::echo();
        mock.joules_per_dispatch = 10.0;
        let mut a = AgentTier::new(mock);
        // Three steps × 10 J = 30 J, but only 15 J remaining.
        let res = a.try_answer(&q("a and b and c"), 15.0);
        assert!(matches!(res, Err(AnswerError::BudgetExhausted { .. })));
    }

    #[test]
    fn structured_sub_answer_is_stringified() {
        struct StructCascade;
        impl AgentCascade for StructCascade {
            fn dispatch(&mut self, _q: &Query) -> Result<Answer, AnswerError> {
                Ok(Answer {
                    output: AnswerOutput::Structured(b"{\"k\":1}".to_vec()),
                    tier_used: TierId::L0,
                    joules_spent: 0.1,
                    confidence: 0.9,
                    trace: ExecutionTrace::default(),
                    verification: VerificationStatus::Resolved,
                })
            }
        }
        let mut a = AgentTier::new(StructCascade);
        let ans = a.try_answer(&q("a and b"), 100.0).unwrap();
        match ans.output {
            AnswerOutput::Text(t) => assert!(t.contains("\"k\":1")),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn custom_composer_separator() {
        let a = AgentTier::with_parts(
            KeywordPlanner,
            MockCascade::echo(),
            Concatenator::with_separator(" || "),
            2,
        );
        let mut a = a;
        let ans = a.try_answer(&q("a and b"), 100.0).unwrap();
        match ans.output {
            AnswerOutput::Text(t) => assert!(t.contains(" || ")),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn end_to_end_via_runtime() {
        use jouleclaw_cascade::tier::Cascade;
        use jouleclaw_cascade::tier::Runtime;
        // Register the agent as the only tier behind the runtime's L0.
        let mut cascade = Cascade::new();
        cascade.register(Box::new(AgentTier::new(MockCascade::echo())));
        let mut rt = Runtime::new(cascade);
        let ans = rt
            .answer(q("alpha and beta"))
            .expect("runtime should resolve via agent");
        match ans.output {
            AnswerOutput::Text(t) => {
                assert!(t.contains("alpha"));
                assert!(t.contains("beta"));
            }
            _ => panic!("expected composed text"),
        }
    }

    #[test]
    fn min_steps_configurable() {
        // min_steps = 3 → a two-clause query is not applicable.
        let a = AgentTier::with_parts(
            KeywordPlanner,
            MockCascade::echo(),
            Concatenator::default(),
            3,
        );
        assert!(a.estimate_cost(&q("a and b")).is_none());
        assert!(a.estimate_cost(&q("a and b and c")).is_some());
    }
}
