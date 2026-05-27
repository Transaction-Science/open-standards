//! [`jouleclaw_cascade::Tier`] implementation for the L0.5 tool-compute tier.
//!
//! The tier owns a [`ToolRouter`] (built-in matchers + any extras the caller
//! registers) and dispatches matched queries into
//! [`jouleclaw_tools::execute`]. Hits return [`AnswerOutput::Structured`]
//! with the canonical tool result JSON-encoded; misses return
//! [`AnswerOutput::Refused(RefusalReason::Inapplicable)`] so the cascade
//! moves to the next tier without spending the tier budget.

use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, Query, QueryInput, RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;

use crate::router::{MIN_MATCH_CONFIDENCE, ToolRouter};

/// Flat dispatch energy. ~15 µJ — covers the regex routing pass plus a
/// pure-CPU tool call. The cost is constant rather than measured because
/// the work is bounded (microseconds of CPU); a sampler here would burn
/// more energy than it accounts for.
pub const TOOL_TIER_JOULES: f64 = 15e-6;

/// Flat dispatch latency. ~5 µs — regex match plus tool dispatch.
pub const TOOL_TIER_LATENCY: Duration = Duration::from_micros(5);

/// Confidence floor reported by `estimate_cost`. The tier only attempts
/// execution for queries that the router classified at or above
/// [`MIN_MATCH_CONFIDENCE`]; for those, the tool's output is by
/// construction faithful to its inputs (zero-hallucination by design), so
/// the floor is `1.0`.
pub const TOOL_TIER_CONFIDENCE_FLOOR: f32 = 1.0;

/// Errors surfaced by the tool tier. None are user-facing today; surfacing
/// them via [`thiserror`] keeps the door open for downstream consumers
/// that want to wrap dispatch failures into their own error type.
#[derive(Debug, thiserror::Error)]
pub enum ToolTierError {
    /// The matched tool failed to execute. Carries the message returned
    /// by [`jouleclaw_tools::execute`].
    #[error("tool dispatch failed: {0}")]
    DispatchFailed(String),
    /// JSON serialisation of the tool result failed.
    #[error("tool result serialization failed: {0}")]
    SerializeFailed(String),
}

/// The L0.5 tool-compute tier.
///
/// Construct with [`ToolTier::new`] for the built-in cascade of matchers,
/// or [`ToolTier::with_router`] to supply a router with extra matchers
/// pre-registered.
pub struct ToolTier {
    router: ToolRouter,
}

impl ToolTier {
    /// Construct a tier with the built-in router (no extra matchers).
    pub fn new() -> Self {
        Self {
            router: ToolRouter::new(),
        }
    }

    /// Construct a tier wrapping a caller-supplied router.
    pub fn with_router(router: ToolRouter) -> Self {
        Self { router }
    }

    /// Mutable access to the router, for `register_matcher` / `register_tool`.
    pub fn router_mut(&mut self) -> &mut ToolRouter {
        &mut self.router
    }

    /// Borrow the router (e.g. to probe `match_query` for diagnostics).
    pub fn router(&self) -> &ToolRouter {
        &self.router
    }
}

impl Default for ToolTier {
    fn default() -> Self {
        Self::new()
    }
}

impl Tier for ToolTier {
    fn id(&self) -> TierId {
        TierId::L0_5ToolCompute
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        // We only know how to handle text queries.
        let text = match &q.input {
            QueryInput::Text(s) => s.as_str(),
            _ => return None,
        };
        // If the query doesn't look like a tool invocation, decline. The
        // router runs in nanoseconds (LazyLock-cached regexes); doing it
        // here lets the runtime skip us cleanly.
        let matched = self.router.match_query(text)?;
        if matched.confidence < MIN_MATCH_CONFIDENCE {
            return None;
        }
        Some(TierEstimate {
            joules: TOOL_TIER_JOULES,
            latency: TOOL_TIER_LATENCY,
            confidence_floor: TOOL_TIER_CONFIDENCE_FLOOR,
        })
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget_remaining: f64,
    ) -> Result<Answer, AnswerError> {
        let text = match &q.input {
            QueryInput::Text(s) => s.as_str(),
            _ => return Ok(refused(RefusalReason::Inapplicable)),
        };

        let matched = match self.router.match_query(text) {
            Some(m) if m.confidence >= MIN_MATCH_CONFIDENCE => m,
            _ => return Ok(refused(RefusalReason::Inapplicable)),
        };

        match jouleclaw_tools::execute(&matched.tool) {
            Ok(result) => {
                let bytes = serde_json::to_vec(&result).map_err(|e| AnswerError::TierFailed {
                    tier: TierId::L0_5ToolCompute,
                    cause: format!("serialize: {e}"),
                })?;
                Ok(Answer {
                    output: AnswerOutput::Structured(bytes),
                    tier_used: TierId::L0_5ToolCompute,
                    joules_spent: TOOL_TIER_JOULES,
                    confidence: 1.0,
                    trace: ExecutionTrace::default(),
                    verification: VerificationStatus::Resolved,
                })
            }
            Err(_msg) => {
                // Tool matched but execution failed (bad input, etc.).
                // Refuse so the cascade keeps walking — never block on a
                // tool error.
                Ok(refused(RefusalReason::TierSpecific(
                    "tool execution failed".to_string(),
                )))
            }
        }
    }
}

fn refused(reason: RefusalReason) -> Answer {
    Answer {
        output: AnswerOutput::Refused(reason),
        tier_used: TierId::L0_5ToolCompute,
        joules_spent: TOOL_TIER_JOULES,
        confidence: 0.0,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
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
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn id_is_l0_5_tool_compute() {
        let t = ToolTier::new();
        assert_eq!(t.id(), TierId::L0_5ToolCompute);
        assert_eq!(t.id().wire_tag(), "L0.5");
        assert_eq!(t.id().name(), "ToolCompute");
    }

    #[test]
    fn estimate_text_with_match_returns_some() {
        let t = ToolTier::new();
        let q = text_query("calculate 2 + 2");
        let est = t.estimate_cost(&q).expect("matched math → estimate");
        assert_eq!(est.joules, TOOL_TIER_JOULES);
        assert_eq!(est.latency, TOOL_TIER_LATENCY);
        assert!((est.confidence_floor - TOOL_TIER_CONFIDENCE_FLOOR).abs() < f32::EPSILON);
    }

    #[test]
    fn estimate_unmatched_text_returns_none() {
        let t = ToolTier::new();
        let q = text_query("best programming language 2026");
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_empty_text_returns_none() {
        let t = ToolTier::new();
        let q = text_query("");
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_non_text_returns_none() {
        let t = ToolTier::new();
        let q = Query {
            input: QueryInput::Binary(vec![1, 2, 3]),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn try_answer_math() {
        let mut t = ToolTier::new();
        let q = text_query("calculate 2 + 2");
        let ans = t.try_answer(&q, 1.0).expect("math succeeds");
        match ans.output {
            AnswerOutput::Structured(bytes) => {
                let s: String = serde_json::from_slice(&bytes).expect("string payload");
                assert!(s.contains('4'), "expected '4' in: {s}");
            }
            other => panic!("expected Structured, got {other:?}"),
        }
        assert_eq!(ans.tier_used, TierId::L0_5ToolCompute);
        assert!((ans.confidence - 1.0).abs() < f32::EPSILON);
        assert_eq!(ans.joules_spent, TOOL_TIER_JOULES);
    }

    #[test]
    fn try_answer_unit_conversion() {
        let mut t = ToolTier::new();
        let q = text_query("convert 5 miles to km");
        let ans = t.try_answer(&q, 1.0).expect("conversion succeeds");
        match ans.output {
            AnswerOutput::Structured(bytes) => {
                let s: String = serde_json::from_slice(&bytes).expect("string payload");
                // 5 mi ≈ 8.04 km.
                assert!(s.contains("8.04") || s.contains("8,04"), "got: {s}");
            }
            other => panic!("expected Structured, got {other:?}"),
        }
    }

    #[test]
    fn try_answer_uuid_produces_dashed_output() {
        let mut t = ToolTier::new();
        let q = text_query("generate uuid");
        let ans = t.try_answer(&q, 1.0).expect("uuid succeeds");
        match ans.output {
            AnswerOutput::Structured(bytes) => {
                let s: String = serde_json::from_slice(&bytes).expect("string payload");
                assert!(s.contains('-'), "uuid lacks dashes: {s}");
            }
            other => panic!("expected Structured, got {other:?}"),
        }
    }

    #[test]
    fn try_answer_sha256() {
        let mut t = ToolTier::new();
        let q = text_query("sha256 of hello");
        let ans = t.try_answer(&q, 1.0).expect("sha256 succeeds");
        match ans.output {
            AnswerOutput::Structured(bytes) => {
                let s: String = serde_json::from_slice(&bytes).expect("string payload");
                // SHA-256("hello") = 2cf24dba… — accept either bare hex or
                // a tool-side prefix like "SHA256: …".
                assert!(s.contains("2cf24dba"), "got: {s}");
            }
            other => panic!("expected Structured, got {other:?}"),
        }
    }

    #[test]
    fn try_answer_percentage() {
        let mut t = ToolTier::new();
        let q = text_query("what is 25% of 200");
        let ans = t.try_answer(&q, 1.0).expect("percentage succeeds");
        match ans.output {
            AnswerOutput::Structured(bytes) => {
                let s: String = serde_json::from_slice(&bytes).expect("string payload");
                assert!(s.contains("50"), "25% of 200 = 50, got: {s}");
            }
            other => panic!("expected Structured, got {other:?}"),
        }
    }

    #[test]
    fn try_answer_unmatched_refuses() {
        let mut t = ToolTier::new();
        let q = text_query("best programming language 2026");
        let ans = t.try_answer(&q, 1.0).expect("non-tool query refuses");
        match ans.output {
            AnswerOutput::Refused(RefusalReason::Inapplicable) => {}
            other => panic!("expected Refused(Inapplicable), got {other:?}"),
        }
        assert_eq!(ans.confidence, 0.0);
        assert_eq!(ans.tier_used, TierId::L0_5ToolCompute);
    }

    #[test]
    fn try_answer_non_text_refuses() {
        let mut t = ToolTier::new();
        let q = Query {
            input: QueryInput::Binary(vec![1, 2, 3]),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let ans = t.try_answer(&q, 1.0).expect("non-text refuses cleanly");
        assert!(matches!(
            ans.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn end_to_end_via_cascade_runtime() {
        // Wire the tier into a Runtime and make sure the cascade walker
        // actually dispatches to us.
        use jouleclaw_cascade::tier::{Cascade, Runtime};

        let mut cascade = Cascade::new();
        cascade.register(Box::new(ToolTier::new()));
        let mut rt = Runtime::new_without_l0(cascade);

        let q = text_query("calculate 7 * 6");
        let ans = rt.answer(q).expect("runtime answer");
        assert_eq!(ans.tier_used, TierId::L0_5ToolCompute);
        assert!((ans.confidence - 1.0).abs() < f32::EPSILON);
        match ans.output {
            AnswerOutput::Structured(bytes) => {
                let s: String = serde_json::from_slice(&bytes).expect("string payload");
                assert!(s.contains("42"), "7*6 = 42, got: {s}");
            }
            other => panic!("expected Structured, got {other:?}"),
        }
    }

    #[test]
    fn router_extension_via_register_tool() {
        // Demonstrate that downstream crates can extend the dispatch
        // surface without touching the built-in cascade.
        let mut t = ToolTier::new();
        t.router_mut()
            .register_tool("ping", DeterministicToolKind::Uuid);
        let q = text_query("ping");
        let ans = t.try_answer(&q, 1.0).expect("ping → uuid");
        match ans.output {
            AnswerOutput::Structured(bytes) => {
                let s: String = serde_json::from_slice(&bytes).expect("string payload");
                assert!(s.contains('-'), "uuid output: {s}");
            }
            other => panic!("expected Structured, got {other:?}"),
        }
    }

    // Bring DeterministicToolKind into scope for the test above.
    use jouleclaw_tools::DeterministicToolKind;
}
