//! R1 tests — L0 cache and cascade walker.
//!
//! The load-bearing tests:
//!   1. L0 cache hit returns the right answer.
//!   2. L0 hit is orders of magnitude cheaper than L4 dispatch.
//!   3. Cascade falls through L0 miss to L4 and records the answer
//!      in the cache; the second identical query hits L0.

use jouleclaw_cascade::*;

// ============================================================
// A mock L4 tier for testing
// ============================================================

/// A tier that pretends to be L4 — answers every query with a fixed
/// string at a fixed (expensive) cost. Used to demonstrate the joule
/// difference between L0 hits and L4 dispatch.
struct MockL4 {
    cost_joules: f64,
    confidence: f32,
    answer_text: String,
    call_count: u32,
}

impl MockL4 {
    fn new(cost_joules: f64, answer_text: impl Into<String>) -> Self {
        Self {
            cost_joules,
            confidence: 0.95,
            answer_text: answer_text.into(),
            call_count: 0,
        }
    }
}

impl Tier for MockL4 {
    fn id(&self) -> TierId {
        TierId::L4(L4ModelId(0))
    }

    fn estimate_cost(&self, _q: &Query) -> Option<TierEstimate> {
        Some(TierEstimate {
            joules: self.cost_joules,
            latency: std::time::Duration::from_millis(100),
            confidence_floor: self.confidence,
        })
    }

    fn try_answer(
        &mut self,
        _q: &Query,
        _budget: f64,
    ) -> Result<Answer, AnswerError> {
        self.call_count += 1;
        Ok(Answer {
            output: AnswerOutput::Text(self.answer_text.clone()),
            tier_used: self.id(),
            joules_spent: self.cost_joules,
            confidence: self.confidence,
            trace: ExecutionTrace::default(),
            verification: crate::verification::VerificationStatus::Resolved,
        })
    }
}

fn text_query(text: &str) -> Query {
    Query {
        input: QueryInput::Text(text.to_string()),
        budget: JouleBudget::standard(),
        quality: QualityFloor::chat(),
        context: ContextRef::fresh(),
        deadline: None,
    }
}

// ============================================================
// L0Cache unit tests
// ============================================================

#[test]
fn l0_empty_cache_refuses_all() {
    let mut cache = L0Cache::new();
    let q = text_query("hello");
    let a = cache.try_answer(&q, 1.0).unwrap();
    assert!(matches!(a.output, AnswerOutput::Refused(_)));
    assert_eq!(a.tier_used, TierId::L0);
    assert_eq!(cache.stats().hits, 0);
    assert_eq!(cache.stats().misses, 1);
}

#[test]
fn l0_put_then_get_hits() {
    let mut cache = L0Cache::new();
    let q = text_query("what is 2+2?");
    let answer = Answer {
        output: AnswerOutput::Text("4".to_string()),
        tier_used: TierId::L1(L1Primitive::Execute),
        joules_spent: 1e-7,
        confidence: 1.0,
        trace: ExecutionTrace::default(),
        verification: crate::verification::VerificationStatus::Resolved,
    };
    cache.put(&q, &answer);
    assert_eq!(cache.len(), 1);

    let a = cache.try_answer(&q, 1.0).unwrap();
    assert_eq!(a.output, AnswerOutput::Text("4".to_string()));
    assert_eq!(a.tier_used, TierId::L0);  // tier_used is L0, not the originator
    assert_eq!(a.confidence, 1.0);
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.stats().misses, 0);
}

#[test]
fn l0_same_text_different_context_misses() {
    let mut cache = L0Cache::new();
    let mut q1 = text_query("hello");
    cache.put(&q1, &Answer {
        output: AnswerOutput::Text("world".to_string()),
        tier_used: TierId::L4(L4ModelId(0)),
        joules_spent: 1.0,
        confidence: 0.9,
        trace: ExecutionTrace::default(),
        verification: crate::verification::VerificationStatus::Resolved,
    });

    // Change context fingerprint.
    q1.context.history_fingerprint.0[0] = 1;
    let a = cache.try_answer(&q1, 1.0).unwrap();
    assert!(matches!(a.output, AnswerOutput::Refused(_)),
        "different context fingerprint should miss");
}

#[test]
fn l0_hit_cost_is_independent_of_call_count() {
    // The cost model says lookup is O(1) in calls, not amortized.
    let mut cache = L0Cache::new();
    let q = text_query("test");
    cache.put(&q, &Answer {
        output: AnswerOutput::Text("answer".to_string()),
        tier_used: TierId::L4(L4ModelId(0)),
        joules_spent: 1.0,
        confidence: 0.9,
        trace: ExecutionTrace::default(),
        verification: crate::verification::VerificationStatus::Resolved,
    });
    let c1 = cache.try_answer(&q, 1.0).unwrap().joules_spent;
    let c2 = cache.try_answer(&q, 1.0).unwrap().joules_spent;
    let c100 = (0..100).map(|_| cache.try_answer(&q, 1.0).unwrap().joules_spent).last().unwrap();
    assert_eq!(c1, c2);
    assert_eq!(c1, c100);
}

#[test]
fn l0_cost_scales_with_input_length() {
    let cache = L0Cache::new();
    let short = text_query("hi");
    let long = text_query(&"x".repeat(10000));
    let short_est = cache.estimate_cost(&short).unwrap().joules;
    let long_est = cache.estimate_cost(&long).unwrap().joules;
    assert!(long_est > short_est,
        "long input should cost more to hash: short={:.3e}, long={:.3e}",
        short_est, long_est);
}

#[test]
fn l0_key_is_deterministic() {
    let q = text_query("deterministic query");
    let k1 = L0Cache::key_for(&q);
    let k2 = L0Cache::key_for(&q);
    let k3 = L0Cache::key_for(&q);
    assert_eq!(k1, k2);
    assert_eq!(k2, k3);
}

#[test]
fn l0_keys_differ_for_different_inputs() {
    let q1 = text_query("a");
    let q2 = text_query("b");
    assert_ne!(L0Cache::key_for(&q1), L0Cache::key_for(&q2));
}

// ============================================================
// Cascade walker tests
// ============================================================

#[test]
fn runtime_falls_through_empty_cache_to_l4() {
    let mut cascade = Cascade::new();
    // Runtime owns the L0 cache; tests don't register it.
    cascade.register(Box::new(MockL4::new(0.5, "Paris")));
    let mut rt = Runtime::new(cascade);

    let q = text_query("capital of France?");
    let a = rt.answer(q).unwrap();
    assert_eq!(a.output, AnswerOutput::Text("Paris".to_string()));
    assert_eq!(a.tier_used, TierId::L4(L4ModelId(0)));
    // Trace shows L0 refused (Inapplicable on miss), then L4 hit.
    assert_eq!(a.trace.attempts.len(), 2);
    assert_eq!(a.trace.attempts[0].tier, TierId::L0);
    assert!(matches!(a.trace.attempts[0].outcome, TraceOutcome::Refused(_)));
    assert_eq!(a.trace.attempts[1].tier, TierId::L4(L4ModelId(0)));
    assert_eq!(a.trace.attempts[1].outcome, TraceOutcome::Hit);
}

/// THE proof: same query twice, second is L0 hit, ~10^6× cheaper.
#[test]
fn second_query_hits_cache_and_is_orders_of_magnitude_cheaper() {
    let mock_l4_cost = 0.5;  // 500 mJ for L4 (realistic for a 7B local model)

    let mut cascade = Cascade::new();
    // Runtime auto-creates L0 internally; we only register L4.
    cascade.register(Box::new(MockL4::new(mock_l4_cost, "Paris")));
    let mut rt = Runtime::new(cascade);

    let q = text_query("capital of France?");

    // First call: L0 miss → falls through to L4. The runtime
    // auto-records the L4 answer to L0 for next time.
    let a1 = rt.answer(q.clone()).unwrap();
    assert_eq!(a1.tier_used, TierId::L4(L4ModelId(0)));
    let l4_cost = a1.joules_spent;
    assert!(l4_cost >= mock_l4_cost,
        "first call should cost at least L4's price: {:.3e}", l4_cost);

    // Second call: L0 hit.
    let a2 = rt.answer(q).unwrap();
    assert_eq!(a2.tier_used, TierId::L0,
        "second identical query must hit L0; got {:?}", a2.tier_used);
    assert_eq!(a2.output, AnswerOutput::Text("Paris".to_string()));

    let l0_cost = a2.joules_spent;

    // The actual claim: orders of magnitude cheaper. L4 ≈ 500 mJ, L0 ≈
    // tens of nJ → ratio ~10^7. We require at least 10^4.
    let ratio = l4_cost / l0_cost;
    assert!(ratio > 1e4,
        "L0 hit should be at least 10^4× cheaper than L4: \
         L4 cost {:.3e} J, L0 cost {:.3e} J, ratio {:.1e}×",
        l4_cost, l0_cost, ratio);
}

#[test]
fn runtime_with_only_l0_and_empty_cache_returns_budget_exhausted() {
    // No external tiers, but the runtime has its own L0 (which will
    // miss). Since L0 miss is treated as Inapplicable (not a refusal
    // we count), the runtime falls through with no tiers attempted.
    let cascade = Cascade::new();
    let mut rt = Runtime::new(cascade);

    let q = text_query("anything");
    let result = rt.answer(q);
    // With no external tiers, the runtime has nothing to dispatch to
    // after L0 misses. We get BudgetExhausted (attempted_tiers empty).
    assert!(matches!(result, Err(AnswerError::BudgetExhausted { .. })),
        "L0 miss with no fallback should produce BudgetExhausted; got {:?}",
        result);
}

#[test]
fn runtime_respects_budget_and_skips_unaffordable_tier() {
    let mut cascade = Cascade::new();
    cascade.register(Box::new(MockL4::new(10.0, "expensive answer")));
    let mut rt = Runtime::new(cascade);

    // Budget is too small for L4.
    let q = Query {
        input: QueryInput::Text("x".to_string()),
        budget: JouleBudget { hard_limit: 1.0, soft_target: 0.1 },
        quality: QualityFloor::chat(),
        context: ContextRef::fresh(),
        deadline: None,
    };

    let result = rt.answer(q);
    assert!(matches!(result, Err(AnswerError::BudgetExhausted { .. })),
        "budget too small for L4 should produce BudgetExhausted; got {:?}",
        result);
}

#[test]
fn runtime_respects_quality_floor() {
    let mut cascade = Cascade::new();
    // L4 with confidence_floor = 0.5
    let mut weak_l4 = MockL4::new(0.5, "uncertain answer");
    weak_l4.confidence = 0.5;
    cascade.register(Box::new(weak_l4));
    let mut rt = Runtime::new(cascade);

    // Query demanding 0.9 confidence — weak L4 should be skipped.
    let q = Query {
        input: QueryInput::Text("x".to_string()),
        budget: JouleBudget::standard(),
        quality: QualityFloor { min_confidence: 0.9, accept_partial: false },
        context: ContextRef::fresh(),
        deadline: None,
    };

    let result = rt.answer(q);
    assert!(matches!(result, Err(AnswerError::BudgetExhausted { .. })),
        "quality floor should skip the weak tier; got {:?}", result);
}

// ============================================================
// Demo
// ============================================================

#[test]
fn cascade_demo() {
    println!("\n=== R1: L0 cache fires in the cascade ===\n");

    let l4_cost = 0.5;  // 500 mJ
    let mut cascade = Cascade::new();
    cascade.register(Box::new(MockL4::new(l4_cost, "Paris")));
    let mut rt = Runtime::new(cascade);

    let q = text_query("capital of France?");

    println!("Query: \"capital of France?\"");
    println!("Budget: 1 J hard / 100 mJ soft");
    println!();

    println!("--- First call ---");
    let a1 = rt.answer(q.clone()).unwrap();
    print_trace(&a1);

    println!("\n--- Second call (identical query) ---");
    let a2 = rt.answer(q).unwrap();
    print_trace(&a2);

    let ratio = a1.joules_spent / a2.joules_spent;
    println!("\nL0 hit is {:.1e}× cheaper than the L4 dispatch.", ratio);
    println!("Same answer, near-zero joule cost. This is the cascade.");
}

fn print_trace(a: &Answer) {
    println!("  tier_used:    {:?}", a.tier_used);
    println!("  answer:       {:?}", a.output);
    println!("  joules:       {:.3e} J", a.joules_spent);
    println!("  confidence:   {}", a.confidence);
    println!("  attempts:");
    for entry in &a.trace.attempts {
        println!("    {:<20} {:?}  ({:.3e} J)",
            format!("{:?}", entry.tier), entry.outcome, entry.joules);
    }
}
