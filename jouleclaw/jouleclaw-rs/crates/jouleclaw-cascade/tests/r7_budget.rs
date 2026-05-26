//! R7 tests — budget enforcement and tier calibration.
//!
//! What R7 changes:
//!   1. Tiers that overrun their budget-cap produce `BudgetExhausted`
//!      rather than silently consuming the overage.
//!   2. Every tier dispatch records `(estimate, actual)` for
//!      calibration.
//!   3. `Runtime::calibration()` exposes the report.

use jouleclaw_cascade::*;
use std::time::Duration;

fn text(s: &str) -> Query {
    Query {
        input: QueryInput::Text(s.to_string()),
        budget: JouleBudget::standard(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    }
}

// ============================================================
// Mock tiers with controllable cost behavior
// ============================================================

/// A tier that estimates one cost but consumes a configurable other
/// cost. Used to test budget-overrun handling and calibration recording.
struct OvershootingTier {
    id: TierId,
    estimated_joules: f64,
    actual_joules: f64,
}

impl Tier for OvershootingTier {
    fn id(&self) -> TierId { self.id }
    fn estimate_cost(&self, _q: &Query) -> Option<TierEstimate> {
        Some(TierEstimate {
            joules: self.estimated_joules,
            latency: Duration::from_micros(10),
            confidence_floor: 0.9,
        })
    }
    fn try_answer(&mut self, _q: &Query, _b: f64) -> Result<Answer, AnswerError> {
        Ok(Answer {
            output: AnswerOutput::Text("done".into()),
            tier_used: self.id,
            joules_spent: self.actual_joules,
            confidence: 0.9,
            trace: ExecutionTrace::default(),
            verification: crate::verification::VerificationStatus::Resolved,
        })
    }
}

struct CheapHonestTier {
    id: TierId,
    cost: f64,
    text: String,
}

impl Tier for CheapHonestTier {
    fn id(&self) -> TierId { self.id }
    fn estimate_cost(&self, _q: &Query) -> Option<TierEstimate> {
        Some(TierEstimate {
            joules: self.cost,
            latency: Duration::from_micros(1),
            confidence_floor: 1.0,
        })
    }
    fn try_answer(&mut self, _q: &Query, _b: f64) -> Result<Answer, AnswerError> {
        Ok(Answer {
            output: AnswerOutput::Text(self.text.clone()),
            tier_used: self.id,
            joules_spent: self.cost,
            confidence: 1.0,
            trace: ExecutionTrace::default(),
            verification: crate::verification::VerificationStatus::Resolved,
        })
    }
}

// ============================================================
// Calibration recording
// ============================================================

#[test]
fn calibration_records_estimate_and_actual() {
    let mut cascade = Cascade::new();
    cascade.register(Box::new(CheapHonestTier {
        id: TierId::L1(L1Primitive::Execute),
        cost: 1e-7, text: "ok".into(),
    }));
    let mut rt = Runtime::new(cascade);
    let _ = rt.answer(text("test")).unwrap();
    let cal = rt.calibration();
    let entry = cal.per_tier.get(&TierId::L1(L1Primitive::Execute)).unwrap();
    assert_eq!(entry.samples, 1);
    assert_eq!(entry.total_estimated, 1e-7);
    assert_eq!(entry.total_actual, 1e-7);
    assert!((entry.mean_ratio() - 1.0).abs() < 1e-9,
        "honest tier should have mean ratio 1.0; got {}", entry.mean_ratio());
}

#[test]
fn calibration_aggregates_over_many_calls() {
    let mut cascade = Cascade::new();
    cascade.register(Box::new(CheapHonestTier {
        id: TierId::L1(L1Primitive::Execute),
        cost: 1e-7, text: "ok".into(),
    }));
    let mut rt = Runtime::new(cascade);
    for i in 0..20 {
        rt.answer(text(&format!("q{}", i))).unwrap();
    }
    let cal = rt.calibration();
    let entry = cal.per_tier.get(&TierId::L1(L1Primitive::Execute)).unwrap();
    assert_eq!(entry.samples, 20);
    assert!((entry.mean_ratio() - 1.0).abs() < 1e-9);
}

#[test]
fn calibration_detects_underestimating_tier() {
    // A tier that says it'll cost 1 nJ but actually spends 1 mJ. Wildly
    // dishonest — the budget should give 1 J standard, so the tier
    // still fits, but calibration should flag it.
    let mut cascade = Cascade::new();
    cascade.register(Box::new(OvershootingTier {
        id: TierId::L4(L4ModelId(0)),
        estimated_joules: 1e-9,
        actual_joules: 1e-3,
    }));
    let mut rt = Runtime::new(cascade);
    for _ in 0..10 {
        let q = Query {
            input: QueryInput::Text(format!("q-{}", rand_str())),
            budget: JouleBudget::standard(),
            quality: QualityFloor::chat(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let _ = rt.answer(q);
    }
    let cal = rt.calibration();
    let entry = cal.per_tier.get(&TierId::L4(L4ModelId(0))).unwrap();
    assert!(entry.mean_ratio() > 100.0,
        "underestimating tier should have mean_ratio >> 1; got {}",
        entry.mean_ratio());
    let dishonest = cal.dishonest_tiers(0.2);
    assert!(!dishonest.is_empty());
}

fn rand_str() -> String {
    // Deterministic but distinct strings to avoid L0 caching.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("test-query-{}", n)
}

// ============================================================
// Budget enforcement
// ============================================================

#[test]
fn budget_overshoot_triggers_budget_exhausted_error() {
    // Tier estimates 1 µJ, actually spends 1 J — way over the query's
    // 1 mJ budget. The runtime should catch this and return
    // BudgetExhausted.
    let mut cascade = Cascade::new();
    cascade.register(Box::new(OvershootingTier {
        id: TierId::L4(L4ModelId(0)),
        estimated_joules: 1e-6,   // says it's cheap
        actual_joules: 1.0,       // actually expensive
    }));
    let mut rt = Runtime::new(cascade);

    let q = Query {
        input: QueryInput::Text("test".to_string()),
        budget: JouleBudget { hard_limit: 1e-3, soft_target: 1e-4 },  // 1 mJ
        quality: QualityFloor::chat(),
        context: ContextRef::fresh(),
        deadline: None,
    };

    let result = rt.answer(q);
    assert!(matches!(result, Err(AnswerError::BudgetExhausted { .. })),
        "tier overrun should produce BudgetExhausted; got {:?}", result);

    // Calibration should have recorded a violation.
    let cal = rt.calibration();
    let entry = cal.per_tier.get(&TierId::L4(L4ModelId(0))).unwrap();
    assert_eq!(entry.budget_violations, 1);
}

#[test]
fn tier_within_budget_still_works() {
    // Tier estimates 1 µJ, actually spends 1 µJ. Budget is 1 mJ. Fine.
    let mut cascade = Cascade::new();
    cascade.register(Box::new(OvershootingTier {
        id: TierId::L1(L1Primitive::Execute),
        estimated_joules: 1e-6,
        actual_joules: 1e-6,
    }));
    let mut rt = Runtime::new(cascade);

    let q = Query {
        input: QueryInput::Text("test".to_string()),
        budget: JouleBudget { hard_limit: 1e-3, soft_target: 1e-4 },
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    };

    let result = rt.answer(q).unwrap();
    assert_eq!(result.output, AnswerOutput::Text("done".into()));
    let cal = rt.calibration();
    let entry = cal.per_tier.get(&TierId::L1(L1Primitive::Execute)).unwrap();
    assert_eq!(entry.budget_violations, 0);
}

#[test]
fn small_overshoot_within_budget_is_allowed() {
    // Tier estimates 1 µJ, actually spends 10 µJ (10× over estimate).
    // Budget is 1 mJ — so it still fits comfortably. The runtime
    // records calibration drift but doesn't fail the query.
    let mut cascade = Cascade::new();
    cascade.register(Box::new(OvershootingTier {
        id: TierId::L1(L1Primitive::Execute),
        estimated_joules: 1e-6,
        actual_joules: 1e-5,
    }));
    let mut rt = Runtime::new(cascade);

    let q = Query {
        input: QueryInput::Text("test".to_string()),
        budget: JouleBudget { hard_limit: 1e-3, soft_target: 1e-4 },
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    };

    let result = rt.answer(q).unwrap();
    assert_eq!(result.output, AnswerOutput::Text("done".into()));

    let cal = rt.calibration();
    let entry = cal.per_tier.get(&TierId::L1(L1Primitive::Execute)).unwrap();
    assert!(entry.mean_ratio() > 5.0);   // 10x ratio
    assert_eq!(entry.budget_violations, 0);
}

#[test]
fn cumulative_budget_exhausted_after_walking_failed_tiers() {
    // Each tier in turn spends almost the full budget, but refuses.
    // After two refusals the budget is exhausted; the third tier
    // should be skipped.
    let mut cascade = Cascade::new();

    struct ExpensiveRefuser { id: TierId, cost: f64 }
    impl Tier for ExpensiveRefuser {
        fn id(&self) -> TierId { self.id }
        fn estimate_cost(&self, _q: &Query) -> Option<TierEstimate> {
            Some(TierEstimate {
                joules: self.cost, latency: Duration::from_micros(1),
                confidence_floor: 1.0,
            })
        }
        fn try_answer(&mut self, _q: &Query, _b: f64) -> Result<Answer, AnswerError> {
            Ok(Answer {
                output: AnswerOutput::Refused(RefusalReason::TierSpecific("nope".into())),
                tier_used: self.id, joules_spent: self.cost,
                confidence: 0.0, trace: ExecutionTrace::default(),
                verification: crate::verification::VerificationStatus::Resolved,
            })
        }
    }

    cascade.register(Box::new(ExpensiveRefuser {
        id: TierId::L1(L1Primitive::Execute), cost: 0.4,
    }));
    cascade.register(Box::new(ExpensiveRefuser {
        id: TierId::L1(L1Primitive::Regex), cost: 0.4,
    }));
    cascade.register(Box::new(ExpensiveRefuser {
        id: TierId::L1(L1Primitive::TemplateFill), cost: 0.4,
    }));
    let mut rt = Runtime::new(cascade);

    let q = Query {
        input: QueryInput::Text("test".to_string()),
        budget: JouleBudget { hard_limit: 1.0, soft_target: 0.5 },
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    };

    let result = rt.answer(q);
    // After the first two refusals (spending 0.8 J), the third tier's
    // 0.4 J estimate exceeds the remaining 0.2 J budget. The runtime
    // should skip it and return NoTierSatisfied (refusals) or
    // BudgetExhausted (if it interprets the unaffordable third as
    // budget exhaustion).
    match result {
        Err(AnswerError::NoTierSatisfied { refusals }) => {
            // Two refusals recorded; third was skipped.
            assert_eq!(refusals.len(), 2);
        }
        Err(AnswerError::BudgetExhausted { .. }) => {
            // Acceptable alternative outcome.
        }
        other => panic!("expected refusals/budget; got {:?}", other),
    }
}

// ============================================================
// Demo
// ============================================================

#[test]
fn r7_calibration_demo() {
    println!("\n=== R7: budget enforcement + tier calibration ===\n");

    // Each tier here only fires when its specific cascade is active.
    // We'll run three separate runtimes, one per tier-honesty case.

    fn run_tier(label: &str, est: f64, actual: f64, tid: TierId, budget: JouleBudget) -> CalibrationReport {
        let mut cascade = Cascade::new();
        cascade.register(Box::new(OvershootingTier {
            id: tid, estimated_joules: est, actual_joules: actual,
        }));
        let mut rt = Runtime::new(cascade);
        for i in 0..20 {
            let q = Query {
                input: QueryInput::Text(format!("{}-q{}", label, i)),
                budget,
                quality: QualityFloor::any(),
                context: ContextRef::fresh(),
                deadline: None,
            };
            let _ = rt.answer(q);
        }
        rt.calibration().clone()
    }

    let exec_cal = run_tier(
        "exec", 1e-6, 1e-6, TierId::L1(L1Primitive::Execute),
        JouleBudget::standard());
    let regex_cal = run_tier(
        "regex", 1e-6, 5e-6, TierId::L1(L1Primitive::Regex),
        JouleBudget::standard());
    let l4_cal = run_tier(
        "l4", 1.0, 1e-2, TierId::L4(L4ModelId(0)),
        JouleBudget::expensive());

    // Combine summaries for printing.
    let mut combined = CalibrationReport::default();
    for (tid, e) in exec_cal.per_tier { combined.per_tier.insert(tid, e); }
    for (tid, e) in regex_cal.per_tier { combined.per_tier.insert(tid, e); }
    for (tid, e) in l4_cal.per_tier { combined.per_tier.insert(tid, e); }

    println!("Calibration report after 20 queries per tier:");
    for line in combined.summary() {
        println!("  {}", line);
    }
    println!();

    let dishonest = combined.dishonest_tiers(0.2);
    println!("Tiers with mean_ratio drifting >20% from 1.0:");
    for (tid, ratio) in &dishonest {
        let direction = if *ratio > 1.0 { "UNDERESTIMATING" } else { "OVERESTIMATING" };
        println!("  {:<25} ratio={:.3}  ({})",
            format!("{:?}", tid), ratio, direction);
    }
    println!();
    println!("L1::Execute is honest — keep it as is.");
    println!("L1::Regex understates 5×  — bump its c_per_byte constant up.");
    println!("L4 overstates ~100×       — its cost model is too conservative.");
    println!();
    println!("This data feeds future per-tier cost-model recalibration.");
    println!("A tier whose estimates drift triggers a calibration cycle");
    println!("that updates the underlying constants.");
}
