//! R37 — the calibration loop is now *closed*.
//!
//! Before R37 the Runtime recorded estimate-vs-actual per coord cell
//! but never used it: budget/routing decisions ran on the tier's raw
//! `estimate_cost`. R37 scales the estimate by the learned μ-correction
//! before the budget check, so a tier that consistently under-reports
//! gets its future estimates corrected from observed reality.
//!
//! This test proves the behavioural difference: a tier that estimates
//! 5× too low is, after enough samples, correctly skipped when the
//! budget only covers the *honest* (corrected) cost — something the
//! open loop could never do.

use jouleclaw_cascade::coord::*;
use jouleclaw_cascade::*;
use std::time::Duration;

/// A tier that always claims 1 nJ but actually spends 5 nJ — a
/// consistent 5× under-estimator. Distinct coord cell so per-cell
/// calibration applies.
struct UnderEstimator;

impl Tier for UnderEstimator {
    fn id(&self) -> TierId {
        TierId::L1(L1Primitive::Execute)
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        match &q.input {
            QueryInput::Text(_) => Some(TierEstimate {
                joules: 1e-9, // claims 1 nJ
                latency: Duration::from_nanos(1),
                confidence_floor: 1.0,
            }),
            _ => None,
        }
    }

    fn try_answer(&mut self, _q: &Query, _b: f64) -> Result<Answer, AnswerError> {
        let actual = 5e-9; // truth: 5 nJ — 5× the estimate
        let mut trace = ExecutionTrace::default();
        trace.attempts.push(TraceEntry {
            tier: self.id(),
            outcome: TraceOutcome::Hit,
            joules: actual,
        });
        Ok(Answer {
            output: AnswerOutput::Text("ok".into()),
            tier_used: self.id(),
            joules_spent: actual,
            confidence: 1.0,
            trace,
            verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
        })
    }

    fn coord(&self) -> Option<Coord> {
        Some(Coord::new(
            Zone::Z1,
            Entity::Reactive,
            Thermo::L1_Measure,
            Interface::Tokens,
            Verify::Full,
            Encoding::Facts,
        ))
    }
}

fn ask(rt: &mut Runtime, budget: JouleBudget) -> Result<Answer, AnswerError> {
    rt.answer(Query {
        input: QueryInput::Text("x".into()),
        budget,
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    })
}

#[test]
fn learned_mu_converges_to_the_true_ratio() {
    let mut cascade = Cascade::new();
    cascade.register(Box::new(UnderEstimator));
    let mut rt = Runtime::new_without_l0(cascade);

    let coord = UnderEstimator.coord().unwrap();
    // Cold start: no samples → μ = 1.0 (loop must not perturb startup).
    assert_eq!(rt.calibration().learned_mu(&coord), 1.0);

    // Warm it up with a generous budget so it always dispatches.
    for _ in 0..6 {
        let _ = ask(&mut rt, JouleBudget::expensive());
    }
    let mu = rt.calibration().learned_mu(&coord);
    // actual/estimate = 5e-9 / 1e-9 = 5.0.
    assert!(
        (mu - 5.0).abs() < 0.5,
        "learned μ should converge to ~5, got {}",
        mu
    );
}

#[test]
fn closed_loop_skips_a_tier_the_open_loop_would_have_run() {
    let mut cascade = Cascade::new();
    cascade.register(Box::new(UnderEstimator));
    let mut rt = Runtime::new_without_l0(cascade);

    // Train calibration (6 dispatches under a generous budget).
    for _ in 0..6 {
        let _ = ask(&mut rt, JouleBudget::expensive());
    }
    assert!(rt.calibration().learned_mu(&UnderEstimator.coord().unwrap()) > 4.0);

    // Budget = 2 nJ. Raw estimate (1 nJ) fits → the OPEN loop would
    // dispatch and overspend. Corrected estimate (1 nJ × ~5 = ~5 nJ)
    // exceeds 2 nJ → the CLOSED loop must skip and surface no-tier.
    let r = ask(
        &mut rt,
        JouleBudget { hard_limit: 2e-9, soft_target: 2e-9 },
    );
    match r {
        Err(AnswerError::BudgetExhausted { .. })
        | Err(AnswerError::NoTierSatisfied { .. }) => { /* correct: skipped */ }
        Ok(a) => panic!(
            "closed loop must skip the under-estimator at a 2 nJ budget; \
             instead it ran and spent {:.3e} J",
            a.joules_spent
        ),
        Err(e) => panic!("unexpected error: {:?}", e),
    }

    // Sanity: with a budget above the *corrected* cost it still runs.
    let ok = ask(&mut rt, JouleBudget::expensive());
    assert!(ok.is_ok());
}
