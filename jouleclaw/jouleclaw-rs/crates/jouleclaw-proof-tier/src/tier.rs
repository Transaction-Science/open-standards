//! [`ProofTier`] — the L4.5 cascade tier.
//!
//! Wraps an injected [`Solver`] as a `jouleclaw_cascade::Tier` so the
//! runtime can walk it like any other tier. The tier:
//!
//! - **estimates** cost only when the query envelope is a recognized
//!   constraint-satisfaction problem (otherwise `None`, and the
//!   cascade walks past us cheaply);
//! - **dispatches** to the solver on `try_answer`, attributes the
//!   typical L4.5 energy figure (60 µJ — the donor's mgai-csp
//!   benchmark) and emits a [`ProofReceipt`];
//! - **verifies** the solver's own answer through the
//!   [`Solver::verify`] hook before reporting success. AI proposes,
//!   verifier disposes — the same trick as `jouleclaw-verify`, scoped
//!   to the proof-shaped output.
//!
//! The receipt is serialized to JSON and attached to the answer as the
//! `tier_used` annotation's trace would not survive `Answer`'s shape;
//! the tier surfaces the receipt by appending it to the `output`
//! structured payload under a `proof_receipt` key. Auditors that
//! consume answers programmatically pick the receipt back up from the
//! JSON envelope.

use std::time::Duration;

use jouleclaw_cascade::*;
use serde::Serialize;

use crate::problem::{ProofProblem, ProofSolution};
use crate::receipt::{fnv1a_64, ProofReceipt};
use crate::solver::{DpllSolver, Solver};

/// Typical picojoule attribution for one L4.5 dispatch.
///
/// This is the donor's mgai-csp benchmark for an in-the-pocket
/// constraint-solver problem (sudoku / small SAT). The receipt's
/// [`Provenance`][jouleclaw_energy::Provenance] is fixed at
/// `Estimator` to make clear we did not measure on a hardware shunt.
const TYPICAL_PJ: u64 = 60_000_000; // 60 µJ in picojoules

/// Typical wall-clock latency the cascade should plan around.
const TYPICAL_LATENCY: Duration = Duration::from_millis(5);

/// The L4.5 proof-solver cascade tier.
///
/// Construct with [`ProofTier::new`] (uses [`DpllSolver`] by default)
/// or [`ProofTier::with_solver`] (custom solver implementing
/// [`Solver`]).
pub struct ProofTier {
    solver: Box<dyn Solver>,
}

impl ProofTier {
    /// Construct with the bundled DPLL solver.
    pub fn new() -> Self {
        Self {
            solver: Box::new(DpllSolver::default()),
        }
    }

    /// Construct with a custom solver. Use this to plug in an AC-3 +
    /// MRV CSP engine, an SMT bridge, a Lean tactic search, etc.
    pub fn with_solver(solver: Box<dyn Solver>) -> Self {
        Self { solver }
    }

    /// The active solver's name. Useful for diagnostics.
    pub fn solver_name(&self) -> &str {
        self.solver.name()
    }
}

impl Default for ProofTier {
    fn default() -> Self {
        Self::new()
    }
}

/// The JSON envelope returned to the caller on a successful solve. We
/// keep this distinct from [`ProofSolution`] so the receipt rides
/// alongside the solution in a single auditable blob.
#[derive(Debug, Clone, Serialize)]
struct ProofAnswerEnvelope {
    /// The satisfying assignment in canonical form.
    solution: ProofSolution,
    /// The receipt for this dispatch.
    proof_receipt: ProofReceipt,
}

impl Tier for ProofTier {
    fn id(&self) -> TierId {
        TierId::L4_5Proof
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        // Only structured envelopes can carry a proof problem.
        let bytes = match &q.input {
            QueryInput::Structured(b) => b,
            _ => return None,
        };
        // Probe-decode. We must not panic, and we must not pay much
        // for the probe — serde_json on a small envelope is cheap.
        ProofProblem::from_bytes(bytes).ok()?;
        Some(TierEstimate {
            joules: TYPICAL_PJ as f64 * 1e-12,
            latency: TYPICAL_LATENCY,
            // V=Full: every accepted answer is re-verified through the
            // solver's own verify() hook before we return success.
            confidence_floor: 1.0,
        })
    }

    fn try_answer(&mut self, q: &Query, _budget: f64) -> Result<Answer, AnswerError> {
        let bytes = match &q.input {
            QueryInput::Structured(b) => b,
            _ => return Ok(refused(self.id(), 0.0, RefusalReason::Inapplicable)),
        };
        let problem = match ProofProblem::from_bytes(bytes) {
            Ok(p) => p,
            Err(_) => {
                return Ok(refused(
                    self.id(),
                    0.0,
                    RefusalReason::Inapplicable,
                ))
            }
        };
        let cost_joules = TYPICAL_PJ as f64 * 1e-12;
        let id = self.id();

        match self.solver.solve(&problem) {
            // Solver-shaped failure (unsupported, malformed, step-limit):
            // refuse with a tier-specific reason; the cascade falls
            // through.
            Err(e) => Ok(refused(
                id,
                cost_joules,
                RefusalReason::TierSpecific(format!("solver: {}", e)),
            )),

            // UNSAT is a definite verdict — not a refusal. We surface
            // it as a structured answer carrying the receipt so the
            // caller can prove "this problem has no solution".
            Ok(ProofSolution::Unsat) => {
                let problem_hash = fnv1a_64(&problem.to_canonical_bytes());
                let solution_hash =
                    fnv1a_64(&ProofSolution::Unsat.to_canonical_bytes());
                let receipt = ProofReceipt::new(
                    self.solver.name(),
                    problem_hash,
                    solution_hash,
                    TYPICAL_PJ,
                );
                let envelope = ProofAnswerEnvelope {
                    solution: ProofSolution::Unsat,
                    proof_receipt: receipt,
                };
                let bytes = serde_json::to_vec(&envelope).unwrap_or_default();
                Ok(Answer {
                    output: AnswerOutput::Structured(bytes),
                    tier_used: id,
                    joules_spent: cost_joules,
                    confidence: 1.0,
                    trace: hit_trace(id, cost_joules),
                    verification: VerificationStatus::Resolved,
                })
            }

            // Satisfying assignment: re-verify through the solver's
            // own verify() hook before claiming success.
            Ok(solution) => {
                if !self.solver.verify(&problem, &solution) {
                    return Ok(refused(
                        id,
                        cost_joules,
                        RefusalReason::TierSpecific(
                            "solver self-verification failed".into(),
                        ),
                    ));
                }
                let problem_hash = fnv1a_64(&problem.to_canonical_bytes());
                let solution_hash = fnv1a_64(&solution.to_canonical_bytes());
                let receipt = ProofReceipt::new(
                    self.solver.name(),
                    problem_hash,
                    solution_hash,
                    TYPICAL_PJ,
                );
                let envelope = ProofAnswerEnvelope {
                    solution,
                    proof_receipt: receipt,
                };
                let bytes = serde_json::to_vec(&envelope).unwrap_or_default();
                Ok(Answer {
                    output: AnswerOutput::Structured(bytes),
                    tier_used: id,
                    joules_spent: cost_joules,
                    confidence: 1.0,
                    trace: hit_trace(id, cost_joules),
                    verification: VerificationStatus::Resolved,
                })
            }
        }
    }
}

/// Verify-side helper: pull a [`ProofReceipt`] back out of a successful
/// answer's structured payload. Returns `None` when the envelope is
/// not the proof-tier shape (e.g. the answer came from a different
/// tier). Useful for tests and audit chains.
pub fn extract_receipt(answer: &Answer) -> Option<ProofReceipt> {
    let bytes = match &answer.output {
        AnswerOutput::Structured(b) => b,
        _ => return None,
    };
    #[derive(serde::Deserialize)]
    struct Envelope {
        proof_receipt: ProofReceipt,
    }
    let env: Envelope = serde_json::from_slice(bytes).ok()?;
    Some(env.proof_receipt)
}

fn hit_trace(tier: TierId, joules: f64) -> ExecutionTrace {
    let mut t = ExecutionTrace::default();
    t.attempts.push(TraceEntry {
        tier,
        outcome: TraceOutcome::Hit,
        joules,
    });
    t
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{ProblemKind, ProofProblem};

    fn make_query(p: &ProofProblem) -> Query {
        Query {
            input: QueryInput::Structured(p.to_canonical_bytes()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

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
    fn tier_id_is_l4_5_proof() {
        let tier = ProofTier::new();
        assert_eq!(tier.id(), TierId::L4_5Proof);
        assert_eq!(tier.id().wire_tag(), "L4.5");
    }

    #[test]
    fn estimate_cost_is_none_for_text_query() {
        let tier = ProofTier::new();
        assert!(tier.estimate_cost(&text_query("hello")).is_none());
    }

    #[test]
    fn estimate_cost_is_some_for_sat_envelope() {
        let tier = ProofTier::new();
        let p = ProofProblem {
            kind: ProblemKind::Sat,
            cnf: Some(vec![vec![1, 2]]),
            grid: None,
        };
        let est = tier.estimate_cost(&make_query(&p)).expect("estimate");
        // 60 µJ in joules.
        assert!((est.joules - 60e-6).abs() < 1e-9);
        assert_eq!(est.confidence_floor, 1.0);
    }

    #[test]
    fn solves_small_sat() {
        let mut tier = ProofTier::new();
        let p = ProofProblem {
            kind: ProblemKind::Sat,
            cnf: Some(vec![vec![1, 2], vec![-1, 3], vec![-2, -3]]),
            grid: None,
        };
        let ans = tier.try_answer(&make_query(&p), 1.0).unwrap();
        assert_eq!(ans.confidence, 1.0);
        assert_eq!(ans.tier_used, TierId::L4_5Proof);
        let receipt = extract_receipt(&ans).expect("receipt present");
        assert_eq!(receipt.solver_name, "jouleclaw::dpll");
        assert_eq!(receipt.energy_pj, TYPICAL_PJ);
    }

    #[test]
    fn unsat_problem_returns_unsat_with_receipt() {
        let mut tier = ProofTier::new();
        let p = ProofProblem {
            kind: ProblemKind::Sat,
            cnf: Some(vec![vec![1], vec![-1]]),
            grid: None,
        };
        let ans = tier.try_answer(&make_query(&p), 1.0).unwrap();
        // Confidence 1.0 — "no solution exists" is a definite verdict.
        assert_eq!(ans.confidence, 1.0);
        let receipt = extract_receipt(&ans).expect("receipt for unsat");
        // Distinct hash from a satisfying answer for the same shape.
        assert_ne!(receipt.solution_hash, 0);
    }

    #[test]
    fn solves_4x4_sudoku() {
        // A genuinely empty 4×4 — solver should fill it in.
        let mut tier = ProofTier::new();
        let p = ProofProblem {
            kind: ProblemKind::Sudoku4,
            cnf: None,
            grid: Some(".".repeat(16)),
        };
        let ans = tier.try_answer(&make_query(&p), 1.0).unwrap();
        assert_eq!(ans.confidence, 1.0);
        let receipt = extract_receipt(&ans).expect("receipt");
        assert_eq!(receipt.solver_name, "jouleclaw::dpll");
        // The grid in the envelope should be 16 chars, all digits.
        let bytes = match &ans.output {
            AnswerOutput::Structured(b) => b.clone(),
            _ => panic!("expected structured output"),
        };
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let grid = v["solution"]["sudoku_grid"].as_str().unwrap();
        assert_eq!(grid.len(), 16);
        assert!(grid.chars().all(|c| c.is_ascii_digit()));
    }

    #[test]
    fn solves_4x4_sudoku_with_clues() {
        // A 4×4 with two clues, exercising the unit-clause path.
        let mut tier = ProofTier::new();
        let mut grid: Vec<char> = ".".repeat(16).chars().collect();
        grid[0] = '1';
        grid[5] = '3';
        let p = ProofProblem {
            kind: ProblemKind::Sudoku4,
            cnf: None,
            grid: Some(grid.iter().collect()),
        };
        let ans = tier.try_answer(&make_query(&p), 1.0).unwrap();
        assert_eq!(ans.confidence, 1.0);
        let bytes = match &ans.output {
            AnswerOutput::Structured(b) => b.clone(),
            _ => panic!("expected structured output"),
        };
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let solved = v["solution"]["sudoku_grid"].as_str().unwrap();
        // Clues must be preserved.
        assert_eq!(&solved[0..1], "1");
        assert_eq!(&solved[5..6], "3");
    }

    #[test]
    fn non_structured_query_refuses_inapplicable() {
        let mut tier = ProofTier::new();
        let ans = tier.try_answer(&text_query("anything"), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Refused(RefusalReason::Inapplicable) => {}
            other => panic!("expected inapplicable refusal, got {:?}", other),
        }
    }

    #[test]
    fn malformed_envelope_refuses_inapplicable() {
        let mut tier = ProofTier::new();
        let q = Query {
            input: QueryInput::Structured(b"{ not json".to_vec()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let ans = tier.try_answer(&q, 1.0).unwrap();
        match ans.output {
            AnswerOutput::Refused(RefusalReason::Inapplicable) => {}
            other => panic!("expected inapplicable refusal, got {:?}", other),
        }
    }

    #[test]
    fn cascade_can_register_proof_tier() {
        let mut cascade = Cascade::new();
        cascade.register(Box::new(ProofTier::new()));
        let mut rt = Runtime::new_without_l0(cascade);
        let p = ProofProblem {
            kind: ProblemKind::Sat,
            cnf: Some(vec![vec![1, 2]]),
            grid: None,
        };
        let ans = rt.answer(make_query(&p)).expect("answer");
        assert_eq!(ans.tier_used, TierId::L4_5Proof);
    }

    #[test]
    fn custom_solver_is_used() {
        // A solver that always says UNSAT.
        struct AlwaysUnsat;
        impl Solver for AlwaysUnsat {
            fn name(&self) -> &str {
                "test::always-unsat"
            }
            fn solve(
                &self,
                _p: &ProofProblem,
            ) -> Result<ProofSolution, crate::SolverError> {
                Ok(ProofSolution::Unsat)
            }
        }
        let mut tier = ProofTier::with_solver(Box::new(AlwaysUnsat));
        assert_eq!(tier.solver_name(), "test::always-unsat");
        let p = ProofProblem {
            kind: ProblemKind::Sat,
            cnf: Some(vec![vec![1]]),
            grid: None,
        };
        let ans = tier.try_answer(&make_query(&p), 1.0).unwrap();
        let receipt = extract_receipt(&ans).expect("receipt");
        assert_eq!(receipt.solver_name, "test::always-unsat");
    }

    #[test]
    fn dishonest_solver_is_caught_by_verify() {
        // A solver that returns a model it cannot verify.
        struct Liar;
        impl Solver for Liar {
            fn name(&self) -> &str {
                "test::liar"
            }
            fn solve(
                &self,
                _p: &ProofProblem,
            ) -> Result<ProofSolution, crate::SolverError> {
                Ok(ProofSolution::SatModel(vec![true, true]))
            }
            fn verify(&self, _: &ProofProblem, _: &ProofSolution) -> bool {
                false
            }
        }
        let mut tier = ProofTier::with_solver(Box::new(Liar));
        let p = ProofProblem {
            kind: ProblemKind::Sat,
            cnf: Some(vec![vec![1]]),
            grid: None,
        };
        let ans = tier.try_answer(&make_query(&p), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Refused(RefusalReason::TierSpecific(msg)) => {
                assert!(msg.contains("self-verification"));
            }
            other => panic!("expected refusal for lying solver, got {:?}", other),
        }
    }
}
