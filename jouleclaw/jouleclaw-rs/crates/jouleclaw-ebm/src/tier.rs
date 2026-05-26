//! `EbmTier` — cascade tier for energy-based constraint satisfaction.
//!
//! Coordinate:
//!
//!   Z = Z1           (crisp constraints, deterministic search)
//!   E = Reactive
//!   T = L1_Measure   (per-decision cost is concrete arithmetic)
//!   I = Tokens       (text in: "sudoku <81 chars>"; text out: solved board)
//!   V = Full         (output is verifiable against the constraint set)
//!   R = WorldModel   (the energy function encodes the rules)
//!   P = { Parse, Compose, Arithmetic }  (search composes legal moves)
//!
//! Query protocol (R34.0):
//!
//!   "sudoku <81 chars with digits 1-9 / 0 / . for empty>"
//!     → backtracking solve; returns the completed board as text
//!
//! R34.1 will add more EBM problem shapes (SAT, scheduling, layout)
//! behind the same tier.

use std::time::Duration;

use jouleclaw_cascade::*;

use crate::coloring::GraphColoring;
use crate::nqueens::NQueens;
use crate::sat::Cnf;
use crate::solver::{backtrack_measured, BacktrackError};
use crate::sudoku::Sudoku;

pub struct EbmTier {
    pub model_id: u32,
    /// Cap on search decisions per query. Defaults to 5×10⁶, which
    /// handles hard Inkala Sudoku / 16-queens / non-trivial SAT in
    /// milliseconds and rejects adversarial pathological inputs.
    pub max_steps: usize,
}

/// Which constraint problem a query targets.
enum Problem {
    Sudoku(String),
    Sat(String),
    NQueens(String),
    Color(String),
}

impl EbmTier {
    pub fn new() -> Self {
        Self { model_id: 0, max_steps: 5_000_000 }
    }

    pub fn with_max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }

    /// Recognize one of the four supported problem prefixes.
    fn classify(text: &str) -> Option<Problem> {
        let t = text.trim();
        if let Some(r) = t.strip_prefix("sudoku ").or_else(|| t.strip_prefix("sudoku\n")) {
            Some(Problem::Sudoku(r.trim().to_string()))
        } else if let Some(r) = t.strip_prefix("sat ") {
            Some(Problem::Sat(r.trim().to_string()))
        } else if let Some(r) = t.strip_prefix("nqueens ") {
            Some(Problem::NQueens(r.trim().to_string()))
        } else if let Some(r) = t.strip_prefix("color ") {
            Some(Problem::Color(r.trim().to_string()))
        } else {
            None
        }
    }

    /// Conservative joule estimate: max_steps × ~1 pJ/decision, capped
    /// at 1 mJ. R34.2 will calibrate against observed actuals via the
    /// jouleclaw-cascade calibration ledger.
    fn estimated_solve_joules(&self) -> f64 {
        (self.max_steps as f64 * 1e-9 * 1e-3).min(1e-3)
    }
}

impl Default for EbmTier {
    fn default() -> Self {
        Self::new()
    }
}

impl Tier for EbmTier {
    fn id(&self) -> TierId {
        TierId::L1(L1Primitive::Ebm)
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        let text = match &q.input {
            QueryInput::Text(s) => s,
            _ => return None,
        };
        Self::classify(text)?;
        Some(TierEstimate {
            joules: self.estimated_solve_joules(),
            latency: Duration::from_millis(10),
            // V=Full: the solution is verifiable against the constraints.
            confidence_floor: 1.0,
        })
    }

    fn try_answer(&mut self, q: &Query, _budget: f64) -> Result<Answer, AnswerError> {
        let text = match &q.input {
            QueryInput::Text(s) => s,
            _ => return Ok(refused(self.id(), 0.0, RefusalReason::Inapplicable)),
        };
        let Some(problem) = Self::classify(text) else {
            return Ok(refused(
                self.id(),
                0.0,
                RefusalReason::TierSpecific(
                    "expected one of: 'sudoku <board>' | 'sat <cnf>' | \
                     'nqueens <n>' | 'color <k> : <u>-<v> ...'".into(),
                ),
            ));
        };
        let cost = self.estimated_solve_joules();
        let id = self.id();

        let out: Result<String, RefusalReason> = match problem {
            Problem::Sudoku(body) => match Sudoku::parse(&body) {
                Err(e) => Err(RefusalReason::TierSpecific(format!("sudoku parse: {}", e))),
                // Energy-gradient (minimum-remaining-values) ordering:
                // ~4-5× fewer backtracking decisions than naive on hard
                // instances. The decision count is reported so the win
                // is visible in the trace.
                Ok(board) => match backtrack_measured(&board, self.max_steps, true) {
                    Ok((solved, decisions)) => Ok(format!(
                        "sudoku solved ({} energy-gradient decisions):\n{}compact: {}",
                        decisions,
                        solved.render(),
                        solved.render_compact()
                    )),
                    Err(BacktrackError::Infeasible) => Err(RefusalReason::TierSpecific(
                        "sudoku is infeasible — no completion exists".into(),
                    )),
                    Err(BacktrackError::StepLimitExceeded { limit }) => {
                        Err(RefusalReason::TierSpecific(format!(
                            "sudoku exceeded {} decisions — raise max_steps",
                            limit
                        )))
                    }
                },
            },
            Problem::Sat(body) => match Cnf::parse(&body) {
                Err(e) => Err(RefusalReason::TierSpecific(format!("sat parse: {}", e))),
                Ok(cnf) => match cnf.solve(self.max_steps) {
                    Err(e) => Err(RefusalReason::TierSpecific(format!("sat: {}", e))),
                    Ok(Some(model)) => Ok(cnf.render_model(&model)),
                    Ok(None) => Ok("UNSAT — no satisfying assignment exists.".to_string()),
                },
            },
            Problem::NQueens(body) => match NQueens::parse(&body) {
                Err(e) => Err(RefusalReason::TierSpecific(format!("nqueens: {}", e))),
                Ok(q) => match q.solve(self.max_steps) {
                    Some(p) => Ok(q.render(&p)),
                    None => Ok(format!(
                        "no {}-queens solution exists (n=2 and n=3 have none)",
                        q.n
                    )),
                },
            },
            Problem::Color(body) => match GraphColoring::parse(&body) {
                Err(e) => Err(RefusalReason::TierSpecific(format!("coloring: {}", e))),
                Ok(g) => match g.solve(self.max_steps) {
                    Some(c) => Ok(g.render(&c)),
                    None => Ok(format!(
                        "no proper {}-coloring exists for this graph",
                        g.k
                    )),
                },
            },
        };

        match out {
            Ok(text) => Ok(Answer {
                output: AnswerOutput::Text(text),
                tier_used: id,
                joules_spent: cost,
                confidence: 1.0,
                trace: hit_trace(id, cost),
                verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
            }),
            Err(reason) => Ok(refused(id, cost, reason)),
        }
    }

    fn coord(&self) -> Option<jouleclaw_cascade::coord::Coord> {
        use jouleclaw_cascade::coord::{
            Coord, Encoding, Entity, Interface, NamedPrimitive, PrimitiveSet, Thermo,
            Verify, Zone,
        };
        Some(
            Coord::new(
                Zone::Z1,
                Entity::Reactive,
                Thermo::L1_Measure,
                Interface::Tokens,
                Verify::Full,
                Encoding::WorldModel,
            )
            .with_primitives(PrimitiveSet::of(&[
                NamedPrimitive::Parse,
                NamedPrimitive::Arithmetic,
            ])),
        )
    }
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
        verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    const EASY: &str =
        "sudoku 53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";

    #[test]
    fn tier_declares_z1_full_l1_ebm() {
        let tier = EbmTier::new();
        assert_eq!(tier.id(), TierId::L1(L1Primitive::Ebm));
        let coord = tier.coord().expect("coord");
        assert!(matches!(coord.zone, jouleclaw_cascade::coord::Zone::Z1));
        assert!(matches!(coord.verify, jouleclaw_cascade::coord::Verify::Full));
        assert!(matches!(coord.encoding, jouleclaw_cascade::coord::Encoding::WorldModel));
    }

    #[test]
    fn estimate_cost_is_none_for_non_sudoku_queries() {
        let tier = EbmTier::new();
        assert!(tier.estimate_cost(&text_query("compute gcd 12 8")).is_none());
        assert!(tier.estimate_cost(&text_query("what's the weather")).is_none());
    }

    #[test]
    fn estimate_cost_is_some_for_sudoku_query() {
        let tier = EbmTier::new();
        let est = tier.estimate_cost(&text_query(EASY)).expect("sudoku should estimate");
        assert!(est.joules > 0.0);
        assert!(est.joules <= 1e-3, "should cap at 1 mJ, got {}", est.joules);
    }

    #[test]
    fn try_answer_solves_easy_sudoku() {
        let mut tier = EbmTier::new();
        let ans = tier.try_answer(&text_query(EASY), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Text(s) => {
                assert!(s.contains("sudoku solved"));
                assert!(s.contains("energy-gradient decisions"));
                // The compact line should be the known easy solution.
                assert!(
                    s.contains("534678912672195348198342567859761423426853791713924856961537284287419635345286179"),
                    "output:\n{}",
                    s
                );
            }
            other => panic!("expected Text, got {:?}", other),
        }
        assert!(ans.joules_spent > 0.0);
        assert_eq!(ans.confidence, 1.0);
    }

    #[test]
    fn malformed_sudoku_query_refuses_with_structured_reason() {
        let mut tier = EbmTier::new();
        let ans = tier.try_answer(&text_query("sudoku ?"), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Refused(RefusalReason::TierSpecific(msg)) => {
                assert!(msg.contains("sudoku"));
            }
            other => panic!("expected refusal, got {:?}", other),
        }
    }

    #[test]
    fn unknown_problem_prefix_refuses_with_grammar() {
        let mut tier = EbmTier::new();
        let ans = tier.try_answer(&text_query("knapsack 10"), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Refused(RefusalReason::TierSpecific(msg)) => {
                assert!(msg.contains("sudoku") && msg.contains("sat")
                    && msg.contains("nqueens") && msg.contains("color"));
            }
            other => panic!("expected refusal, got {:?}", other),
        }
    }

    #[test]
    fn solves_sat_instance() {
        let mut tier = EbmTier::new();
        let ans = tier
            .try_answer(&text_query("sat 1 2 ; -1 3 ; -2 -3"), 1.0)
            .unwrap();
        match ans.output {
            AnswerOutput::Text(s) => assert!(s.contains("SAT — model:"), "{}", s),
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn reports_unsat() {
        let mut tier = EbmTier::new();
        let ans = tier.try_answer(&text_query("sat 1 ; -1"), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Text(s) => assert!(s.contains("UNSAT"), "{}", s),
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn solves_8_queens() {
        let mut tier = EbmTier::new();
        let ans = tier.try_answer(&text_query("nqueens 8"), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Text(s) => assert!(s.contains("8-queens solution:"), "{}", s),
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn three_colors_a_triangle() {
        let mut tier = EbmTier::new();
        let ans = tier
            .try_answer(&text_query("color 3 : 0-1 1-2 0-2"), 1.0)
            .unwrap();
        match ans.output {
            AnswerOutput::Text(s) => assert!(s.contains("3-coloring"), "{}", s),
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn two_colors_cannot_do_a_triangle() {
        let mut tier = EbmTier::new();
        let ans = tier
            .try_answer(&text_query("color 2 : 0-1 1-2 0-2"), 1.0)
            .unwrap();
        match ans.output {
            AnswerOutput::Text(s) => assert!(s.contains("no proper 2-coloring"), "{}", s),
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn cascade_can_register_ebm_tier() {
        let mut cascade = Cascade::new();
        cascade.register(Box::new(EbmTier::new()));
        let mut rt = Runtime::new_without_l0(cascade);
        let _ = rt.answer(text_query(EASY));
    }
}
