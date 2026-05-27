//! Solver-agnostic trait + bundled [`DpllSolver`].
//!
//! The L4.5 tier is solver-agnostic by construction: it depends on the
//! [`Solver`] contract, never on a concrete solver crate. This mirrors
//! how the cascade is LLM-provider-agnostic. A Lean tactic-search
//! backend, a Z3 SMT bridge, or a state-of-the-art CDCL engine all
//! plug in by implementing this one trait.
//!
//! The bundled [`DpllSolver`] is a deliberately textbook DPLL on CNF:
//! it ships, it handles small SAT and CNF-encoded Sudoku in
//! microseconds, and it is small enough that an auditor can hold the
//! whole search procedure in their head. Heavier instances should plug
//! in a stronger solver; the tier's surface does not change.

use crate::problem::{ProofProblem, ProofSolution, ProblemKind};
use crate::sudoku::{decode_sudoku_assignment, sudoku_to_cnf, SudokuSize};
use thiserror::Error;

/// What the solver layer can fail with **independent of UNSAT**. UNSAT
/// is *not* an error — it is a definite, finished verdict and shows
/// up as [`ProofSolution::Unsat`].
#[derive(Debug, Error)]
pub enum SolverError {
    /// The bundled solver does not handle this problem kind. Kept
    /// distinct from "no solution" so the cascade can fall through.
    #[error("solver does not support {0:?}")]
    Unsupported(ProblemKind),
    /// The CSP payload is malformed (e.g. CNF clause containing literal
    /// 0, or a Sudoku grid of the wrong length).
    #[error("malformed input: {0}")]
    Malformed(String),
    /// Search budget exceeded before the solver could decide. Distinct
    /// from UNSAT — the answer is unknown.
    #[error("step limit exceeded: {0}")]
    StepLimit(usize),
}

/// The contract every solver behind L4.5 implements.
///
/// Solvers MUST be deterministic on a given input (so the receipt's
/// solution hash is reproducible) and MUST never claim a satisfying
/// assignment they cannot verify. The [`Solver::verify`] hook is the
/// dispose-side of "AI proposes / verifier disposes": even the
/// solver's own answer is checked through this entry before the tier
/// reports success.
pub trait Solver: Send + Sync {
    /// Stable name used in the [`crate::ProofReceipt`].
    fn name(&self) -> &str;

    /// Run the solver. Returns:
    /// - `Ok(ProofSolution::SatModel(_) | SudokuGrid(_))` on a
    ///   satisfying assignment;
    /// - `Ok(ProofSolution::Unsat)` when the problem has no solution;
    /// - `Err(_)` for *solver-shaped* failures (unsupported shape,
    ///   malformed input, step-limit). UNSAT is **not** an error.
    fn solve(&self, problem: &ProofProblem) -> Result<ProofSolution, SolverError>;

    /// Re-check a proposed solution against the problem. Returning
    /// `false` means the solver itself doesn't believe its own answer
    /// — the tier MUST refuse rather than silently passing it
    /// through. Default implementation re-runs `solve` and compares;
    /// solvers with a cheaper independent verifier should override.
    fn verify(&self, problem: &ProofProblem, solution: &ProofSolution) -> bool {
        match self.solve(problem) {
            Ok(s) => &s == solution,
            Err(_) => false,
        }
    }
}

// ============================================================
// DPLL on CNF — the bundled default solver
// ============================================================

/// Textbook DPLL SAT solver — the JouleClaw default for L4.5.
///
/// Search procedure (unit propagation + simple branching, no clause
/// learning, no watched literals — kept small on purpose):
///   1. Unit-propagate to a fixed point.
///   2. If any clause is empty → UNSAT in this branch.
///   3. If every clause has a satisfied literal → SAT.
///   4. Pick the lowest unassigned variable. Try `true` first; on
///      failure, try `false`.
///
/// `max_steps` bounds the number of decisions before the solver bails
/// with [`SolverError::StepLimit`]; the default (`5_000_000`) handles
/// CNF-encoded 4×4 Sudoku and small SAT in microseconds and rejects
/// adversarial inputs.
#[derive(Debug, Clone, Copy)]
pub struct DpllSolver {
    /// Hard cap on branching decisions per `solve` call.
    pub max_steps: usize,
}

impl Default for DpllSolver {
    fn default() -> Self {
        Self { max_steps: 5_000_000 }
    }
}

impl DpllSolver {
    /// Construct with an explicit step cap.
    pub fn with_max_steps(max_steps: usize) -> Self {
        Self { max_steps }
    }

    /// Solve a CNF formula directly. Variables are positive integers
    /// starting at 1; negation is the negative sign. Returns
    /// `Some(model)` on SAT (length = max variable index), `None` on
    /// UNSAT.
    fn solve_cnf(
        &self,
        clauses: &[Vec<i32>],
    ) -> Result<Option<Vec<bool>>, SolverError> {
        // Reject literal 0 and discover the number of variables.
        let mut n: usize = 0;
        for cl in clauses {
            for &lit in cl {
                if lit == 0 {
                    return Err(SolverError::Malformed(
                        "CNF literal must be non-zero".into(),
                    ));
                }
                let v = lit.unsigned_abs() as usize;
                if v > n {
                    n = v;
                }
            }
        }
        if n == 0 {
            // Empty CNF (or only-trivial-empty clauses). Vacuously SAT.
            // An empty clause list = no constraints, every assignment
            // works; return the empty model.
            if clauses.iter().all(|cl| !cl.is_empty()) {
                return Ok(Some(Vec::new()));
            }
            // An explicit empty clause is UNSAT.
            return Ok(None);
        }
        // Assignment: None = unassigned, Some(b) = decided.
        let mut assign: Vec<Option<bool>> = vec![None; n];
        let mut steps: usize = 0;
        match dpll(clauses, &mut assign, &mut steps, self.max_steps)? {
            true => {
                // Any unforced variable defaults to `false`. (Either
                // value is sound; pick deterministically.)
                let model: Vec<bool> =
                    assign.into_iter().map(|v| v.unwrap_or(false)).collect();
                Ok(Some(model))
            }
            false => Ok(None),
        }
    }
}

impl Solver for DpllSolver {
    fn name(&self) -> &str {
        "jouleclaw::dpll"
    }

    fn solve(&self, problem: &ProofProblem) -> Result<ProofSolution, SolverError> {
        match problem.kind {
            ProblemKind::Sat => {
                let cnf = problem.cnf.as_ref().ok_or_else(|| {
                    SolverError::Malformed("sat: missing 'cnf' field".into())
                })?;
                match self.solve_cnf(cnf)? {
                    Some(model) => Ok(ProofSolution::SatModel(model)),
                    None => Ok(ProofSolution::Unsat),
                }
            }
            ProblemKind::Sudoku | ProblemKind::Sudoku4 => {
                let grid = problem.grid.as_ref().ok_or_else(|| {
                    SolverError::Malformed("sudoku: missing 'grid' field".into())
                })?;
                let size = if problem.kind == ProblemKind::Sudoku4 {
                    SudokuSize::N4
                } else {
                    SudokuSize::N9
                };
                let (clauses, n_vars) = sudoku_to_cnf(grid, size)?;
                match self.solve_cnf(&clauses)? {
                    Some(mut model) => {
                        // Pad model so decode can index by variable id.
                        if model.len() < n_vars {
                            model.resize(n_vars, false);
                        }
                        let solved = decode_sudoku_assignment(&model, size)?;
                        Ok(ProofSolution::SudokuGrid(solved))
                    }
                    None => Ok(ProofSolution::Unsat),
                }
            }
        }
    }
}

/// The recursive DPLL core. Returns `Ok(true)` if a satisfying
/// extension of `assign` exists, `Ok(false)` if not, `Err(_)` on
/// step-limit.
fn dpll(
    clauses: &[Vec<i32>],
    assign: &mut [Option<bool>],
    steps: &mut usize,
    max_steps: usize,
) -> Result<bool, SolverError> {
    *steps += 1;
    if *steps > max_steps {
        return Err(SolverError::StepLimit(max_steps));
    }

    // Unit-propagate to a fixed point.
    loop {
        let mut progressed = false;
        for cl in clauses {
            let mut unassigned_lit: Option<i32> = None;
            let mut satisfied = false;
            let mut undecided_count = 0;
            for &lit in cl {
                let v = lit.unsigned_abs() as usize - 1;
                match assign[v] {
                    Some(b) => {
                        if (lit > 0) == b {
                            satisfied = true;
                            break;
                        }
                    }
                    None => {
                        undecided_count += 1;
                        unassigned_lit = Some(lit);
                    }
                }
            }
            if satisfied {
                continue;
            }
            if undecided_count == 0 {
                // Empty clause under current assignment → conflict.
                return Ok(false);
            }
            if undecided_count == 1 {
                let lit = match unassigned_lit {
                    Some(l) => l,
                    None => return Ok(false),
                };
                let v = lit.unsigned_abs() as usize - 1;
                assign[v] = Some(lit > 0);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }

    // Check overall status.
    let mut all_sat = true;
    let mut next_var: Option<usize> = None;
    for cl in clauses {
        let mut sat = false;
        let mut any_undecided = false;
        for &lit in cl {
            let v = lit.unsigned_abs() as usize - 1;
            match assign[v] {
                Some(b) => {
                    if (lit > 0) == b {
                        sat = true;
                        break;
                    }
                }
                None => any_undecided = true,
            }
        }
        if sat {
            continue;
        }
        if !any_undecided {
            return Ok(false);
        }
        all_sat = false;
    }
    if all_sat {
        return Ok(true);
    }

    // Branch on the lowest unassigned variable.
    for (i, slot) in assign.iter().enumerate() {
        if slot.is_none() {
            next_var = Some(i);
            break;
        }
    }
    let Some(v) = next_var else {
        return Ok(true);
    };

    for &b in &[true, false] {
        let snapshot: Vec<Option<bool>> = assign.to_vec();
        assign[v] = Some(b);
        match dpll(clauses, assign, steps, max_steps)? {
            true => return Ok(true),
            false => {
                // Restore and try the other branch.
                for (i, val) in snapshot.iter().enumerate() {
                    assign[i] = *val;
                }
            }
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sat_problem(cnf: Vec<Vec<i32>>) -> ProofProblem {
        ProofProblem {
            kind: ProblemKind::Sat,
            cnf: Some(cnf),
            grid: None,
        }
    }

    #[test]
    fn solves_trivial_sat() {
        let solver = DpllSolver::default();
        let out = solver.solve(&sat_problem(vec![vec![1], vec![-2]])).unwrap();
        match out {
            ProofSolution::SatModel(m) => {
                assert!(m[0]);
                assert!(!m[1]);
            }
            other => panic!("expected SatModel, got {:?}", other),
        }
    }

    #[test]
    fn detects_simple_unsat() {
        let solver = DpllSolver::default();
        let out = solver.solve(&sat_problem(vec![vec![1], vec![-1]])).unwrap();
        assert_eq!(out, ProofSolution::Unsat);
    }

    #[test]
    fn rejects_zero_literal() {
        let solver = DpllSolver::default();
        let err = solver.solve(&sat_problem(vec![vec![0]])).unwrap_err();
        assert!(matches!(err, SolverError::Malformed(_)));
    }

    #[test]
    fn three_clause_sat_finds_model() {
        // (a ∨ b) ∧ (¬a ∨ c) ∧ (¬b ∨ ¬c)
        let solver = DpllSolver::default();
        let out = solver
            .solve(&sat_problem(vec![vec![1, 2], vec![-1, 3], vec![-2, -3]]))
            .unwrap();
        assert!(out.is_satisfying());
    }

    #[test]
    fn solver_verify_round_trip_matches() {
        let solver = DpllSolver::default();
        let p = sat_problem(vec![vec![1, 2], vec![-1, 3]]);
        let s = solver.solve(&p).unwrap();
        assert!(solver.verify(&p, &s));
    }

    #[test]
    fn step_limit_triggers_error() {
        // Choose a tiny limit; the recursion entry alone bumps steps.
        let solver = DpllSolver::with_max_steps(0);
        let err = solver
            .solve(&sat_problem(vec![vec![1, 2, 3], vec![-1, -2, -3]]))
            .unwrap_err();
        assert!(matches!(err, SolverError::StepLimit(_)));
    }

    #[test]
    fn solver_name_is_stable() {
        assert_eq!(DpllSolver::default().name(), "jouleclaw::dpll");
    }
}
