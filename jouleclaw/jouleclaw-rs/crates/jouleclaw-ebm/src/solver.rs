//! `solver::backtrack` — deterministic depth-first search for
//! constraint-satisfaction problems.
//!
//! The shape here matches the classical Sudoku/SAT solver pattern:
//!
//! 1. Find the next "open" decision variable (an empty cell, an
//!    unassigned propositional variable, etc.).
//! 2. Try each candidate value in order.
//! 3. If a candidate is legal w.r.t. the current partial state,
//!    commit and recurse.
//! 4. If recursion fails, undo and try the next candidate.
//! 5. If no candidate works, backtrack to the caller.
//!
//! For R34.0 the solver is hardcoded against [`crate::sudoku::Sudoku`].
//! R34.1 will generalize this to any constraint-satisfaction shape via
//! a trait — the EnergyFunction trait already gives us the score side;
//! solver-side we need a "next decision variable" abstraction.

use crate::sudoku::Sudoku;

#[derive(Debug, Clone, PartialEq)]
pub enum BacktrackError {
    Infeasible,
    StepLimitExceeded { limit: usize },
}

impl std::fmt::Display for BacktrackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Infeasible => write!(f, "no satisfying assignment exists"),
            Self::StepLimitExceeded { limit } => {
                write!(f, "exceeded step limit of {} decisions", limit)
            }
        }
    }
}

impl std::error::Error for BacktrackError {}

/// Solve a Sudoku via depth-first backtracking with cap on decisions.
/// Returns the solved board or an error.
///
/// `max_steps` is a defensive cap: classical Sudoku solves in well
/// under 10⁶ decisions; pathological adversarial boards can be much
/// worse. Cap protects the tier's joule budget.
pub fn backtrack(start: &Sudoku, max_steps: usize) -> Result<Sudoku, BacktrackError> {
    // Sanity: the starting board itself must not contain a duplicate
    // value within a row/column/box. Otherwise the backtracker would
    // happily fill the remaining cells around the pre-existing conflict
    // and "succeed" with an invalid board.
    for r in 0..9 {
        for c in 0..9 {
            let v = start.at(r, c);
            if v != 0 && !start.is_legal(r, c, v) {
                return Err(BacktrackError::Infeasible);
            }
        }
    }
    let mut board = start.clone();
    let mut steps = 0usize;
    if solve_recursive(&mut board, &mut steps, max_steps)? {
        Ok(board)
    } else {
        Err(BacktrackError::Infeasible)
    }
}

/// Backtracking solve that also reports the number of decisions made.
/// `guided=true` uses energy-gradient (minimum-remaining-values) cell
/// ordering; `guided=false` is naive row-major. Same answer either way
/// (the solver is exact); the decision count is the measurable win.
pub fn backtrack_measured(
    start: &Sudoku,
    max_steps: usize,
    guided: bool,
) -> Result<(Sudoku, usize), BacktrackError> {
    for r in 0..9 {
        for c in 0..9 {
            let v = start.at(r, c);
            if v != 0 && !start.is_legal(r, c, v) {
                return Err(BacktrackError::Infeasible);
            }
        }
    }
    let mut board = start.clone();
    let mut steps = 0usize;
    let ok = solve_inner(&mut board, &mut steps, max_steps, guided)?;
    if ok {
        Ok((board, steps))
    } else {
        Err(BacktrackError::Infeasible)
    }
}

fn solve_recursive(
    board: &mut Sudoku,
    steps: &mut usize,
    max_steps: usize,
) -> Result<bool, BacktrackError> {
    solve_inner(board, steps, max_steps, false)
}

fn solve_inner(
    board: &mut Sudoku,
    steps: &mut usize,
    max_steps: usize,
    guided: bool,
) -> Result<bool, BacktrackError> {
    let pick = if guided {
        board.most_constrained_empty()
    } else {
        board.next_empty()
    };
    let (r, c) = match pick {
        Some(p) => p,
        None => return Ok(true), // fully filled, no constraint violations possible
    };
    for v in 1u8..=9 {
        *steps += 1;
        if *steps > max_steps {
            return Err(BacktrackError::StepLimitExceeded { limit: max_steps });
        }
        if board.is_legal(r, c, v) {
            board.set(r, c, v);
            if solve_inner(board, steps, max_steps, guided)? {
                return Ok(true);
            }
            board.set(r, c, 0); // undo
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EASY: &str =
        "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";

    const EASY_SOLVED: &str =
        "534678912672195348198342567859761423426853791713924856961537284287419635345286179";

    // A famously hard puzzle from Arto Inkala (2012).
    const HARD: &str =
        "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..";

    #[test]
    fn solves_easy_puzzle() {
        let s = Sudoku::parse(EASY).unwrap();
        let out = backtrack(&s, 1_000_000).unwrap();
        assert!(out.is_solved());
        assert_eq!(out.render_compact(), EASY_SOLVED);
    }

    #[test]
    fn solves_inkala_hard_puzzle() {
        let s = Sudoku::parse(HARD).unwrap();
        let out = backtrack(&s, 5_000_000).unwrap();
        assert!(out.is_solved());
    }

    #[test]
    fn solver_is_deterministic() {
        let s = Sudoku::parse(EASY).unwrap();
        let a = backtrack(&s, 1_000_000).unwrap();
        let b = backtrack(&s, 1_000_000).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn infeasible_board_returns_infeasible() {
        // Place two 1s in the same row — no completion possible.
        let mut s = Sudoku::parse(
            "1.1..............................................................................",
        )
        .unwrap();
        // Force an actual conflict in the starting board.
        s.set(0, 1, 1);
        let r = backtrack(&s, 100_000);
        // is_legal would have prevented the conflict if we'd built up
        // from scratch, but the starting board itself is conflicted —
        // backtracking from the next empty cell can never succeed.
        assert!(matches!(r, Err(BacktrackError::Infeasible)));
    }

    #[test]
    fn step_limit_protects_against_runaway() {
        let s = Sudoku::parse(HARD).unwrap();
        // A laughably tiny budget for a hard puzzle.
        let r = backtrack(&s, 100);
        assert!(matches!(r, Err(BacktrackError::StepLimitExceeded { .. })));
    }
}

#[cfg(test)]
mod r34_2_tests {
    use super::*;
    use crate::sudoku::Sudoku;

    // Arto Inkala's "world's hardest" Sudoku (2012) — pathological for
    // naive row-major backtracking.
    const HARD: &str =
        "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..";

    #[test]
    fn energy_gradient_ordering_slashes_decision_count() {
        let s = Sudoku::parse(HARD).unwrap();
        let (sol_naive, steps_naive) =
            backtrack_measured(&s, 50_000_000, false).unwrap();
        let (sol_guided, steps_guided) =
            backtrack_measured(&s, 50_000_000, true).unwrap();

        // Exact solver: identical solution regardless of ordering.
        assert_eq!(sol_naive, sol_guided);
        assert!(sol_guided.is_solved());

        // The measurable win: energy-gradient (MRV) ordering must cut
        // decisions by a large factor on this pathological instance.
        eprintln!(
            "Inkala hard: naive={} decisions, energy-gradient={} decisions ({}x fewer)",
            steps_naive,
            steps_guided,
            steps_naive / steps_guided.max(1)
        );
        // Honest, measured threshold: on the Inkala instance the win is
        // ~4–5× (445,778 → 90,665 decisions). Assert a clear >2× to be
        // robust across platforms while not over-claiming.
        assert!(
            steps_guided * 2 < steps_naive,
            "energy-gradient ordering should be >2x fewer decisions: \
             naive={} guided={}",
            steps_naive,
            steps_guided
        );
    }
}
