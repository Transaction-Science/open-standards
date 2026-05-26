//! EBM — Energy-Based reasoning tiers for pattern-lang's cascade.
//!
//! The thesis (Hopfield → LeCun → Logical Intelligence's Konna):
//! reasoning that involves global constraints — Sudoku, SAT, scheduling,
//! layout, planning under safety — is **not** a next-token problem.
//! It's an optimization problem. You define an energy function over
//! candidate states (low energy = satisfies constraints; high = violates)
//! and search for a zero-energy state.
//!
//! Pattern-lang's cascade already prices answers by joules. EBMs slot
//! in naturally as a tier class:
//!
//! - `L1::Ebm` for problems with **crisp** constraints (Z1, V=Full).
//!   Output is a concrete state checkable against the rules.
//! - Future `L2::Ebm` for soft / approximate constraint satisfaction
//!   (Z2, V=Statistical) — not in this revision.
//!
//! What ships (R34.0 + R34.1):
//!
//! - [`EnergyFunction`] trait — abstracts "score a state, lower is better"
//! - [`sudoku::Sudoku`] — 9×9 Sudoku, backtracking solve (R34.0)
//! - [`sat::Cnf`] — Boolean SAT via DPLL with unit propagation (R34.1)
//! - [`nqueens::NQueens`] — N-Queens via column backtracking (R34.1)
//! - [`coloring::GraphColoring`] — graph k-coloring via backtracking (R34.1)
//! - [`tier::EbmTier`] — joule cascade tier. Dispatches by query prefix:
//!   `sudoku <board>` · `sat <cnf>` · `nqueens <n>` · `color <k> : <edges>`
//!
//! Each problem is a concrete `EnergyFunction`: energy = constraint
//! violation count, a zero-energy state is a solution. Every solver is
//! deterministic and step-capped (the cap protects the joule budget
//! against adversarial inputs).
//!
//! Deferred to R34.2 / later:
//!
//! - Simulated annealing / Metropolis for soft / continuous energy
//! - CDCL (conflict-driven clause learning) — the DPLL here is the
//!   textbook version, not a competition solver
//! - Konna-class *learned* energy functions (these use explicit
//!   constraint counters; Konna learns the energy)
//! - Integration with `lean-bridge` so EbmTier's output (a solved
//!   state) flows into LeanProofTier (a machine-checkable proof that
//!   the state satisfies the constraints)

pub mod coloring;
pub mod nqueens;
pub mod sat;
pub mod solver;
pub mod sudoku;
pub mod tier;

pub use coloring::{Coloring, ColoringError, GraphColoring};
pub use nqueens::{NQueens, NQueensError, Placement};
pub use sat::{Cnf, Lit, Model, SatError};
pub use solver::{backtrack, backtrack_measured, BacktrackError};
pub use sudoku::{Sudoku, SudokuError};
pub use tier::EbmTier;

/// A scalar energy function: states with lower energy are preferred.
/// A zero-energy state is fully constraint-satisfying.
pub trait EnergyFunction {
    type State: Clone;

    /// Score a state. Zero = all constraints satisfied; positive =
    /// violations. Implementations should be deterministic.
    fn energy(&self, state: &Self::State) -> f64;
}
