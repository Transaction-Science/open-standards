//! # jouleclaw-proof-tier
//!
//! **L4.5 — the proof tier.** Deterministic constraint / proof solving
//! (SAT, Sudoku, type inference, scheduling) as a JouleClaw cascade
//! tier. The whole point of this tier is the inversion: "find a state
//! that fits" problems do NOT need an LLM — they need a solver. The
//! cost lands in the tens of microjoules, not the megajoules of a
//! frontier RPC. L4.5 sits *after* L4 in the tier ID space but resolves
//! cheaper than L4 by construction, because constraint satisfaction is
//! a different shape of problem than open-ended generation.
//!
//! Architectural shape (ported from `verity-cascade::layers::l45_proof`,
//! with the OpenIE-IP solver deps replaced by a clean pure-Rust
//! [`Solver`] trait + default [`DpllSolver`]):
//!
//! - [`Solver`] — solver-agnostic contract. Callers can swap in a
//!   stronger SAT engine, an SMT bridge, a Lean tactic search, etc.
//! - [`DpllSolver`] — the bundled default. A textbook DPLL on CNF that
//!   handles small SAT and (via [`sudoku_to_cnf`]) small Sudoku.
//! - [`ProofTier`] — implements `jouleclaw_cascade::Tier`, dispatches
//!   the active solver, emits a [`ProofReceipt`], and (optionally)
//!   re-checks the solver's own answer through the public [`Solver`]
//!   contract before returning success. AI proposes; verifier disposes.
//! - [`ProofReceipt`] — verifiable receipt with problem hash, solution
//!   hash, solver name, attributed picojoules, and an honest
//!   [`Provenance`][jouleclaw_energy::Provenance] floor of
//!   [`Estimator`][jouleclaw_energy::Provenance::Estimator] (we did not
//!   measure on a hardware shunt, so we do not claim we did).
//!
//! The query protocol is JSON, carried in
//! `QueryInput::Structured(bytes)`:
//!
//! ```json
//! { "kind": "sat",    "cnf":   [[1, 2], [-1, 3], [-2, -3]] }
//! { "kind": "sudoku", "grid":  "53..7....6..195....98....6.8..." }
//! { "kind": "sudoku4","grid":  ".2..3..1...3..." }   // 4×4 variant
//! ```
//!
//! Anything else → [`estimate_cost`][ProofTier::estimate_cost] returns
//! `None` and the cascade walks past this tier. The 60 µJ figure used
//! in the typical estimate is the donor's mgai-csp benchmark for a
//! solved problem; substantially larger instances obviously cost more,
//! and we surface the per-solve actual through the receipt rather than
//! pretending the estimate is the truth.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod problem;
pub mod receipt;
pub mod solver;
pub mod sudoku;
pub mod tier;

pub use problem::{ProofProblem, ProofSolution, ProblemKind, ProblemError};
pub use receipt::ProofReceipt;
pub use solver::{DpllSolver, Solver, SolverError};
pub use sudoku::{sudoku_to_cnf, decode_sudoku_assignment, SudokuSize};
pub use tier::ProofTier;
