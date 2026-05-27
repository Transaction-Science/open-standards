//! The proof-tier query shape and its parser.
//!
//! Queries arrive as `QueryInput::Structured(bytes)` carrying a small
//! JSON envelope. The envelope is the constraint-satisfaction problem
//! the tier solves. A free-text → constraint extractor is the explicit
//! next step (and is intentionally NOT wired in as a regex shim — see
//! the donor's prose for the reasoning); callers that already hold a
//! structured problem (the common case) use this shape directly.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The kinds of problems the bundled solver knows about. New problem
/// shapes plug in by extending this enum *and* the matching solver
/// handler — never one without the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProblemKind {
    /// Boolean satisfiability over a CNF formula. Variables are
    /// positive `i32` integers; negation is the negative sign.
    Sat,
    /// 9×9 Sudoku encoded as CNF.
    Sudoku,
    /// 4×4 Sudoku — small enough to round-trip in unit tests in
    /// microseconds.
    Sudoku4,
}

/// The on-wire shape carried by `QueryInput::Structured`.
///
/// We use untagged-ish encoding so the JSON stays human-writable and
/// the structure self-documents: `"kind"` selects the shape, the
/// remaining fields are the payload for that shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofProblem {
    /// Which CSP shape this query carries.
    pub kind: ProblemKind,
    /// CNF formula. Each inner vector is one clause, where each
    /// non-zero integer is a literal: positive for the variable,
    /// negative for its negation. Only populated when `kind == Sat`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cnf: Option<Vec<Vec<i32>>>,
    /// Sudoku grid as a compact 81-char (or 16-char) string. `.`, `0`,
    /// and `_` mean empty; otherwise digits 1–9 (or 1–4). Populated
    /// when `kind == Sudoku` or `kind == Sudoku4`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid: Option<String>,
}

impl ProofProblem {
    /// Parse a JSON envelope from raw bytes (the cascade's
    /// `QueryInput::Structured` shape).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProblemError> {
        serde_json::from_slice(bytes).map_err(ProblemError::Decode)
    }

    /// Canonical JSON serialization for hashing. Field order is fixed
    /// by serde's struct layout — sufficient for our hash, since the
    /// only consumer is our own receipt re-check.
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        // serde_json is deterministic for a `Serialize` impl with no
        // map-with-arbitrary-order fields; ProofProblem has none.
        serde_json::to_vec(self).unwrap_or_default()
    }
}

/// The solution returned from the solver, before tier-level wrapping.
///
/// Variant choice tracks the problem kind: a SAT solution is a model
/// (the variable assignment), a Sudoku solution is the completed grid.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofSolution {
    /// Satisfying assignment to the SAT variables. The vector is
    /// indexed from 0; entry `i` is the truth value of variable `i+1`.
    SatModel(Vec<bool>),
    /// Solved Sudoku grid in compact form.
    SudokuGrid(String),
    /// The problem is unsatisfiable / has no solution. Distinct from a
    /// solver error: this is a finished, definite answer.
    Unsat,
}

impl ProofSolution {
    /// `true` iff the solver produced a concrete satisfying state
    /// (i.e. not `Unsat`).
    pub fn is_satisfying(&self) -> bool {
        !matches!(self, ProofSolution::Unsat)
    }

    /// Canonical bytes for hashing into a [`crate::ProofReceipt`].
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }
}

/// Things that can go wrong parsing a [`ProofProblem`].
#[derive(Debug, Error)]
pub enum ProblemError {
    /// JSON decode of the structured envelope failed.
    #[error("json decode failed: {0}")]
    Decode(#[source] serde_json::Error),
    /// The envelope decoded but is missing the payload its `kind`
    /// requires (e.g. `kind: "sat"` with no `cnf` field).
    #[error("malformed problem: {0}")]
    Malformed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_sat() {
        let p = ProofProblem {
            kind: ProblemKind::Sat,
            cnf: Some(vec![vec![1, 2], vec![-1, 3]]),
            grid: None,
        };
        let bytes = p.to_canonical_bytes();
        let back = ProofProblem::from_bytes(&bytes).expect("decode");
        assert_eq!(back.kind, ProblemKind::Sat);
        assert_eq!(back.cnf.as_ref().map(|c| c.len()), Some(2));
    }

    #[test]
    fn decode_rejects_garbage() {
        let err = ProofProblem::from_bytes(b"not json").unwrap_err();
        assert!(matches!(err, ProblemError::Decode(_)));
    }

    #[test]
    fn proof_solution_canonical_bytes_is_deterministic() {
        let s = ProofSolution::SatModel(vec![true, false, true]);
        let a = s.to_canonical_bytes();
        let b = s.to_canonical_bytes();
        assert_eq!(a, b);
        assert!(s.is_satisfying());
    }

    #[test]
    fn unsat_is_not_satisfying() {
        assert!(!ProofSolution::Unsat.is_satisfying());
    }
}
