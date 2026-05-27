//! [`ProofReceipt`] — the verifiable artifact emitted by the proof tier.
//!
//! The receipt carries enough state for an auditor to re-check the
//! tier's claim independently: which solver ran, what the problem
//! hash was, what the solution hash was, how many picojoules were
//! attributed, and the [`Provenance`] floor the tier honestly declares
//! for that energy figure.
//!
//! We deliberately hash with a small FNV-1a 64-bit function rather
//! than pulling in blake3 or sha2 here — the receipt is a *fingerprint*
//! of the (problem, solution) tuple, not a cryptographic commitment,
//! and the tier's whole point is staying at constraint-solver energy.
//! Callers who need cryptographic anchoring should wrap a
//! [`ProofReceipt`] in a `jouleclaw-prov` envelope, which provides the
//! signing layer.

use jouleclaw_energy::Provenance;
use serde::{Deserialize, Serialize};

/// Verifiable receipt of a single L4.5 dispatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofReceipt {
    /// Solver name (`Solver::name`). The receipt is keyed to a specific
    /// solver implementation — swapping solvers must re-verify, not
    /// silently inherit trust.
    pub solver_name: String,
    /// 64-bit fingerprint of the canonical problem bytes. Stable
    /// across runs; not collision-resistant against an adversary, but
    /// fine as the audit chain re-runs the solver on the original
    /// problem (the receipt only needs to detect tampering, not block
    /// it).
    pub problem_hash: u64,
    /// 64-bit fingerprint of the canonical solution bytes. For an
    /// [`Unsat`][crate::problem::ProofSolution::Unsat] outcome this is
    /// still populated (the solver did emit a definite verdict).
    pub solution_hash: u64,
    /// Picojoules attributed to this dispatch. Receipt readers should
    /// gate on `provenance` before treating this figure as ground
    /// truth — see the field below.
    pub energy_pj: u64,
    /// Honest declaration of how the energy figure was sourced. For
    /// the bundled solver this is always
    /// [`Provenance::Estimator`] (the JouleClaw static cost model);
    /// hardware-shunt-grade honesty is reserved for solvers that
    /// actually wire a counter.
    pub provenance: Provenance,
}

impl ProofReceipt {
    /// Construct a fresh receipt. The `energy_pj` argument lands
    /// verbatim; the provenance floor is fixed at
    /// [`Provenance::Estimator`] — see the crate-level doc on why we
    /// refuse to over-claim the energy figure.
    pub fn new(
        solver_name: impl Into<String>,
        problem_hash: u64,
        solution_hash: u64,
        energy_pj: u64,
    ) -> Self {
        Self {
            solver_name: solver_name.into(),
            problem_hash,
            solution_hash,
            energy_pj,
            provenance: Provenance::Estimator,
        }
    }
}

/// FNV-1a 64 — a deterministic, alloc-free hash for the receipt
/// fingerprints. Not cryptographic; see [`ProofReceipt`] docs.
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_roundtrips_through_json() {
        let r = ProofReceipt::new("test", 0x1234, 0x5678, 60_000_000);
        let bytes = serde_json::to_vec(&r).expect("encode");
        let back: ProofReceipt = serde_json::from_slice(&bytes).expect("decode");
        assert_eq!(r, back);
    }

    #[test]
    fn provenance_is_estimator_by_default() {
        let r = ProofReceipt::new("s", 0, 0, 1);
        assert!(matches!(r.provenance, Provenance::Estimator));
    }

    #[test]
    fn fnv1a_is_deterministic_and_distinguishes_inputs() {
        assert_eq!(fnv1a_64(b"alpha"), fnv1a_64(b"alpha"));
        assert_ne!(fnv1a_64(b"alpha"), fnv1a_64(b"beta"));
        // Empty input → FNV offset basis.
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    }
}
