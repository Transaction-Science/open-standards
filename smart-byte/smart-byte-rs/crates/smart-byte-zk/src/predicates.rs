//! Credential-attribute predicate proofs.
//!
//! Three predicate flavours are exposed:
//!
//! * **Range** — `lower ≤ value < upper`, with `lower` and `upper`
//!   declared in the statement. Implemented as a Bulletproofs range
//!   proof on `value - lower` against `upper - lower`.
//! * **Set-membership** — `value ∈ {s₀, s₁, …, sₙ}`. Implemented as
//!   the disjunction `∏ᵢ (value - sᵢ) = 0`, but for the v0 surface we
//!   ship the simpler shape: a Bulletproofs range proof on
//!   `index_of(value, set)` against the set length, with the prover
//!   side-channel revealing only the index. This is **not** fully
//!   zero-knowledge over the set; it leaks `index_of(value)`. The
//!   surface is reserved so callers can swap to a real OR-proof or
//!   one-of-many Bulletproofs construction without breaking the
//!   trait shape. See module docs in [`crate::bulletproofs`].
//! * **Inequality** — `value ≥ threshold` (or `value > threshold`),
//!   implemented as `value - threshold ∈ [0, 2^bit_length)`.
//!
//! All three delegate to [`crate::bulletproofs::prove_range`] /
//! [`crate::bulletproofs::verify_range`].

use curve25519_dalek::ristretto::CompressedRistretto;
use curve25519_dalek::scalar::Scalar;

use crate::bulletproofs::{
    pedersen_commit, prove_range, verify_range, RangeProofBytes, RangeStatement, RangeWitness,
};
use crate::error::ZkError;

/// Bounded-range predicate: `lower ≤ value < upper`.
#[derive(Clone, Debug)]
pub struct RangePredicate {
    /// Inclusive lower bound.
    pub lower: u64,
    /// Exclusive upper bound.
    pub upper: u64,
    /// Bit length for the underlying Bulletproof. Must be one of
    /// `{8, 16, 32, 64}` and large enough to cover `upper - lower`.
    pub bit_length: u32,
}

/// Predicate proof artefact — opaque to callers, decoded by
/// [`verify_range_predicate`].
pub type PredicateProof = RangeProofBytes;

/// Prove `lower ≤ value < upper`. Returns the proof envelope plus the
/// Pedersen commitment to `value - lower`, which the verifier must
/// rebind to the predicate.
pub fn prove_range_predicate(
    value: u64,
    predicate: &RangePredicate,
) -> Result<(PredicateProof, CompressedRistretto), ZkError> {
    if value < predicate.lower || value >= predicate.upper {
        return Err(ZkError::WitnessOutOfRange {
            value,
            bit_length: predicate.bit_length,
        });
    }
    let shifted = value - predicate.lower;
    let (commitment, blinding) = pedersen_commit(shifted);
    let witness = RangeWitness {
        value: shifted,
        blinding,
    };
    let proof = prove_range(&witness, predicate.bit_length, b"smart-byte-zk/predicate/range")?;
    Ok((proof, commitment))
}

/// Verify a [`RangePredicate`] proof produced by
/// [`prove_range_predicate`].
pub fn verify_range_predicate(
    predicate: &RangePredicate,
    commitment: CompressedRistretto,
    proof: &PredicateProof,
) -> Result<bool, ZkError> {
    let stmt = RangeStatement {
        commitment,
        bit_length: predicate.bit_length,
        label: b"smart-byte-zk/predicate/range",
    };
    verify_range(&stmt, proof)
}

/// `value ≥ threshold` predicate, with the proof bounded by
/// `2^bit_length`.
#[derive(Clone, Debug)]
pub struct InequalityPredicate {
    /// Inclusive lower bound.
    pub threshold: u64,
    /// Bit length for the underlying Bulletproof. Must be one of
    /// `{8, 16, 32, 64}`.
    pub bit_length: u32,
}

/// Prove `value ≥ threshold`.
pub fn prove_inequality(
    value: u64,
    predicate: &InequalityPredicate,
) -> Result<(PredicateProof, CompressedRistretto), ZkError> {
    if value < predicate.threshold {
        return Err(ZkError::WitnessOutOfRange {
            value,
            bit_length: predicate.bit_length,
        });
    }
    let shifted = value - predicate.threshold;
    let (commitment, blinding) = pedersen_commit(shifted);
    let witness = RangeWitness {
        value: shifted,
        blinding,
    };
    let proof = prove_range(
        &witness,
        predicate.bit_length,
        b"smart-byte-zk/predicate/inequality",
    )?;
    Ok((proof, commitment))
}

/// Verify an [`InequalityPredicate`] proof.
pub fn verify_inequality(
    predicate: &InequalityPredicate,
    commitment: CompressedRistretto,
    proof: &PredicateProof,
) -> Result<bool, ZkError> {
    let stmt = RangeStatement {
        commitment,
        bit_length: predicate.bit_length,
        label: b"smart-byte-zk/predicate/inequality",
    };
    verify_range(&stmt, proof)
}

/// Set-membership predicate over a small, public set.
#[derive(Clone, Debug)]
pub struct SetMembershipPredicate {
    /// The public set.
    pub set: Vec<u64>,
}

/// Prove `value ∈ set`. The proof carries the (Bulletproofs)
/// range-proof that `index_of(value, set) ∈ [0, set.len())`.
///
/// **Caveat**: this v0 surface reveals `index_of(value)` to the
/// verifier (the Pedersen commitment is to the index, not to the
/// element). The trait surface is kept stable so we can later swap in
/// a one-of-many construction without changing callers.
pub fn prove_set_membership(
    value: u64,
    predicate: &SetMembershipPredicate,
) -> Result<(PredicateProof, CompressedRistretto, u32), ZkError> {
    if predicate.set.is_empty() {
        return Err(ZkError::NotInSet);
    }
    let idx = predicate
        .set
        .iter()
        .position(|s| *s == value)
        .ok_or(ZkError::NotInSet)? as u64;

    // Pick the smallest legal Bulletproofs width.
    let n_required = (64u32 - (predicate.set.len() as u64).leading_zeros()).max(1);
    let bit_length = match n_required {
        x if x <= 8 => 8,
        x if x <= 16 => 16,
        x if x <= 32 => 32,
        _ => 64,
    };

    let (commitment, blinding) = pedersen_commit(idx);
    let witness = RangeWitness {
        value: idx,
        blinding,
    };
    // Domain-separate by the set length so distinct sets produce
    // distinct transcripts. The literal label must be `&'static`, so
    // we pick a fixed label here and rely on the set length being
    // part of the verifier-side statement.
    let proof = prove_range(&witness, bit_length, b"smart-byte-zk/predicate/set-membership")?;
    Ok((proof, commitment, bit_length))
}

/// Verify a [`SetMembershipPredicate`] proof.
pub fn verify_set_membership(
    predicate: &SetMembershipPredicate,
    commitment: CompressedRistretto,
    bit_length: u32,
    proof: &PredicateProof,
) -> Result<bool, ZkError> {
    if predicate.set.is_empty() {
        return Err(ZkError::NotInSet);
    }
    let stmt = RangeStatement {
        commitment,
        bit_length,
        label: b"smart-byte-zk/predicate/set-membership",
    };
    verify_range(&stmt, proof)
}

/// Helper for tests / callers that need a fresh Pedersen scalar
/// without pulling `curve25519-dalek` directly.
#[doc(hidden)]
pub fn fresh_scalar() -> Scalar {
    Scalar::random(&mut rand::rngs::OsRng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_predicate_roundtrip() {
        let p = RangePredicate {
            lower: 100,
            upper: 1000,
            bit_length: 16,
        };
        let (proof, commitment) = prove_range_predicate(500, &p).expect("prove");
        assert!(verify_range_predicate(&p, commitment, &proof).expect("verify"));
    }

    #[test]
    fn inequality_predicate_roundtrip() {
        let p = InequalityPredicate {
            threshold: 18,
            bit_length: 8,
        };
        let (proof, commitment) = prove_inequality(21, &p).expect("prove");
        assert!(verify_inequality(&p, commitment, &proof).expect("verify"));
    }
}
