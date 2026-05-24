//! Real Bulletproofs backend over Ristretto255.
//!
//! Backed by [`::bulletproofs`] 5.x and `curve25519-dalek` 4.x. We
//! expose a thin range-proof façade plus a [`ZkScheme`] impl wired
//! against it.
//!
//! Use this module directly for the high-throughput / minimum-overhead
//! path, or go through [`crate::predicates`] for the credential-level
//! predicate API.

use ::bulletproofs::{BulletproofGens, PedersenGens, RangeProof};
use curve25519_dalek::ristretto::CompressedRistretto;
use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

use crate::error::ZkError;
use crate::scheme::{Proof, ProvingKey, VerifyingKey, ZkScheme};

/// Statement: prove that the value committed under `commitment` lies
/// in `[0, 2^bit_length)`.
///
/// `bit_length` must be one of `{8, 16, 32, 64}` — Bulletproofs only
/// supports those four widths in its single-value range-proof API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeStatement {
    /// Pedersen commitment to the witness value.
    pub commitment: CompressedRistretto,
    /// Width in bits.
    pub bit_length: u32,
    /// Transcript domain-separator label.
    pub label: &'static [u8],
}

/// Witness: the cleartext value + Pedersen blinding factor used to
/// open the commitment in [`RangeStatement`].
#[derive(Clone, Debug)]
pub struct RangeWitness {
    /// Cleartext value being range-bound.
    pub value: u64,
    /// Pedersen blinding scalar.
    pub blinding: Scalar,
}

/// Output of [`prove_range`]: the serialised proof plus its
/// commitment (echoed for convenience so the verifier sees a single
/// envelope).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeProofBytes {
    /// Compressed proof bytes (Bulletproofs canonical encoding).
    #[serde(with = "serde_bytes")]
    pub proof: Vec<u8>,
    /// Commitment the proof binds to.
    #[serde(with = "serde_bytes")]
    pub commitment: [u8; 32],
    /// Bit width the proof was generated for.
    pub bit_length: u32,
}

fn check_bit_length(n: u32) -> Result<usize, ZkError> {
    match n {
        8 | 16 | 32 | 64 => Ok(n as usize),
        _ => Err(ZkError::UnsupportedBitLength(n)),
    }
}

fn check_witness_fits(value: u64, bit_length: u32) -> Result<(), ZkError> {
    if bit_length == 64 || value < (1u64 << bit_length) {
        Ok(())
    } else {
        Err(ZkError::WitnessOutOfRange { value, bit_length })
    }
}

/// Commit to `value` under a fresh Pedersen blinding sampled from
/// `OsRng`. Returns `(commitment, blinding)`.
pub fn pedersen_commit(value: u64) -> (CompressedRistretto, Scalar) {
    let pc_gens = PedersenGens::default();
    let blinding = Scalar::random(&mut OsRng);
    let commitment = pc_gens.commit(Scalar::from(value), blinding).compress();
    (commitment, blinding)
}

/// Produce a Bulletproof that `witness.value ∈ [0, 2^bit_length)`,
/// bound to the Pedersen commitment that `witness.value` /
/// `witness.blinding` open.
///
/// The returned [`RangeProofBytes::commitment`] equals
/// `pc_gens.commit(value, blinding)` and *should* match the
/// `commitment` field of any matching [`RangeStatement`] the caller
/// later builds.
pub fn prove_range(witness: &RangeWitness, bit_length: u32, label: &'static [u8]) -> Result<RangeProofBytes, ZkError> {
    let n = check_bit_length(bit_length)?;
    check_witness_fits(witness.value, bit_length)?;

    let pc_gens = PedersenGens::default();
    let bp_gens = BulletproofGens::new(n, 1);
    let mut transcript = Transcript::new(label);

    let (proof, commitment) = RangeProof::prove_single(
        &bp_gens,
        &pc_gens,
        &mut transcript,
        witness.value,
        &witness.blinding,
        n,
    )
    .map_err(|e| ZkError::Encoding(format!("range-proof prove failed: {e:?}")))?;

    Ok(RangeProofBytes {
        proof: proof.to_bytes(),
        commitment: commitment.to_bytes(),
        bit_length,
    })
}

/// Verify a Bulletproof produced by [`prove_range`] against a
/// declared [`RangeStatement`].
pub fn verify_range(statement: &RangeStatement, bytes: &RangeProofBytes) -> Result<bool, ZkError> {
    if statement.bit_length != bytes.bit_length {
        return Ok(false);
    }
    if statement.commitment.to_bytes() != bytes.commitment {
        return Ok(false);
    }
    let n = check_bit_length(statement.bit_length)?;

    let proof = RangeProof::from_bytes(&bytes.proof)
        .map_err(|e| ZkError::BulletproofVerification(format!("decode: {e:?}")))?;
    let pc_gens = PedersenGens::default();
    let bp_gens = BulletproofGens::new(n, 1);
    let mut transcript = Transcript::new(statement.label);

    let commitment = CompressedRistretto(bytes.commitment);

    match proof.verify_single(&bp_gens, &pc_gens, &mut transcript, &commitment, n) {
        Ok(()) => Ok(true),
        Err(e) => Err(ZkError::BulletproofVerification(format!("{e:?}"))),
    }
}

/// [`ZkScheme`] adapter exposing the range-proof façade through the
/// scheme-agnostic trait surface.
#[derive(Clone, Copy, Debug, Default)]
pub struct BulletproofsRangeScheme;

impl ZkScheme for BulletproofsRangeScheme {
    type Statement = RangeStatement;
    type Witness = RangeWitness;

    fn name(&self) -> &'static str {
        "bulletproofs"
    }

    fn keygen(&self, _statement: &Self::Statement) -> Result<(ProvingKey, VerifyingKey), ZkError> {
        // Bulletproofs has no trusted setup; keys are the
        // deterministic generator bases, so we encode an empty payload
        // and rely on the verifier to rebuild generators from the
        // declared bit length.
        Ok((ProvingKey(Vec::new()), VerifyingKey(Vec::new())))
    }

    fn prove(
        &self,
        _pk: &ProvingKey,
        statement: &Self::Statement,
        witness: &Self::Witness,
    ) -> Result<Proof, ZkError> {
        let bytes = prove_range(witness, statement.bit_length, statement.label)?;
        let encoded = serde_cbor_encode(&bytes)?;
        Ok(Proof(encoded))
    }

    fn verify(
        &self,
        _vk: &VerifyingKey,
        statement: &Self::Statement,
        proof: &Proof,
    ) -> Result<bool, ZkError> {
        let bytes: RangeProofBytes = serde_cbor_decode(&proof.0)?;
        verify_range(statement, &bytes)
    }
}

// Local CBOR helpers — we keep `serde_cbor` out of `lib.rs` exports
// to avoid leaking a transitive dependency surface.
fn serde_cbor_encode(value: &RangeProofBytes) -> Result<Vec<u8>, ZkError> {
    // The crate already has `serde_cbor` available as a dev-dep for
    // tests; for the main path we hand-roll a minimal length-prefixed
    // serialisation so we don't pull `serde_cbor` into normal builds.
    let mut out = Vec::with_capacity(4 + 32 + value.proof.len() + 4);
    out.extend_from_slice(&value.bit_length.to_be_bytes());
    out.extend_from_slice(&value.commitment);
    out.extend_from_slice(&(value.proof.len() as u32).to_be_bytes());
    out.extend_from_slice(&value.proof);
    Ok(out)
}

fn serde_cbor_decode(bytes: &[u8]) -> Result<RangeProofBytes, ZkError> {
    if bytes.len() < 4 + 32 + 4 {
        return Err(ZkError::Encoding("range-proof envelope too short".to_string()));
    }
    let (bit_bytes, rest) = bytes.split_at(4);
    let bit_length = u32::from_be_bytes([bit_bytes[0], bit_bytes[1], bit_bytes[2], bit_bytes[3]]);
    let (commit_bytes, rest) = rest.split_at(32);
    let mut commitment = [0u8; 32];
    commitment.copy_from_slice(commit_bytes);
    let (len_bytes, rest) = rest.split_at(4);
    let proof_len =
        u32::from_be_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
    if rest.len() != proof_len {
        return Err(ZkError::Encoding(format!(
            "range-proof envelope length mismatch: declared {proof_len}, have {}",
            rest.len()
        )));
    }
    Ok(RangeProofBytes {
        proof: rest.to_vec(),
        commitment,
        bit_length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_envelope() {
        let bytes = RangeProofBytes {
            proof: vec![1, 2, 3, 4, 5],
            commitment: [9u8; 32],
            bit_length: 32,
        };
        let enc = serde_cbor_encode(&bytes).expect("encode");
        let dec = serde_cbor_decode(&enc).expect("decode");
        assert_eq!(dec.proof, bytes.proof);
        assert_eq!(dec.commitment, bytes.commitment);
        assert_eq!(dec.bit_length, bytes.bit_length);
    }
}
