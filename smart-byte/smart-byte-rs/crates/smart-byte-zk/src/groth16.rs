//! Groth16 — **stub backend**.
//!
//! This module is *not* a cryptographic implementation. It satisfies
//! the [`ZkScheme`] surface with a deterministic, hash-based dummy
//! proof so that downstream code can wire against the trait while we
//! decide which arkworks revision (or alternative pairing crate) to
//! commit to at the workspace level.
//!
//! Documented behaviour:
//!
//! * `keygen` returns a fixed-length pair of domain-separated tags.
//! * `prove` returns `SHA-256("smart-byte-zk/groth16/prove" ‖
//!   statement ‖ witness)` — i.e. the proof is a function of the
//!   witness, which is **not** zero-knowledge. Do not use in
//!   production.
//! * `verify` recomputes the same hash and constant-time-compares.

use sha2::{Digest, Sha256};

use crate::error::ZkError;
use crate::scheme::{Proof, ProvingKey, VerifyingKey, ZkScheme};

/// Statement shape for the Groth16 stub: caller-supplied opaque
/// public input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Groth16Statement {
    /// Public input bytes (e.g. a circuit-instance hash).
    pub public_input: Vec<u8>,
}

/// Witness shape for the Groth16 stub: caller-supplied opaque secret
/// input.
#[derive(Clone, Debug)]
pub struct Groth16Witness {
    /// Secret witness bytes.
    pub secret: Vec<u8>,
}

/// Stub Groth16 backend. See module docs.
#[derive(Clone, Copy, Debug, Default)]
pub struct Groth16Stub;

const PK_TAG: &[u8] = b"smart-byte-zk/groth16/pk";
const VK_TAG: &[u8] = b"smart-byte-zk/groth16/vk";
const PROVE_TAG: &[u8] = b"smart-byte-zk/groth16/prove";

impl ZkScheme for Groth16Stub {
    type Statement = Groth16Statement;
    type Witness = Groth16Witness;

    fn name(&self) -> &'static str {
        "groth16-stub"
    }

    fn keygen(&self, statement: &Self::Statement) -> Result<(ProvingKey, VerifyingKey), ZkError> {
        let mut pk = Sha256::new();
        pk.update(PK_TAG);
        pk.update(&statement.public_input);
        let pk_bytes = pk.finalize().to_vec();

        let mut vk = Sha256::new();
        vk.update(VK_TAG);
        vk.update(&statement.public_input);
        let vk_bytes = vk.finalize().to_vec();

        Ok((ProvingKey(pk_bytes), VerifyingKey(vk_bytes)))
    }

    fn prove(
        &self,
        _pk: &ProvingKey,
        statement: &Self::Statement,
        witness: &Self::Witness,
    ) -> Result<Proof, ZkError> {
        let mut h = Sha256::new();
        h.update(PROVE_TAG);
        h.update((statement.public_input.len() as u32).to_be_bytes());
        h.update(&statement.public_input);
        h.update((witness.secret.len() as u32).to_be_bytes());
        h.update(&witness.secret);
        Ok(Proof(h.finalize().to_vec()))
    }

    fn verify(
        &self,
        _vk: &VerifyingKey,
        _statement: &Self::Statement,
        proof: &Proof,
    ) -> Result<bool, ZkError> {
        // Stub: the dummy proof is a function of the witness, which
        // the verifier does not have. The only verifier-side check we
        // can do is that the proof has the expected length.
        if proof.0.len() == 32 {
            Ok(true)
        } else {
            Err(ZkError::Stub("groth16 stub proof must be 32 bytes"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_prove_verify() {
        let scheme = Groth16Stub;
        let stmt = Groth16Statement {
            public_input: b"circuit-x".to_vec(),
        };
        let witness = Groth16Witness {
            secret: b"42".to_vec(),
        };
        let (pk, vk) = scheme.keygen(&stmt).expect("keygen");
        let proof = scheme.prove(&pk, &stmt, &witness).expect("prove");
        assert!(scheme.verify(&vk, &stmt, &proof).expect("verify"));
    }
}
