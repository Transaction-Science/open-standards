//! PLONK — **stub backend**.
//!
//! Same shape as [`crate::groth16`]: a deterministic, hash-based
//! dummy proof that satisfies the [`ZkScheme`] surface while we hold
//! the door open for a future PLONK-family integration (e.g.
//! `halo2`, `plonky2`, or arkworks-plonk).
//!
//! Documented behaviour mirrors the Groth16 stub: not
//! cryptographically binding, not zero-knowledge; only the trait
//! surface is real.

use sha2::{Digest, Sha256};

use crate::error::ZkError;
use crate::scheme::{Proof, ProvingKey, VerifyingKey, ZkScheme};

/// Statement shape for the PLONK stub.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlonkStatement {
    /// Public input bytes (e.g. a circuit-instance hash).
    pub public_input: Vec<u8>,
}

/// Witness shape for the PLONK stub.
#[derive(Clone, Debug)]
pub struct PlonkWitness {
    /// Secret witness bytes.
    pub secret: Vec<u8>,
}

/// Stub PLONK backend. See module docs.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlonkStub;

const PK_TAG: &[u8] = b"smart-byte-zk/plonk/pk";
const VK_TAG: &[u8] = b"smart-byte-zk/plonk/vk";
const PROVE_TAG: &[u8] = b"smart-byte-zk/plonk/prove";

impl ZkScheme for PlonkStub {
    type Statement = PlonkStatement;
    type Witness = PlonkWitness;

    fn name(&self) -> &'static str {
        "plonk-stub"
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
        if proof.0.len() == 32 {
            Ok(true)
        } else {
            Err(ZkError::Stub("plonk stub proof must be 32 bytes"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_prove_verify() {
        let scheme = PlonkStub;
        let stmt = PlonkStatement {
            public_input: b"circuit-y".to_vec(),
        };
        let witness = PlonkWitness {
            secret: b"99".to_vec(),
        };
        let (pk, vk) = scheme.keygen(&stmt).expect("keygen");
        let proof = scheme.prove(&pk, &stmt, &witness).expect("prove");
        assert!(scheme.verify(&vk, &stmt, &proof).expect("verify"));
    }
}
