//! W3C `bbs-2023` Data Integrity cryptosuite.
//!
//! Reference: W3C Verifiable Credential Data Integrity 1.0 + the
//! `vc-di-bbs` companion specification (Editor's Draft, 2024-12).
//!
//! ## Scope
//!
//! `bbs-2023` is a Data Integrity cryptosuite that encodes BBS+
//! signatures over an *RDF dataset canonicalisation* of the credential.
//! Full canonicalisation requires URDNA2015 + JSON-LD expansion which
//! drag in heavy dependencies (the `rdf-canon` feature in
//! `smart-byte-vc` gates that path). This module ships the
//! *JCS variant* of the cryptosuite — sign the JCS-canonical encoding
//! of the credential minus its `proof` array — so the smart-byte
//! reference implementation can issue and verify BBS+-secured VCs
//! without the URDNA2015 toolchain.
//!
//! This matches the pragmatic convention adopted by AnonCreds 2.0 and
//! by the IETF SD-JWT-BBS interop suite: BBS+ over a stable canonical
//! byte representation of the issuer-asserted claims, with the message
//! vector being one scalar per top-level claim.

use serde::{Deserialize, Serialize};

use crate::encode::message_to_scalar;
use crate::error::BbsError;
use crate::generators::message_generators;
use crate::keys::{PublicKey, SecretKey};
use crate::proof::{DisclosureProof, create_proof, verify_proof};
use crate::sign::{Signature, sign, verify};

/// Cryptosuite identifier as recorded in
/// [`DataIntegrityProof::cryptosuite`].
pub const CRYPTOSUITE_BBS_2023: &str = "bbs-2023";

/// Domain string used when deriving generators for the cryptosuite.
/// Includes the cryptosuite name so generators from different suites
/// cannot collide.
pub const CRYPTOSUITE_GENERATOR_DOMAIN: &[u8] = b"smart-byte-bbs-2023";

/// A `Bbs2023Suite` proof payload — what travels in
/// `DataIntegrityProof::proof_value` for an issuance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bbs2023IssuanceProof {
    /// The compact BBS+ signature `(A, e)` (80 bytes).
    #[serde(with = "serde_bytes_array_80")]
    pub signature: [u8; 80],
    /// The number of messages signed. Required so the verifier can
    /// reproduce the generator vector.
    pub message_count: u32,
}

/// A `Bbs2023Suite` proof payload — what travels in
/// `DataIntegrityProof::proof_value` for a holder presentation
/// (selective-disclosure proof).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bbs2023DisclosureProof {
    /// CBOR-encoded [`crate::proof::DisclosureProof`].
    #[serde(with = "serde_bytes")]
    pub disclosure: Vec<u8>,
    /// Total message count of the underlying signed credential.
    pub message_count: u32,
}

/// Custom serde shim for fixed-size `[u8; 80]` over byte string.
mod serde_bytes_array_80 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Serialise the 80-byte array as a byte string.
    pub fn serialize<S: Serializer>(
        bytes: &[u8; 80],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        serde_bytes::Bytes::new(bytes).serialize(s)
    }

    /// Deserialise the 80-byte array from a byte string.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[u8; 80], D::Error> {
        let bb: serde_bytes::ByteBuf = serde_bytes::ByteBuf::deserialize(d)?;
        if bb.len() != 80 {
            return Err(serde::de::Error::custom(format!(
                "expected 80 bytes, got {}",
                bb.len()
            )));
        }
        let mut out = [0u8; 80];
        out.copy_from_slice(&bb);
        Ok(out)
    }
}

/// Trait surface for proof-suite implementations, aligned with the
/// trait that `smart-byte-vc::proof` would expect for cryptosuite
/// pluggability. `smart-byte-vc` currently inlines its
/// `eddsa-jcs-2022` handling, so this trait is local to the BBS+
/// crate; downstream crates can adopt it without re-exporting.
pub trait ProofSuite {
    /// The cryptosuite identifier as it appears in `DataIntegrityProof::cryptosuite`.
    fn name(&self) -> &'static str;
}

/// Marker type for the `bbs-2023` cryptosuite.
pub struct Bbs2023Suite;

impl ProofSuite for Bbs2023Suite {
    fn name(&self) -> &'static str {
        CRYPTOSUITE_BBS_2023
    }
}

/// Convert an ordered list of arbitrary claim byte strings into the
/// BBS+ scalar message vector.
pub fn claims_to_messages<I, B>(claims: I) -> Vec<bls12_381::Scalar>
where
    I: IntoIterator<Item = B>,
    B: AsRef<[u8]>,
{
    claims
        .into_iter()
        .map(|c| message_to_scalar(c.as_ref()))
        .collect()
}

/// Issue a BBS+ signature over `claims`. Generators are derived from
/// [`CRYPTOSUITE_GENERATOR_DOMAIN`] and the claim count.
pub fn issue_bbs_2023(
    claims: &[&[u8]],
    secret_key: &SecretKey,
    public_key: &PublicKey,
) -> Result<Bbs2023IssuanceProof, BbsError> {
    let msgs = claims_to_messages(claims.iter().copied());
    let gens = message_generators(
        CRYPTOSUITE_GENERATOR_DOMAIN,
        msgs.len() + 2,
    );
    let sig = sign(&msgs, secret_key, public_key, &gens)?;
    Ok(Bbs2023IssuanceProof {
        signature: sig.to_bytes(),
        message_count: msgs.len() as u32,
    })
}

/// Verify a `bbs-2023` issuance against the full claim vector. Used
/// when the verifier has the original (undisclosed) credential.
pub fn verify_bbs_2023_issuance(
    proof: &Bbs2023IssuanceProof,
    claims: &[&[u8]],
    public_key: &PublicKey,
) -> Result<(), BbsError> {
    if proof.message_count as usize != claims.len() {
        return Err(BbsError::ProofVerification(format!(
            "message count {} != claims len {}",
            proof.message_count,
            claims.len()
        )));
    }
    let sig = Signature::from_bytes(&proof.signature)?;
    let msgs = claims_to_messages(claims.iter().copied());
    let gens = message_generators(
        CRYPTOSUITE_GENERATOR_DOMAIN,
        msgs.len() + 2,
    );
    verify(&msgs, &sig, public_key, &gens)
}

/// Create a holder's selective-disclosure proof from a BBS+ issuance.
pub fn create_bbs_2023_disclosure(
    issuance: &Bbs2023IssuanceProof,
    claims: &[&[u8]],
    disclosed_indices: &[usize],
    public_key: &PublicKey,
    nonce: &[u8],
    rng: &mut (impl rand::RngCore + rand::CryptoRng),
) -> Result<Bbs2023DisclosureProof, BbsError> {
    if issuance.message_count as usize != claims.len() {
        return Err(BbsError::ProofVerification(format!(
            "issuance asserts {} messages but holder provided {}",
            issuance.message_count,
            claims.len()
        )));
    }
    let sig = Signature::from_bytes(&issuance.signature)?;
    let msgs = claims_to_messages(claims.iter().copied());
    let gens = message_generators(
        CRYPTOSUITE_GENERATOR_DOMAIN,
        msgs.len() + 2,
    );
    let proof = create_proof(
        &sig,
        public_key,
        &msgs,
        disclosed_indices,
        nonce,
        &gens,
        rng,
    )?;
    let disclosure = serde_cbor::to_vec(&proof)?;
    Ok(Bbs2023DisclosureProof {
        disclosure,
        message_count: issuance.message_count,
    })
}

/// Verify a holder's selective-disclosure proof.
pub fn verify_bbs_2023_disclosure(
    proof: &Bbs2023DisclosureProof,
    public_key: &PublicKey,
    nonce: &[u8],
) -> Result<DisclosureProof, BbsError> {
    let dp: DisclosureProof = serde_cbor::from_slice(&proof.disclosure)?;
    let gens = message_generators(
        CRYPTOSUITE_GENERATOR_DOMAIN,
        proof.message_count as usize + 2,
    );
    verify_proof(&dp, public_key, nonce, &gens)?;
    Ok(dp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::keygen;
    use rand::rngs::OsRng;

    #[test]
    fn cryptosuite_issuance_roundtrip() {
        let kp = keygen(&mut OsRng);
        let claims: Vec<&[u8]> = vec![
            b"id:did:example:alice",
            b"name:Alice",
            b"birth:1990-01-01",
            b"country:NZ",
        ];
        let iss = issue_bbs_2023(&claims, &kp.secret, &kp.public).unwrap();
        verify_bbs_2023_issuance(&iss, &claims, &kp.public).unwrap();
    }

    #[test]
    fn cryptosuite_disclosure_roundtrip() {
        let kp = keygen(&mut OsRng);
        let claims: Vec<&[u8]> = vec![
            b"id",
            b"name",
            b"age",
            b"country",
            b"role",
        ];
        let iss = issue_bbs_2023(&claims, &kp.secret, &kp.public).unwrap();
        let disc = create_bbs_2023_disclosure(
            &iss,
            &claims,
            &[1, 3],
            &kp.public,
            b"verifier-nonce",
            &mut OsRng,
        )
        .unwrap();
        let verified = verify_bbs_2023_disclosure(
            &disc,
            &kp.public,
            b"verifier-nonce",
        )
        .unwrap();
        assert_eq!(verified.disclosed_messages.len(), 2);
    }

    #[test]
    fn cryptosuite_name_matches_spec() {
        assert_eq!(Bbs2023Suite.name(), "bbs-2023");
    }

    #[test]
    fn issuance_serde_cbor_roundtrip() {
        let kp = keygen(&mut OsRng);
        let claims: Vec<&[u8]> = vec![b"a", b"b"];
        let iss = issue_bbs_2023(&claims, &kp.secret, &kp.public).unwrap();
        let bytes = serde_cbor::to_vec(&iss).unwrap();
        let back: Bbs2023IssuanceProof = serde_cbor::from_slice(&bytes).unwrap();
        assert_eq!(back, iss);
    }
}
