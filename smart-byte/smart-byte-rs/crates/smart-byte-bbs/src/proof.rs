//! BBS+ selective-disclosure proof of knowledge.
//!
//! Reference: `draft-irtf-cfrg-bbs-signatures-08` § 3.7 (`ProofGen`)
//! and § 3.8 (`ProofVerify`); Au, Susilo, Mu, "Constant-Size Dynamic
//! k-TAA" (CDL/BBS+); and Camenisch, Drijvers, Lehmann, "Anonymous
//! Attestation Using the Strong Diffie Hellman Assumption Revisited"
//! (the modern BBS+ proof of knowledge).
//!
//! ## Construction
//!
//! Given a signature `(A, e)` on messages `(m_1, ..., m_n)` with
//! generators `(H_0, H_1, ..., H_n)`, the holder proves
//!
//! ```text
//!   "I know (e, m_{undisclosed}) and randomness (r1, r2, r3) such that
//!    e(A', W) == e(A_bar, g2)
//!    AND D == B * r1 + H_0 * r2
//!    AND A_bar - D == A' * (-e) + H_0 * (-s') - sum_{i in U} H_i * m_i"
//! ```
//!
//! where:
//!
//! * `A' = A * r1` — randomised signature commitment.
//! * `A_bar = B * r1 - A' * e` — companion in the pairing identity.
//! * `D = B * r1 + H_0 * r2` — Pedersen commitment to `B * r1` with
//!   additional blinding `r2`.
//! * `s' = r2 / r1` — the "scalar substitution" that links the two
//!   relations.
//!
//! The Fiat-Shamir transcript hashes `(A', A_bar, D, T1, T2, public
//! key, nonce, disclosed messages)` to derive the challenge `c`. The
//! responses `(e_hat, r2_hat, r3_hat, s_hat, undisclosed_responses)`
//! are openings of the corresponding witnesses.
//!
//! Verifier reconstructs the commitments using the responses and the
//! recomputed challenge and accepts iff the recomputed challenge
//! matches `c` *and* the pairing identity holds. Two independent
//! presentations of the *same* signature randomise `r1, r2` afresh
//! and so produce different `(A', A_bar, D)` and different challenges
//! — this is the unlinkability property.

use std::collections::{BTreeMap, BTreeSet};

use bls12_381::{
    G1Affine, G1Projective, G2Affine, G2Prepared, G2Projective, Scalar,
    multi_miller_loop,
};
use ff::Field;
use group::Group;
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::encode::{
    DST_FIAT_SHAMIR, g1_to_bytes, g2_to_bytes, scalar_to_bytes,
};
use crate::error::BbsError;
use crate::keys::PublicKey;
use crate::sign::{Signature, compute_b};

/// A BBS+ selective-disclosure proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisclosureProof {
    /// `A' = A * r1`.
    pub a_prime: G1Affine,
    /// `A_bar = B * r1 - A' * e`.
    pub a_bar: G1Affine,
    /// `D = B * r1 + H_0 * r2`.
    pub d: G1Affine,
    /// Fiat-Shamir challenge.
    pub c: Scalar,
    /// Response for the `e` witness.
    pub e_hat: Scalar,
    /// Response for the `r2` witness.
    pub r2_hat: Scalar,
    /// Response for the `r3 = 1/r1` witness.
    pub r3_hat: Scalar,
    /// Response for the `s' = r2 * r3` witness.
    pub s_hat: Scalar,
    /// Messages the holder chose to disclose, keyed by their position
    /// in the original message vector.
    pub disclosed_messages: BTreeMap<usize, Scalar>,
    /// Schnorr responses for each undisclosed message, in ascending
    /// index order over the undisclosed positions.
    pub undisclosed_responses: Vec<Scalar>,
}

/// Serde shadow with `Vec<u8>` placeholders for the curve types.
#[derive(Serialize, Deserialize)]
struct DisclosureProofShadow {
    #[serde(with = "serde_bytes")]
    a_prime: Vec<u8>,
    #[serde(with = "serde_bytes")]
    a_bar: Vec<u8>,
    #[serde(with = "serde_bytes")]
    d: Vec<u8>,
    #[serde(with = "serde_bytes")]
    c: Vec<u8>,
    #[serde(with = "serde_bytes")]
    e_hat: Vec<u8>,
    #[serde(with = "serde_bytes")]
    r2_hat: Vec<u8>,
    #[serde(with = "serde_bytes")]
    r3_hat: Vec<u8>,
    #[serde(with = "serde_bytes")]
    s_hat: Vec<u8>,
    disclosed_messages: Vec<(u64, serde_bytes::ByteBuf)>,
    undisclosed_responses: Vec<serde_bytes::ByteBuf>,
}

impl Serialize for DisclosureProof {
    fn serialize<S: serde::Serializer>(
        &self,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let shadow = DisclosureProofShadow {
            a_prime: g1_to_bytes(&self.a_prime).to_vec(),
            a_bar: g1_to_bytes(&self.a_bar).to_vec(),
            d: g1_to_bytes(&self.d).to_vec(),
            c: scalar_to_bytes(&self.c).to_vec(),
            e_hat: scalar_to_bytes(&self.e_hat).to_vec(),
            r2_hat: scalar_to_bytes(&self.r2_hat).to_vec(),
            r3_hat: scalar_to_bytes(&self.r3_hat).to_vec(),
            s_hat: scalar_to_bytes(&self.s_hat).to_vec(),
            disclosed_messages: self
                .disclosed_messages
                .iter()
                .map(|(k, v)| {
                    (
                        *k as u64,
                        serde_bytes::ByteBuf::from(scalar_to_bytes(v).to_vec()),
                    )
                })
                .collect(),
            undisclosed_responses: self
                .undisclosed_responses
                .iter()
                .map(|v| serde_bytes::ByteBuf::from(scalar_to_bytes(v).to_vec()))
                .collect(),
        };
        shadow.serialize(s)
    }
}

fn fixed_g1(bytes: &[u8]) -> Result<G1Affine, String> {
    if bytes.len() != 48 {
        return Err(format!("expected 48 G1 bytes, got {}", bytes.len()));
    }
    let mut b = [0u8; 48];
    b.copy_from_slice(bytes);
    crate::encode::g1_from_bytes(&b).map_err(|e| e.to_string())
}

fn fixed_scalar(bytes: &[u8]) -> Result<Scalar, String> {
    if bytes.len() != 32 {
        return Err(format!("expected 32 scalar bytes, got {}", bytes.len()));
    }
    let mut b = [0u8; 32];
    b.copy_from_slice(bytes);
    crate::encode::scalar_from_bytes(&b).map_err(|e| e.to_string())
}

impl<'de> Deserialize<'de> for DisclosureProof {
    fn deserialize<D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<Self, D::Error> {
        let shadow = DisclosureProofShadow::deserialize(d)?;
        let mut disclosed_messages = BTreeMap::new();
        for (k, v) in shadow.disclosed_messages {
            let scalar = fixed_scalar(&v).map_err(serde::de::Error::custom)?;
            disclosed_messages.insert(k as usize, scalar);
        }
        let mut undisclosed_responses = Vec::with_capacity(
            shadow.undisclosed_responses.len(),
        );
        for v in shadow.undisclosed_responses {
            undisclosed_responses
                .push(fixed_scalar(&v).map_err(serde::de::Error::custom)?);
        }
        Ok(DisclosureProof {
            a_prime: fixed_g1(&shadow.a_prime)
                .map_err(serde::de::Error::custom)?,
            a_bar: fixed_g1(&shadow.a_bar)
                .map_err(serde::de::Error::custom)?,
            d: fixed_g1(&shadow.d).map_err(serde::de::Error::custom)?,
            c: fixed_scalar(&shadow.c).map_err(serde::de::Error::custom)?,
            e_hat: fixed_scalar(&shadow.e_hat)
                .map_err(serde::de::Error::custom)?,
            r2_hat: fixed_scalar(&shadow.r2_hat)
                .map_err(serde::de::Error::custom)?,
            r3_hat: fixed_scalar(&shadow.r3_hat)
                .map_err(serde::de::Error::custom)?,
            s_hat: fixed_scalar(&shadow.s_hat)
                .map_err(serde::de::Error::custom)?,
            disclosed_messages,
            undisclosed_responses,
        })
    }
}

fn random_nonzero_scalar<R: RngCore + CryptoRng>(rng: &mut R) -> Scalar {
    loop {
        let mut bytes = [0u8; 64];
        rng.fill_bytes(&mut bytes);
        let s = Scalar::from_bytes_wide(&bytes);
        if !bool::from(s.ct_eq(&Scalar::ZERO)) {
            return s;
        }
    }
}

fn disclosed_partition(
    n: usize,
    disclosed: &[usize],
) -> Result<(BTreeSet<usize>, Vec<usize>), BbsError> {
    let mut set = BTreeSet::new();
    for &i in disclosed {
        if i >= n {
            return Err(BbsError::DisclosedIndexOutOfRange { index: i, len: n });
        }
        if !set.insert(i) {
            return Err(BbsError::DisclosedIndexDuplicate(i));
        }
    }
    let undisclosed: Vec<usize> = (0..n).filter(|i| !set.contains(i)).collect();
    Ok((set, undisclosed))
}

/// Fiat-Shamir transcript over commitments + context. Hashes to a
/// scalar via the project-wide `hash_to_scalar` reduction (SHA-512
/// reduce) seeded with a SHA-256 fingerprint of the byte transcript.
#[allow(clippy::too_many_arguments)]
fn challenge_scalar(
    a_prime: &G1Affine,
    a_bar: &G1Affine,
    d: &G1Affine,
    t1: &G1Affine,
    t2: &G1Affine,
    public_key: &PublicKey,
    disclosed: &BTreeMap<usize, Scalar>,
    nonce: &[u8],
) -> Scalar {
    let mut h = Sha256::new();
    h.update(DST_FIAT_SHAMIR);
    h.update(g1_to_bytes(a_prime));
    h.update(g1_to_bytes(a_bar));
    h.update(g1_to_bytes(d));
    h.update(g1_to_bytes(t1));
    h.update(g1_to_bytes(t2));
    h.update(g2_to_bytes(&public_key.w));
    h.update((disclosed.len() as u64).to_be_bytes());
    for (idx, m) in disclosed {
        h.update((*idx as u64).to_be_bytes());
        h.update(scalar_to_bytes(m));
    }
    h.update((nonce.len() as u64).to_be_bytes());
    h.update(nonce);
    let digest = h.finalize();
    crate::encode::hash_to_scalar(DST_FIAT_SHAMIR, &digest)
}

/// Create a selective-disclosure proof from a BBS+ signature.
///
/// * `signature` — the signer's `(A, e)`.
/// * `public_key` — issuer's BBS+ public key.
/// * `messages` — the *full* message vector that was originally
///   signed.
/// * `disclosed_indices` — positions in `messages` the holder is
///   willing to reveal.
/// * `nonce` — verifier-supplied freshness binding.
/// * `generators` — the same generator vector used by the signer.
/// * `rng` — cryptographic randomness.
pub fn create_proof<R: RngCore + CryptoRng>(
    signature: &Signature,
    public_key: &PublicKey,
    messages: &[Scalar],
    disclosed_indices: &[usize],
    nonce: &[u8],
    generators: &[G1Projective],
    rng: &mut R,
) -> Result<DisclosureProof, BbsError> {
    let n = messages.len();
    if generators.len() < n + 1 {
        return Err(BbsError::GeneratorCount {
            have: generators.len(),
            need: n + 1,
        });
    }
    let (disclosed_set, undisclosed) = disclosed_partition(n, disclosed_indices)?;

    // Build the commitment B that the signer used.
    let b = compute_b(messages, generators)?;
    let h0 = generators[0];

    // Random blinding scalars. `r1` is drawn nonzero so its inverse
    // exists; we still surface the negligible-probability failure
    // rather than panic.
    let r1 = random_nonzero_scalar(rng);
    let r2 = random_nonzero_scalar(rng);
    let Some(r3) = Option::<Scalar>::from(r1.invert()) else {
        return Err(BbsError::ProofVerification(
            "r1 inversion failed (impossible: r1 is nonzero)".into(),
        ));
    };
    let s_prime = r2 * r3; // r2 / r1

    // Public proof commitments.
    //   A' = A * r1
    //   A_bar = B*r1 - A'*e          (so e(A', W) == e(A_bar, g2))
    //   D = B*r1 + H_0*r2
    let a_prime_p = G1Projective::from(signature.a) * r1;
    let a_prime = G1Affine::from(a_prime_p);
    let a_bar_p = b * r1 - a_prime_p * signature.e;
    let a_bar = G1Affine::from(a_bar_p);
    let d_p = b * r1 + h0 * r2;
    let d = G1Affine::from(d_p);

    // Schnorr blinding values.
    let e_tilde = random_nonzero_scalar(rng);
    let r2_tilde = random_nonzero_scalar(rng);
    let r3_tilde = random_nonzero_scalar(rng);
    let s_tilde = random_nonzero_scalar(rng);
    let m_tildes: Vec<Scalar> = (0..undisclosed.len())
        .map(|_| random_nonzero_scalar(rng))
        .collect();

    // Relation 1 (witness `e, r2`):
    //   A_bar - D == A'*(-e) + H_0*(-r2)
    // Blinded T1:
    //   T1 = A'*(-e_tilde) + H_0*(-r2_tilde)
    let mut t1_p = G1Projective::identity();
    t1_p += a_prime_p * (-e_tilde);
    t1_p += h0 * (-r2_tilde);
    let t1 = G1Affine::from(t1_p);

    // Relation 2 (witness `r3, s', m_i for i in undisclosed`):
    //   H_0 + sum_{i in D} H_{i+1}*m_i  ==
    //         D*r3 - H_0*s' - sum_{i in U} H_{i+1}*m_i
    // Blinded T2:
    //   T2 = D*r3_tilde - H_0*s_tilde - sum_{i in U} H_{i+1}*m_tilde_i
    let mut t2_p = G1Projective::identity();
    t2_p += d_p * r3_tilde;
    t2_p -= h0 * s_tilde;
    for (slot, &idx) in undisclosed.iter().enumerate() {
        t2_p -= generators[idx + 1] * m_tildes[slot];
    }
    let t2 = G1Affine::from(t2_p);

    // Fiat-Shamir challenge.
    let disclosed_messages: BTreeMap<usize, Scalar> = disclosed_set
        .iter()
        .map(|&i| (i, messages[i]))
        .collect();
    let c = challenge_scalar(
        &a_prime,
        &a_bar,
        &d,
        &t1,
        &t2,
        public_key,
        &disclosed_messages,
        nonce,
    );

    // Schnorr responses: response = tilde + c * witness
    let e_hat = e_tilde + c * signature.e;
    let r2_hat = r2_tilde + c * r2;
    let r3_hat = r3_tilde + c * r3;
    let s_hat = s_tilde + c * s_prime;
    let undisclosed_responses: Vec<Scalar> = undisclosed
        .iter()
        .zip(m_tildes.iter())
        .map(|(&idx, &m_tilde)| m_tilde + c * messages[idx])
        .collect();

    Ok(DisclosureProof {
        a_prime,
        a_bar,
        d,
        c,
        e_hat,
        r2_hat,
        r3_hat,
        s_hat,
        disclosed_messages,
        undisclosed_responses,
    })
}

/// Verify a selective-disclosure proof.
pub fn verify_proof(
    proof: &DisclosureProof,
    public_key: &PublicKey,
    nonce: &[u8],
    generators: &[G1Projective],
) -> Result<(), BbsError> {
    // Determine n from disclosed indices + undisclosed responses count.
    // The verifier knows the total message count because it equals
    // `disclosed.len() + undisclosed_responses.len()`.
    let n = proof.disclosed_messages.len() + proof.undisclosed_responses.len();
    if generators.len() < n + 1 {
        return Err(BbsError::GeneratorCount {
            have: generators.len(),
            need: n + 1,
        });
    }

    // Reject degenerate A'.
    if bool::from(G1Projective::from(proof.a_prime).is_identity()) {
        return Err(BbsError::ProofVerification("A' is identity".into()));
    }

    // 1. Reconstruct the undisclosed-index set.
    let mut disclosed_set = BTreeSet::new();
    for &i in proof.disclosed_messages.keys() {
        if i >= n {
            return Err(BbsError::DisclosedIndexOutOfRange { index: i, len: n });
        }
        if !disclosed_set.insert(i) {
            return Err(BbsError::DisclosedIndexDuplicate(i));
        }
    }
    let undisclosed: Vec<usize> =
        (0..n).filter(|i| !disclosed_set.contains(i)).collect();
    if undisclosed.len() != proof.undisclosed_responses.len() {
        return Err(BbsError::ProofVerification(
            "undisclosed response count mismatch".into(),
        ));
    }

    let h0 = generators[0];

    let a_prime_p = G1Projective::from(proof.a_prime);
    let a_bar_p = G1Projective::from(proof.a_bar);
    let d_p = G1Projective::from(proof.d);

    // 2. Recompute T1 from responses.
    //    Relation: A_bar - D == A'*(-e) + H_0*(-r2)
    //    Reconstruction:
    //      T1' = -A'*e_hat - H_0*r2_hat + c*(D - A_bar)
    let mut t1_recon = G1Projective::identity();
    t1_recon -= a_prime_p * proof.e_hat;
    t1_recon -= h0 * proof.r2_hat;
    t1_recon += (d_p - a_bar_p) * proof.c;

    // 3. Recompute T2 from responses.
    //    Relation: H_0 + sum_{D} H_{i+1}*m_i == D*r3 - H_0*s' - sum_{U} H_{i+1}*m_i
    //    Reconstruction:
    //      T2' = D*r3_hat - H_0*s_hat - sum_{U} H_{i+1}*m_hat_i
    //              - c*(H_0 + sum_{D} H_{i+1}*m_disclosed_i)
    let mut t2_recon = G1Projective::identity();
    t2_recon += d_p * proof.r3_hat;
    t2_recon -= h0 * proof.s_hat;
    for (slot, &idx) in undisclosed.iter().enumerate() {
        t2_recon -= generators[idx + 1] * proof.undisclosed_responses[slot];
    }
    // Subtract c * (H_0 + sum disclosed)
    let mut disclosed_lhs = h0;
    for (&idx, m) in &proof.disclosed_messages {
        disclosed_lhs += generators[idx + 1] * m;
    }
    t2_recon -= disclosed_lhs * proof.c;

    let t1_aff = G1Affine::from(t1_recon);
    let t2_aff = G1Affine::from(t2_recon);

    // 4. Recompute the challenge.
    let c_recon = challenge_scalar(
        &proof.a_prime,
        &proof.a_bar,
        &proof.d,
        &t1_aff,
        &t2_aff,
        public_key,
        &proof.disclosed_messages,
        nonce,
    );
    if !bool::from(c_recon.ct_eq(&proof.c)) {
        return Err(BbsError::ProofVerification(
            "challenge mismatch".into(),
        ));
    }

    // 5. Pairing identity:  e(A', W) == e(A_bar, g2)
    let prep_w = G2Prepared::from(public_key.w);
    let prep_g2 = G2Prepared::from(G2Affine::generator());
    let neg_a_bar = G1Affine::from(-G1Projective::from(proof.a_bar));
    let gt = multi_miller_loop(&[
        (&proof.a_prime, &prep_w),
        (&neg_a_bar, &prep_g2),
    ])
    .final_exponentiation();
    if !bool::from(gt.is_identity()) {
        return Err(BbsError::ProofVerification(
            "pairing identity failed".into(),
        ));
    }

    let _ = G2Projective::generator(); // anchor the import (used elsewhere)

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::message_to_scalar;
    use crate::generators::message_generators;
    use crate::keys::keygen;
    use crate::sign::sign;
    use rand::rngs::OsRng;

    fn fixture(n: usize) -> (crate::keys::KeyPair, Vec<Scalar>, Vec<G1Projective>) {
        let kp = keygen(&mut OsRng);
        let msgs: Vec<Scalar> = (0..n)
            .map(|i| message_to_scalar(format!("msg-{i}").as_bytes()))
            .collect();
        let gens = message_generators(b"smart-byte-bbs-proof-test", n + 2);
        (kp, msgs, gens)
    }

    #[test]
    fn disclose_subset_verifies() {
        let (kp, msgs, gens) = fixture(5);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        let proof = create_proof(
            &sig,
            &kp.public,
            &msgs,
            &[1, 3],
            b"nonce-1",
            &gens,
            &mut OsRng,
        )
        .unwrap();
        verify_proof(&proof, &kp.public, b"nonce-1", &gens).unwrap();
        assert_eq!(proof.disclosed_messages.len(), 2);
        assert_eq!(proof.undisclosed_responses.len(), 3);
    }

    #[test]
    fn disclose_none_verifies() {
        let (kp, msgs, gens) = fixture(3);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        let proof = create_proof(
            &sig,
            &kp.public,
            &msgs,
            &[],
            b"n",
            &gens,
            &mut OsRng,
        )
        .unwrap();
        verify_proof(&proof, &kp.public, b"n", &gens).unwrap();
    }

    #[test]
    fn disclose_all_verifies() {
        let (kp, msgs, gens) = fixture(3);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        let proof = create_proof(
            &sig,
            &kp.public,
            &msgs,
            &[0, 1, 2],
            b"n",
            &gens,
            &mut OsRng,
        )
        .unwrap();
        verify_proof(&proof, &kp.public, b"n", &gens).unwrap();
    }

    #[test]
    fn wrong_nonce_fails() {
        let (kp, msgs, gens) = fixture(4);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        let proof = create_proof(
            &sig,
            &kp.public,
            &msgs,
            &[2],
            b"good-nonce",
            &gens,
            &mut OsRng,
        )
        .unwrap();
        assert!(
            verify_proof(&proof, &kp.public, b"bad-nonce", &gens).is_err()
        );
    }

    #[test]
    fn wrong_public_key_fails() {
        let (kp, msgs, gens) = fixture(4);
        let other = keygen(&mut OsRng);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        let proof = create_proof(
            &sig,
            &kp.public,
            &msgs,
            &[2],
            b"n",
            &gens,
            &mut OsRng,
        )
        .unwrap();
        assert!(verify_proof(&proof, &other.public, b"n", &gens).is_err());
    }

    #[test]
    fn tampered_disclosed_message_fails() {
        let (kp, msgs, gens) = fixture(4);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        let mut proof = create_proof(
            &sig,
            &kp.public,
            &msgs,
            &[1, 2],
            b"n",
            &gens,
            &mut OsRng,
        )
        .unwrap();
        // Flip one disclosed value.
        if let Some(v) = proof.disclosed_messages.get_mut(&1) {
            *v += Scalar::ONE;
        }
        assert!(verify_proof(&proof, &kp.public, b"n", &gens).is_err());
    }

    #[test]
    fn tampered_response_fails() {
        let (kp, msgs, gens) = fixture(4);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        let mut proof = create_proof(
            &sig,
            &kp.public,
            &msgs,
            &[1],
            b"n",
            &gens,
            &mut OsRng,
        )
        .unwrap();
        proof.e_hat += Scalar::ONE;
        assert!(verify_proof(&proof, &kp.public, b"n", &gens).is_err());
    }

    #[test]
    fn unlinkability_two_proofs_are_distinct() {
        let (kp, msgs, gens) = fixture(5);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        let p1 = create_proof(
            &sig,
            &kp.public,
            &msgs,
            &[1, 3],
            b"n",
            &gens,
            &mut OsRng,
        )
        .unwrap();
        let p2 = create_proof(
            &sig,
            &kp.public,
            &msgs,
            &[1, 3],
            b"n",
            &gens,
            &mut OsRng,
        )
        .unwrap();
        // Same credential, identical disclosure, but the wire bits
        // are *cryptographically distinct* — this is the BBS+
        // unlinkability property.
        assert_ne!(p1.a_prime, p2.a_prime);
        assert_ne!(p1.a_bar, p2.a_bar);
        assert_ne!(p1.d, p2.d);
        assert_ne!(p1.c, p2.c);
        // Both still verify.
        verify_proof(&p1, &kp.public, b"n", &gens).unwrap();
        verify_proof(&p2, &kp.public, b"n", &gens).unwrap();
    }

    #[test]
    fn proof_serde_roundtrips() {
        let (kp, msgs, gens) = fixture(3);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        let proof = create_proof(
            &sig,
            &kp.public,
            &msgs,
            &[1],
            b"n",
            &gens,
            &mut OsRng,
        )
        .unwrap();
        let bytes = serde_cbor::to_vec(&proof).unwrap();
        let back: DisclosureProof = serde_cbor::from_slice(&bytes).unwrap();
        assert_eq!(back, proof);
        verify_proof(&back, &kp.public, b"n", &gens).unwrap();
    }

    #[test]
    fn out_of_range_disclosed_index_errors() {
        let (kp, msgs, gens) = fixture(3);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        let err = create_proof(
            &sig, &kp.public, &msgs, &[10], b"n", &gens, &mut OsRng,
        )
        .unwrap_err();
        assert!(matches!(err, BbsError::DisclosedIndexOutOfRange { .. }));
    }
}
