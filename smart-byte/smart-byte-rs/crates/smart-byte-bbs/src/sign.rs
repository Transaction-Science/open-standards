//! BBS+ signing and verification.
//!
//! Reference: `draft-irtf-cfrg-bbs-signatures-08` § 3.5 (`Sign`) and
//! § 3.6 (`Verify`).
//!
//! ## Compact (A, e) form
//!
//! The IETF draft uses the compact signature representation
//! `(A, e)` ∈ G1 × Fr where:
//!
//! ```text
//! B   = P1 + sum_{i=1}^{n} H_i * msg_i
//! A   = B * (1 / (sk + e))
//! ```
//!
//! and `e ∈ Fr` is derived deterministically (the `Sign` algorithm
//! draws `e` via `hash_to_scalar` over the messages, secret key and
//! generators, which is what the draft does as of revision 06+; we
//! mirror that). Verification checks the pairing identity
//!
//! ```text
//! e(A, W + g2 * e) == e(B, g2)
//! ```
//!
//! Smart Byte's variant fixes `P1` to be the first generator emitted
//! by [`crate::generators::message_generators`] (call it `H_0`) so the
//! complete public parameter set is reproducible from a single domain
//! string + the message count. The per-message generators are
//! `H_1 .. H_n`.

use bls12_381::{
    G1Affine, G1Projective, G2Affine, G2Prepared, G2Projective, Scalar,
    multi_miller_loop,
};
use ff::Field;
use group::Group;
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;

use crate::encode::{
    DST_HASH_TO_SCALAR, g1_from_bytes, g1_to_bytes, hash_to_scalar,
    scalar_from_bytes, scalar_to_bytes,
};
use crate::error::BbsError;
use crate::keys::{PublicKey, SecretKey};

/// A BBS+ signature: `(A, e)` ∈ G1 × Fr.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    /// `A = B / (sk + e)`.
    pub a: G1Affine,
    /// The `e` component drawn deterministically from
    /// `hash_to_scalar(sk || messages)`.
    pub e: Scalar,
}

impl Serialize for Signature {
    fn serialize<S: serde::Serializer>(
        &self,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let bytes = self.to_bytes();
        serde_bytes::Bytes::new(&bytes).serialize(s)
    }
}

impl<'de> Deserialize<'de> for Signature {
    fn deserialize<D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<Self, D::Error> {
        let bb: serde_bytes::ByteBuf = serde_bytes::ByteBuf::deserialize(d)?;
        if bb.len() != 80 {
            return Err(serde::de::Error::custom(format!(
                "expected 80 signature bytes, got {}",
                bb.len()
            )));
        }
        let mut buf = [0u8; 80];
        buf.copy_from_slice(&bb);
        Signature::from_bytes(&buf).map_err(serde::de::Error::custom)
    }
}

impl Signature {
    /// Encode as `g1_compressed(48) || scalar(32)` (80 bytes).
    pub fn to_bytes(&self) -> [u8; 80] {
        let mut out = [0u8; 80];
        out[..48].copy_from_slice(&g1_to_bytes(&self.a));
        out[48..].copy_from_slice(&scalar_to_bytes(&self.e));
        out
    }

    /// Decode the 80-byte form.
    pub fn from_bytes(bytes: &[u8; 80]) -> Result<Self, BbsError> {
        let mut a_buf = [0u8; 48];
        a_buf.copy_from_slice(&bytes[..48]);
        let mut e_buf = [0u8; 32];
        e_buf.copy_from_slice(&bytes[48..]);
        Ok(Self {
            a: g1_from_bytes(&a_buf)?,
            e: scalar_from_bytes(&e_buf)?,
        })
    }
}

/// Compute the commitment
/// `B = H_0 + sum_{i=1}^{n} H_{i} * msg_{i-1}`.
///
/// Requires `generators.len() >= messages.len() + 1`. `H_0` plays the
/// role of the public anchor (`P1` in the draft).
pub(crate) fn compute_b(
    messages: &[Scalar],
    generators: &[G1Projective],
) -> Result<G1Projective, BbsError> {
    if generators.len() < messages.len() + 1 {
        return Err(BbsError::GeneratorCount {
            have: generators.len(),
            need: messages.len() + 1,
        });
    }
    let mut b = generators[0];
    for (i, m) in messages.iter().enumerate() {
        b += generators[i + 1] * m;
    }
    Ok(b)
}

/// Deterministically derive the `e` scalar from the secret key and the
/// message vector. This matches the draft's `Sign` step that selects
/// `e` via `hash_to_scalar`.
fn derive_e(secret_key: &SecretKey, messages: &[Scalar]) -> Scalar {
    let mut buf = Vec::with_capacity(32 + 32 * messages.len() + 8);
    buf.extend_from_slice(&scalar_to_bytes(secret_key.as_scalar()));
    buf.extend_from_slice(&(messages.len() as u64).to_be_bytes());
    for m in messages {
        buf.extend_from_slice(&scalar_to_bytes(m));
    }
    hash_to_scalar(DST_HASH_TO_SCALAR, &buf)
}

/// Sign a message vector with a BBS+ key.
pub fn sign(
    messages: &[Scalar],
    secret_key: &SecretKey,
    _public_key: &PublicKey,
    generators: &[G1Projective],
) -> Result<Signature, BbsError> {
    let b = compute_b(messages, generators)?;
    let e = derive_e(secret_key, messages);
    // `sk + e` must be non-zero. The probability of equality is 2^-255;
    // we still refuse gracefully.
    let denom = secret_key.as_scalar() + e;
    if bool::from(denom.ct_eq(&Scalar::ZERO)) {
        return Err(BbsError::SignatureVerification);
    }
    // `denom` is verified non-zero above, so `invert()` is always Some.
    let Some(inv) = Option::<Scalar>::from(denom.invert()) else {
        return Err(BbsError::SignatureVerification);
    };
    let a = G1Affine::from(b * inv);
    Ok(Signature { a, e })
}

/// Verify a BBS+ signature.
///
/// Returns `Ok(())` if the pairing equation
/// `e(A, W + g2*e) == e(B, g2)` holds.
pub fn verify(
    messages: &[Scalar],
    signature: &Signature,
    public_key: &PublicKey,
    generators: &[G1Projective],
) -> Result<(), BbsError> {
    let b = compute_b(messages, generators)?;

    let w_plus_e_g2 = G2Affine::from(
        G2Projective::from(public_key.w) + G2Projective::generator() * signature.e,
    );
    let g2_gen = G2Affine::generator();

    // Pairing identity: e(A, W + g2*e) * e(-B, g2) == 1
    let prep_a = G2Prepared::from(w_plus_e_g2);
    let prep_b = G2Prepared::from(g2_gen);
    let neg_b = G1Affine::from(-b);
    let gt = multi_miller_loop(&[(&signature.a, &prep_a), (&neg_b, &prep_b)])
        .final_exponentiation();
    if bool::from(gt.is_identity()) {
        Ok(())
    } else {
        Err(BbsError::SignatureVerification)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::message_to_scalar;
    use crate::generators::message_generators;
    use crate::keys::keygen;
    use rand::rngs::OsRng;

    fn fixture(n: usize) -> (Vec<Scalar>, Vec<G1Projective>) {
        let msgs: Vec<Scalar> = (0..n)
            .map(|i| message_to_scalar(format!("msg-{i}").as_bytes()))
            .collect();
        let gens = message_generators(b"smart-byte-bbs-test", n + 2);
        (msgs, gens)
    }

    #[test]
    fn sign_verify_roundtrip() {
        let kp = keygen(&mut OsRng);
        let (msgs, gens) = fixture(5);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        verify(&msgs, &sig, &kp.public, &gens).unwrap();
    }

    #[test]
    fn sign_is_deterministic() {
        let kp = keygen(&mut OsRng);
        let (msgs, gens) = fixture(3);
        let a = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        let b = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn tampered_message_fails() {
        let kp = keygen(&mut OsRng);
        let (mut msgs, gens) = fixture(4);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        msgs[2] = message_to_scalar(b"tampered");
        assert!(verify(&msgs, &sig, &kp.public, &gens).is_err());
    }

    #[test]
    fn wrong_key_fails() {
        let kp = keygen(&mut OsRng);
        let other = keygen(&mut OsRng);
        let (msgs, gens) = fixture(3);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        assert!(verify(&msgs, &sig, &other.public, &gens).is_err());
    }

    #[test]
    fn tampered_signature_fails() {
        let kp = keygen(&mut OsRng);
        let (msgs, gens) = fixture(3);
        let mut sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        sig.e += Scalar::ONE;
        assert!(verify(&msgs, &sig, &kp.public, &gens).is_err());
    }

    #[test]
    fn generator_shortfall_errors() {
        let kp = keygen(&mut OsRng);
        let (msgs, _) = fixture(5);
        let short = message_generators(b"x", 3); // Need at least 6.
        assert!(sign(&msgs, &kp.secret, &kp.public, &short).is_err());
    }

    #[test]
    fn signature_serde_roundtrip() {
        let kp = keygen(&mut OsRng);
        let (msgs, gens) = fixture(2);
        let sig = sign(&msgs, &kp.secret, &kp.public, &gens).unwrap();
        let bytes = sig.to_bytes();
        let back = Signature::from_bytes(&bytes).unwrap();
        assert_eq!(back, sig);
    }
}
