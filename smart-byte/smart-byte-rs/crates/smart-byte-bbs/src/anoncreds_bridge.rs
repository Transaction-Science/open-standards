//! Hyperledger AnonCreds 2.0 interop helpers.
//!
//! AnonCreds 2.0 (Hyperledger, 2024+) standardised on BBS+ over
//! BLS12-381 as the underlying signature scheme. This module exposes
//! the AnonCreds-specific glue that does not belong in the generic
//! [`crate::cryptosuite`] path:
//!
//! * **Revocation handles.** AnonCreds maintains a revocation registry
//!   independent of the BBS+ signature; the credential carries an
//!   opaque `revocation_handle` (a 32-byte identifier) that the
//!   verifier dereferences against the registry. Smart Byte's
//!   revocation primitives live in a separate crate (planned), so
//!   here we only model the handle type.
//! * **Link secrets.** A holder-specific scalar that participates in
//!   the credential's message vector at position 0 (by convention) so
//!   different credentials issued to the same holder can be proven
//!   to belong to the same person without revealing the holder's
//!   identifier.
//! * **Predicate proofs.** AnonCreds supports range predicates of the
//!   form `attr >= threshold` or `attr < threshold` over numeric
//!   attributes. Full predicate-proof support requires a separate
//!   range-proof scheme (Bulletproofs, set membership, etc.). This
//!   module exposes a [`PredicateClaim`] descriptor and a
//!   [`PredicateProof`] envelope so downstream code can plug in the
//!   range-proof system of their choice without re-encoding the
//!   AnonCreds shape.
//!
//! The compatibility tag [`ANONCREDS_2_0_TAG`] records the version of
//! AnonCreds these helpers track. Bump it when AnonCreds publishes a
//! breaking change.

use bls12_381::Scalar;
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use crate::encode::message_to_scalar;

/// Compatibility tag for the AnonCreds version these helpers mirror.
pub const ANONCREDS_2_0_TAG: &str = "hyperledger-anoncreds-2.0";

/// An opaque 32-byte revocation handle bound to an issuance.
///
/// Smart Byte's substrate-wide revocation system will dereference
/// these to a status bit; this type is only the handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevocationHandle(#[serde(with = "serde_bytes_array_32")] pub [u8; 32]);

impl RevocationHandle {
    /// Construct from raw bytes.
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// A holder's link secret. Drawn once per holder identity and reused
/// across all credentials issued to them. Zeroes on drop.
#[derive(Clone)]
pub struct LinkSecret(Scalar);

impl LinkSecret {
    /// Draw a fresh link secret.
    pub fn new<R: RngCore + CryptoRng>(rng: &mut R) -> Self {
        let mut wide = [0u8; 64];
        rng.fill_bytes(&mut wide);
        Self(Scalar::from_bytes_wide(&wide))
    }

    /// Borrow the inner scalar.
    pub fn as_scalar(&self) -> &Scalar {
        &self.0
    }

    /// Convert from arbitrary bytes via the BBS+ message-to-scalar
    /// reduction. Useful for deriving a link secret from a wallet
    /// passphrase.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self(message_to_scalar(bytes))
    }
}

impl Drop for LinkSecret {
    fn drop(&mut self) {
        let mut bytes = self.0.to_bytes();
        bytes.zeroize();
        self.0 = Scalar::from(0u64);
    }
}

/// Descriptor of a predicate claim. `attr_index` is the position in
/// the message vector of the attribute being proven; `predicate` is
/// the relation; `threshold` is the comparand. Both attribute and
/// threshold are interpreted as little-endian u64s for the purposes
/// of range-proof systems.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateClaim {
    /// Position of the attribute in the BBS+ message vector.
    pub attr_index: usize,
    /// Relation kind.
    pub predicate: Predicate,
    /// Numeric threshold.
    pub threshold: u64,
}

/// Supported AnonCreds predicate relations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Predicate {
    /// `attr >= threshold`.
    GreaterOrEqual,
    /// `attr > threshold`.
    Greater,
    /// `attr <= threshold`.
    LessOrEqual,
    /// `attr < threshold`.
    Less,
    /// `attr == threshold`.
    Equal,
}

/// Envelope for a predicate proof. The `body` is the
/// implementation-specific range-proof artefact (Bulletproofs,
/// CL-Predicate, set-membership accumulator, ...). Smart Byte does
/// not commit to a single range-proof scheme yet, so the body is
/// opaque bytes plus a tag identifying the scheme.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateProof {
    /// Which predicate is being proven.
    pub claim: PredicateClaim,
    /// Scheme tag, e.g. `"bulletproofs-v1"` or `"cl-predicate"`.
    pub scheme: String,
    /// Opaque scheme-specific proof bytes.
    #[serde(with = "serde_bytes")]
    pub body: Vec<u8>,
}

impl PredicateProof {
    /// Construct a predicate-proof envelope.
    pub fn new(claim: PredicateClaim, scheme: impl Into<String>, body: Vec<u8>) -> Self {
        Self {
            claim,
            scheme: scheme.into(),
            body,
        }
    }
}

mod serde_bytes_array_32 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        bytes: &[u8; 32],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        serde_bytes::Bytes::new(bytes).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<[u8; 32], D::Error> {
        let bb: serde_bytes::ByteBuf = serde_bytes::ByteBuf::deserialize(d)?;
        if bb.len() != 32 {
            return Err(serde::de::Error::custom(format!(
                "expected 32 bytes, got {}",
                bb.len()
            )));
        }
        let mut out = [0u8; 32];
        out.copy_from_slice(&bb);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn link_secret_drawn_random() {
        let a = LinkSecret::new(&mut OsRng);
        let b = LinkSecret::new(&mut OsRng);
        assert_ne!(a.as_scalar().to_bytes(), b.as_scalar().to_bytes());
    }

    #[test]
    fn link_secret_from_passphrase_is_deterministic() {
        let a = LinkSecret::from_bytes(b"my-wallet-passphrase");
        let b = LinkSecret::from_bytes(b"my-wallet-passphrase");
        assert_eq!(a.as_scalar().to_bytes(), b.as_scalar().to_bytes());
    }

    #[test]
    fn revocation_handle_serde() {
        let h = RevocationHandle::new([7u8; 32]);
        let bytes = serde_cbor::to_vec(&h).unwrap();
        let back: RevocationHandle = serde_cbor::from_slice(&bytes).unwrap();
        assert_eq!(back, h);
    }

    #[test]
    fn predicate_proof_envelope_serde() {
        let claim = PredicateClaim {
            attr_index: 2,
            predicate: Predicate::GreaterOrEqual,
            threshold: 18,
        };
        let pp = PredicateProof::new(claim, "bulletproofs-v1", vec![1, 2, 3]);
        let bytes = serde_cbor::to_vec(&pp).unwrap();
        let back: PredicateProof = serde_cbor::from_slice(&bytes).unwrap();
        assert_eq!(back, pp);
        assert_eq!(back.scheme, "bulletproofs-v1");
    }

    #[test]
    fn anoncreds_tag_is_2_0() {
        assert_eq!(ANONCREDS_2_0_TAG, "hyperledger-anoncreds-2.0");
    }
}
