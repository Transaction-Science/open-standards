//! Git-style ownership chain.
//!
//! Every transfer of an envelope is a [`Transition`] whose `prior_hash`
//! points at the BLAKE3 of the previous transition (CBOR-encoded). The
//! genesis transition has `prior_hash = None`. Replaying the chain
//! reproduces the current owner; tampering with any transition breaks
//! the hash linkage.

use serde::{Deserialize, Serialize};

use crate::Blake3Hash;
use crate::said::Said;

/// Wire-format Ed25519 signature: 64 bytes. We hold the raw bytes here
/// (rather than `ed25519_dalek::Signature`) so that transitions can be
/// CBOR-round-tripped without depending on the dalek crate's
/// Serialize impl quirks.
pub type SignatureBytes = [u8; 64];

/// One link in the ownership chain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transition {
    /// Outgoing owner SAID.
    pub from: Said,
    /// Incoming owner SAID.
    pub to: Said,
    /// Signature by `from` over the canonical CBOR of this transition
    /// with the `signature` field zeroed.
    #[serde(with = "serde_bytes_array")]
    pub signature: SignatureBytes,
    /// BLAKE3 hash of the previous transition's CBOR; `None` only for
    /// the genesis transition.
    pub prior_hash: Option<Blake3Hash>,
}

impl Transition {
    /// Construct a transition whose `signature` field is the all-zero
    /// placeholder. Callers fill in the signature after computing it
    /// over this canonical form.
    pub fn unsigned(from: Said, to: Said, prior_hash: Option<Blake3Hash>) -> Self {
        Self {
            from,
            to,
            signature: [0u8; 64],
            prior_hash,
        }
    }

    /// BLAKE3 of this transition's canonical CBOR.
    pub fn content_hash(&self) -> Blake3Hash {
        let bytes = serde_cbor::to_vec(self).expect("Transition CBOR is infallible");
        *blake3::hash(&bytes).as_bytes()
    }
}

/// Append-only chain of [`Transition`]s.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnershipChain {
    pub transitions: Vec<Transition>,
}

impl OwnershipChain {
    /// Construct an empty chain (no owners yet).
    pub fn empty() -> Self {
        Self::default()
    }

    /// SAID of the current owner, if any.
    pub fn current_owner(&self) -> Option<&Said> {
        self.transitions.last().map(|t| &t.to)
    }

    /// Append a transition. Returns an error if `prior_hash` does not
    /// match the previous transition's content hash.
    pub fn push(&mut self, t: Transition) -> Result<(), OwnershipError> {
        let expected_prior = self.transitions.last().map(|p| p.content_hash());
        if t.prior_hash != expected_prior {
            return Err(OwnershipError::BrokenLink);
        }
        self.transitions.push(t);
        Ok(())
    }
}

/// Ownership-chain errors.
#[derive(Debug, thiserror::Error)]
pub enum OwnershipError {
    #[error("prior_hash does not match preceding transition")]
    BrokenLink,
}

/// Custom serde for fixed-length byte arrays. CBOR encodes a `[u8; 64]`
/// as an array-of-u8 by default, which is correct but inefficient.
/// Using `serde_bytes` produces a `bytes` major type instead.
mod serde_bytes_array {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        bytes: &[u8; 64],
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        serde_bytes::Bytes::new(bytes).serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<[u8; 64], D::Error> {
        let v: serde_bytes::ByteBuf = serde_bytes::ByteBuf::deserialize(de)?;
        let b = v.into_vec();
        if b.len() != 64 {
            return Err(serde::de::Error::custom("expected 64 bytes for signature"));
        }
        let mut out = [0u8; 64];
        out.copy_from_slice(&b);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn principal(seed: u8) -> Said {
        Said::hash(&[seed; 8])
    }

    #[test]
    fn chain_links_through_prior_hash() {
        let mut chain = OwnershipChain::empty();
        let alice = principal(1);
        let bob = principal(2);
        let carol = principal(3);

        let t0 = Transition::unsigned(alice, bob, None);
        chain.push(t0.clone()).unwrap();
        let t1 = Transition::unsigned(bob, carol, Some(t0.content_hash()));
        chain.push(t1).unwrap();

        assert_eq!(chain.current_owner(), Some(&carol));
    }

    #[test]
    fn broken_link_rejected() {
        let mut chain = OwnershipChain::empty();
        let alice = principal(1);
        let bob = principal(2);
        let carol = principal(3);

        chain
            .push(Transition::unsigned(alice, bob, None))
            .unwrap();
        let bogus = Transition::unsigned(bob, carol, Some([7u8; 32]));
        assert!(matches!(chain.push(bogus), Err(OwnershipError::BrokenLink)));
    }
}
