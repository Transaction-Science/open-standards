//! Anonymous-credential sketch with BBS-like structure.
//!
//! This module is intentionally a **sketch** — a credential-shape
//! envelope plus a holder-side link-secret commitment, enough to wire
//! into [`crate::presentation::VerifiablePresentation`] without
//! standing up a full BBS+ stack here. (The real, audited BBS+
//! signatures live in the sibling `smart-byte-bbs` crate.)
//!
//! What this module *does* provide:
//!
//! * [`LinkSecret`] — a per-holder 32-byte secret with a stable
//!   Pedersen commitment, used to bind multiple presentations to the
//!   same holder without revealing the holder identifier.
//! * [`AnonCredAttribute`] — an attribute slot, either disclosed
//!   (cleartext) or hidden (carrying a Pedersen commitment).
//! * [`AnonCred`] — a credential envelope of attributes plus an
//!   opaque signature blob (deferred to a downstream BBS+ /
//!   AnonCreds-v2 backend).
//! * [`AnonCredPresentation`] — a presentation envelope with hidden
//!   attributes replaced by predicate-proof references and disclosed
//!   attributes echoed in cleartext.

use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::ZkError;

/// Holder-bound 32-byte secret. Two presentations from the same
/// holder commit to the same scalar but use independent blinding, so
/// the wire commitments are unlinkable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkSecret(pub Scalar);

impl LinkSecret {
    /// Sample a fresh link secret from `OsRng`.
    pub fn random() -> Self {
        Self(Scalar::random(&mut OsRng))
    }

    /// Derive a link secret deterministically from a seed (e.g. a
    /// hash of a holder DID + nonce). Useful for backup / recovery.
    pub fn from_seed(seed: &[u8]) -> Self {
        let mut h = Sha256::new();
        h.update(b"smart-byte-zk/anoncreds/link-secret");
        h.update(seed);
        let bytes = h.finalize();
        let mut wide = [0u8; 64];
        wide[..32].copy_from_slice(&bytes);
        Self(Scalar::from_bytes_mod_order_wide(&wide))
    }

    /// Pedersen-commit to this link secret under a fresh blinding,
    /// returning `(commitment, blinding)`.
    pub fn commit(&self) -> (CompressedRistretto, Scalar) {
        let pc = ::bulletproofs::PedersenGens::default();
        let blinding = Scalar::random(&mut OsRng);
        let c: RistrettoPoint = pc.commit(self.0, blinding);
        (c.compress(), blinding)
    }
}

/// Per-credential attribute slot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AnonCredAttribute {
    /// A disclosed attribute, in cleartext.
    Disclosed {
        /// Attribute name.
        name: String,
        /// Attribute value, encoded as UTF-8.
        value: String,
    },
    /// A hidden attribute, represented by a Pedersen commitment.
    Hidden {
        /// Attribute name.
        name: String,
        /// Compressed Pedersen commitment.
        #[serde(with = "serde_bytes")]
        commitment: [u8; 32],
    },
}

/// AnonCreds-style credential envelope.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnonCred {
    /// Credential identifier (e.g. issuer-scoped serial).
    pub cred_id: String,
    /// Issuer identifier (e.g. DID URL).
    pub issuer: String,
    /// Attribute slots.
    pub attributes: Vec<AnonCredAttribute>,
    /// Opaque signature blob — to be filled in by a downstream BBS+ /
    /// AnonCreds-v2 backend. The shape of these bytes is deliberately
    /// left to that backend.
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
    /// Pedersen commitment to the holder's [`LinkSecret`].
    #[serde(with = "serde_bytes")]
    pub link_commitment: [u8; 32],
}

/// Holder-side presentation envelope. Hidden attributes carry a
/// per-attribute predicate-proof reference (an opaque byte blob
/// indexed by the predicate scheme), disclosed attributes are echoed
/// in cleartext, and the link commitment is fresh.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnonCredPresentation {
    /// Credential identifier the presentation derives from.
    pub cred_id: String,
    /// Issuer identifier the credential is bound to.
    pub issuer: String,
    /// Disclosed attributes echoed in cleartext.
    pub disclosed: Vec<(String, String)>,
    /// Predicate-proof references for hidden attributes.
    pub predicates: Vec<PredicateRef>,
    /// Fresh Pedersen commitment to the holder's link secret. Two
    /// presentations from the same holder will produce *different*
    /// wire bytes here, but a verifier with the holder's public
    /// link-commitment template can correlate them at audit time.
    #[serde(with = "serde_bytes")]
    pub link_commitment: [u8; 32],
}

/// Reference to a predicate proof carried in
/// [`crate::presentation::VerifiablePresentation`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredicateRef {
    /// Attribute name the predicate is over.
    pub attribute: String,
    /// Predicate kind tag: `"range"`, `"inequality"`,
    /// `"set-membership"`.
    pub kind: String,
    /// Index into the presentation's `proofs` vector.
    pub proof_index: usize,
}

/// Build a fresh presentation envelope from an [`AnonCred`].
pub fn present(cred: &AnonCred, link: &LinkSecret) -> Result<AnonCredPresentation, ZkError> {
    let mut disclosed = Vec::new();
    for attr in &cred.attributes {
        if let AnonCredAttribute::Disclosed { name, value } = attr {
            disclosed.push((name.clone(), value.clone()));
        }
    }
    let (link_commit, _blinding) = link.commit();
    Ok(AnonCredPresentation {
        cred_id: cred.cred_id.clone(),
        issuer: cred.issuer.clone(),
        disclosed,
        predicates: Vec::new(),
        link_commitment: link_commit.to_bytes(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_secret_deterministic_from_seed() {
        let a = LinkSecret::from_seed(b"did:example:alice");
        let b = LinkSecret::from_seed(b"did:example:alice");
        assert_eq!(a, b);
        let c = LinkSecret::from_seed(b"did:example:bob");
        assert_ne!(a, c);
    }

    #[test]
    fn present_echoes_disclosed_attributes() {
        let link = LinkSecret::random();
        let (commit, _) = link.commit();
        let cred = AnonCred {
            cred_id: "cred-1".into(),
            issuer: "did:example:issuer".into(),
            attributes: vec![
                AnonCredAttribute::Disclosed {
                    name: "given_name".into(),
                    value: "Alice".into(),
                },
                AnonCredAttribute::Hidden {
                    name: "age".into(),
                    commitment: [7u8; 32],
                },
            ],
            signature: vec![0u8; 8],
            link_commitment: commit.to_bytes(),
        };
        let pres = present(&cred, &link).expect("present");
        assert_eq!(pres.disclosed.len(), 1);
        assert_eq!(pres.disclosed[0].0, "given_name");
    }
}
