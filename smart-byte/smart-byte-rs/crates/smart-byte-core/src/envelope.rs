//! The Envelope is the substrate's load-bearing primitive.
//!
//! Wire layout (CBOR major-type 5, map):
//!
//! ```text
//! {
//!   "id":          Said (32 bytes),
//!   "provenance":  Provenance,
//!   "ownership":   OwnershipChain,
//!   "cargo":       Cargo,
//!   "joule_cost":  JouleCost,
//! }
//! ```
//!
//! The SAID is computed by:
//!   1. Building the envelope with `id = Said([0; 32])`.
//!   2. Canonical-CBOR-encoding the envelope.
//!   3. BLAKE3-hashing the bytes.
//!
//! After construction the SAID is stamped into `id`. Verifiers
//! recompute the SAID and compare.

use serde::{Deserialize, Serialize};

use crate::cargo::Cargo;
use crate::joule_cost::JouleCost;
use crate::ownership::OwnershipChain;
use crate::provenance::Provenance;
use crate::said::Said;

/// Errors produced when building or validating an envelope.
#[derive(Debug, thiserror::Error)]
pub enum EnvelopeError {
    #[error("CBOR encode/decode failed: {0}")]
    Cbor(String),
    #[error("envelope SAID mismatch (expected {expected}, computed {computed})")]
    SaidMismatch { expected: Said, computed: Said },
}

impl From<serde_cbor::Error> for EnvelopeError {
    fn from(e: serde_cbor::Error) -> Self {
        EnvelopeError::Cbor(e.to_string())
    }
}

/// A signed, content-addressed envelope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Envelope {
    /// Self-Addressing IDentifier: BLAKE3 over canonical CBOR with this
    /// field zeroed.
    pub id: Said,
    pub provenance: Provenance,
    pub ownership: OwnershipChain,
    pub cargo: Cargo,
    pub joule_cost: JouleCost,
}

impl Envelope {
    /// Construct an envelope and stamp its SAID.
    pub fn new(
        provenance: Provenance,
        ownership: OwnershipChain,
        cargo: Cargo,
        joule_cost: JouleCost,
    ) -> Result<Self, EnvelopeError> {
        let mut env = Self {
            id: Said::default(),
            provenance,
            ownership,
            cargo,
            joule_cost,
        };
        env.id = env.compute_said()?;
        Ok(env)
    }

    /// Compute (but do not stamp) what this envelope's SAID should be.
    /// The computation zeroes `id` before hashing so it is deterministic
    /// regardless of the current `id` value.
    pub fn compute_said(&self) -> Result<Said, EnvelopeError> {
        let mut tmp = self.clone();
        tmp.id = Said::default();
        let bytes = serde_cbor::to_vec(&tmp)?;
        Ok(Said::hash(&bytes))
    }

    /// Encode this envelope as canonical CBOR.
    pub fn to_cbor(&self) -> Result<Vec<u8>, EnvelopeError> {
        Ok(serde_cbor::to_vec(self)?)
    }

    /// Decode from canonical CBOR.
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, EnvelopeError> {
        let env: Self = serde_cbor::from_slice(bytes)?;
        Ok(env)
    }

    /// Verify that the stamped SAID matches a freshly-computed SAID.
    pub fn verify_said(&self) -> Result<(), EnvelopeError> {
        let computed = self.compute_said()?;
        if computed == self.id {
            Ok(())
        } else {
            Err(EnvelopeError::SaidMismatch {
                expected: self.id,
                computed,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ownership::{OwnershipChain, Transition};
    use chrono::TimeZone;
    use ed25519_dalek::{SigningKey, ed25519::signature::Signer};
    use rand::rngs::OsRng;

    fn fixture_envelope() -> Envelope {
        let issuer = Said::hash(b"issuer");
        let alice = Said::hash(b"alice");
        let bob = Said::hash(b"bob");
        let issued_at = chrono::Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap();
        let prov = Provenance::new(issuer, issued_at, b"auth".to_vec());

        let mut chain = OwnershipChain::empty();
        chain.push(Transition::unsigned(issuer, alice, None)).unwrap();
        let prior = chain.transitions.last().unwrap().content_hash();
        chain
            .push(Transition::unsigned(alice, bob, Some(prior)))
            .unwrap();

        Envelope::new(
            prov,
            chain,
            Cargo::Usd { minor: 10_000 },
            JouleCost::measured(42),
        )
        .unwrap()
    }

    #[test]
    fn said_is_stable() {
        let a = fixture_envelope();
        let b = fixture_envelope();
        assert_eq!(a.id, b.id);
    }

    #[test]
    fn cbor_roundtrip_preserves_said() {
        let env = fixture_envelope();
        let bytes = env.to_cbor().unwrap();
        let back = Envelope::from_cbor(&bytes).unwrap();
        assert_eq!(env, back);
        back.verify_said().unwrap();
    }

    #[test]
    fn mutation_breaks_said() {
        let mut env = fixture_envelope();
        env.cargo = Cargo::Usd { minor: 99_999 };
        let err = env.verify_said().unwrap_err();
        assert!(matches!(err, EnvelopeError::SaidMismatch { .. }));
    }

    #[test]
    fn sign_and_verify_happy_path() {
        let env = fixture_envelope();
        let key = SigningKey::generate(&mut OsRng);
        let sig = key.sign(env.id.as_bytes());
        crate::sign::verify(&env, &sig, &key.verifying_key()).unwrap();
    }

    #[test]
    fn sign_and_verify_rejects_wrong_key() {
        let env = fixture_envelope();
        let key = SigningKey::generate(&mut OsRng);
        let other = SigningKey::generate(&mut OsRng);
        let sig = key.sign(env.id.as_bytes());
        assert!(crate::sign::verify(&env, &sig, &other.verifying_key()).is_err());
    }

    #[test]
    fn cargo_kinds_round_trip() {
        for cargo in [
            Cargo::Bytes(vec![1, 2, 3]),
            Cargo::Usd { minor: 7 },
            Cargo::JouleClaim { microjoules: 7 },
            Cargo::Custom {
                type_uri: "urn:x:y".into(),
                body: vec![9],
            },
        ] {
            let prov = Provenance::new(
                Said::hash(b"i"),
                chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
                vec![],
            );
            let env = Envelope::new(
                prov,
                OwnershipChain::empty(),
                cargo.clone(),
                JouleCost::default(),
            )
            .unwrap();
            let bytes = env.to_cbor().unwrap();
            let back = Envelope::from_cbor(&bytes).unwrap();
            assert_eq!(back.cargo.kind(), cargo.kind());
        }
    }
}
