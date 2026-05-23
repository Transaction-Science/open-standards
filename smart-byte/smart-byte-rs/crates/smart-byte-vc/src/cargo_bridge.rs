//! Smart Byte cargo bridge for Verifiable Credentials.
//!
//! Rather than modify the closed [`smart_byte_core::Cargo`] enum, this
//! module recognises a VC as a `Cargo::Custom { type_uri, body }`
//! payload where:
//!
//! * `type_uri` is the constant [`VC_CARGO_TYPE_URI`].
//! * `body` is the canonical JCS encoding (RFC 8785) of the
//!   credential. JCS is used so the envelope's SAID — which is
//!   BLAKE3 over the canonical CBOR of the envelope, with `body` an
//!   opaque byte string — is stable across any JSON-shape-preserving
//!   round-trip of the credential.
//!
//! Signing reuses [`smart_byte_core::sign`], i.e. Ed25519 over the
//! envelope's SAID. The W3C Data Integrity proof on the credential is
//! orthogonal and can travel inside the cargo body.

use smart_byte_core::{
    Cargo, Envelope, JouleCost, OwnershipChain, Provenance, Said,
};

use crate::credential::VerifiableCredential;
use crate::error::VcError;

/// Cargo `type_uri` constant for a Verifiable Credential payload.
pub const VC_CARGO_TYPE_URI: &str = "https://www.w3.org/ns/credentials/v2#VerifiableCredential";

/// Pack a credential into a Smart Byte envelope.
///
/// The envelope's SAID is computed by `smart-byte-core` over the
/// canonical CBOR of `{provenance, ownership, Cargo::Custom{type_uri,
/// body=JCS(vc)}, joule_cost}`, with `id` zeroed.
pub fn vc_envelope(
    vc: &VerifiableCredential,
    provenance: Provenance,
    ownership: OwnershipChain,
    joule_cost: JouleCost,
) -> Result<Envelope, VcError> {
    let body = vc.to_jcs()?;
    let cargo = Cargo::Custom {
        type_uri: VC_CARGO_TYPE_URI.to_string(),
        body,
    };
    Envelope::new(provenance, ownership, cargo, joule_cost)
        .map_err(|e| VcError::Bridge(e.to_string()))
}

/// Extract a credential from an envelope whose cargo is the VC type.
///
/// Errors if the cargo is not a `Custom` payload tagged with
/// [`VC_CARGO_TYPE_URI`] or if the body fails to deserialize as a
/// `VerifiableCredential`.
pub fn vc_from_envelope(
    envelope: &Envelope,
) -> Result<VerifiableCredential, VcError> {
    match &envelope.cargo {
        Cargo::Custom { type_uri, body } if type_uri == VC_CARGO_TYPE_URI => {
            let vc: VerifiableCredential = serde_json::from_slice(body)?;
            Ok(vc)
        }
        Cargo::Custom { type_uri, .. } => Err(VcError::Bridge(format!(
            "unexpected cargo type_uri: {type_uri}"
        ))),
        other => Err(VcError::Bridge(format!(
            "cargo kind {} is not a VC payload",
            other.kind()
        ))),
    }
}

/// Compute the SAID an envelope would have if a VC were stamped today
/// with `provenance`, `ownership`, and `joule_cost`. Useful for clients
/// that want to address a credential before actually emitting the
/// envelope.
pub fn vc_said(
    vc: &VerifiableCredential,
    provenance: Provenance,
    ownership: OwnershipChain,
    joule_cost: JouleCost,
) -> Result<Said, VcError> {
    let envelope = vc_envelope(vc, provenance, ownership, joule_cost)?;
    Ok(envelope.id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credential::{CredentialSubject, VcBuilder};
    use crate::issuer::Issuer;
    use chrono::TimeZone;
    use smart_byte_core::Said;

    fn fixture_vc() -> VerifiableCredential {
        let subj = CredentialSubject {
            id: Some("did:example:alice".parse().unwrap()),
            claims: serde_json::Map::new(),
        };
        VcBuilder::new()
            .issuer(Issuer::Uri("did:example:issuer".parse().unwrap()))
            .subject(subj)
            .valid_from(chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap())
            .build()
            .unwrap()
    }

    fn fixture_provenance() -> Provenance {
        Provenance::new(
            Said::hash(b"vc-issuer"),
            chrono::Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap(),
            b"vc-auth".to_vec(),
        )
    }

    #[test]
    fn envelope_roundtrip_preserves_vc() {
        let vc = fixture_vc();
        let env = vc_envelope(
            &vc,
            fixture_provenance(),
            OwnershipChain::empty(),
            JouleCost::measured(7),
        )
        .unwrap();
        env.verify_said().unwrap();
        let back = vc_from_envelope(&env).unwrap();
        assert_eq!(back, vc);
    }

    #[test]
    fn said_is_stable_across_calls() {
        let vc = fixture_vc();
        let a = vc_said(
            &vc,
            fixture_provenance(),
            OwnershipChain::empty(),
            JouleCost::measured(7),
        )
        .unwrap();
        let b = vc_said(
            &vc,
            fixture_provenance(),
            OwnershipChain::empty(),
            JouleCost::measured(7),
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_wrong_cargo_kind() {
        let env = Envelope::new(
            fixture_provenance(),
            OwnershipChain::empty(),
            Cargo::Bytes(vec![1, 2, 3]),
            JouleCost::default(),
        )
        .unwrap();
        assert!(vc_from_envelope(&env).is_err());
    }
}
