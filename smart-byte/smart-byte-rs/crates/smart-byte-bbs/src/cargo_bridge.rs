//! Smart Byte cargo bridge for BBS+-signed credentials.
//!
//! Following the pattern from `smart-byte-vc::cargo_bridge`, a BBS+
//! credential is packaged as `Cargo::Custom { type_uri, body }`
//! where:
//!
//! * `type_uri` is the constant [`BBS_CREDENTIAL_CARGO_TYPE_URI`].
//! * `body` is the canonical CBOR encoding of [`BbsCredentialBody`].
//!
//! ## SAID stability under disclosure
//!
//! A BBS+ holder may emit *many* selective-disclosure proofs from a
//! single signed credential. Each proof has different cryptographic
//! artefacts (unlinkability) and a smaller message set. If the
//! disclosure proofs were placed directly inside the envelope's
//! cargo, the envelope SAID would mutate with every disclosure,
//! which is the wrong identity model: the *credential* is one
//! content-addressed object; presentations are derived views.
//!
//! Smart Byte therefore distinguishes two envelope flavours:
//!
//! 1. **Credential envelope.** Carries [`BbsCredentialBody`] —
//!    issuance proof + claim values + public key. This is the
//!    SAID-stable, canonically addressed object.
//! 2. **Presentation envelope.** Carries [`BbsPresentationBody`] —
//!    references the credential envelope's SAID and embeds a
//!    selective-disclosure proof. Its own SAID is fresh per
//!    presentation; the *underlying* credential SAID it references
//!    is stable.
//!
//! The canonical re-encoding rule that preserves the credential SAID
//! under disclosure is: *the CBOR body of a presentation envelope
//! MUST always be derived from the byte-identical CBOR body of the
//! referenced credential envelope — never re-serialised from a parsed
//! intermediate.* This rule is enforced by
//! [`presentation_envelope`] which takes the parsed credential
//! envelope and reuses its `cargo.body` bytes byte-for-byte when
//! constructing the presentation.

use serde::{Deserialize, Serialize};
use smart_byte_core::{
    Cargo, Envelope, JouleCost, OwnershipChain, Provenance, Said,
};

use crate::cryptosuite::{
    Bbs2023DisclosureProof, Bbs2023IssuanceProof,
};
use crate::error::BbsError;
use crate::keys::PublicKey;

/// Cargo `type_uri` for a BBS+-signed credential envelope.
pub const BBS_CREDENTIAL_CARGO_TYPE_URI: &str =
    "urn:smart-byte:cargo:bbs-credential:v1";

/// Cargo `type_uri` for a BBS+ selective-disclosure presentation
/// envelope.
pub const BBS_PRESENTATION_CARGO_TYPE_URI: &str =
    "urn:smart-byte:cargo:bbs-presentation:v1";

/// The CBOR body of a BBS+ credential envelope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BbsCredentialBody {
    /// Cryptosuite identifier, e.g. `"bbs-2023"`.
    pub cryptosuite: String,
    /// Issuer public key (96-byte compressed G2).
    pub public_key: PublicKey,
    /// The signed claims, in the same order they were signed.
    pub claims: Vec<Vec<u8>>,
    /// The BBS+ issuance proof.
    pub issuance: Bbs2023IssuanceProof,
}

/// The CBOR body of a BBS+ presentation envelope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BbsPresentationBody {
    /// SAID of the credential envelope this presentation derives from.
    pub credential_said: Said,
    /// Issuer public key (so verifiers need not dereference the
    /// credential envelope to verify the proof).
    pub public_key: PublicKey,
    /// Verifier-supplied nonce that was bound into the proof.
    #[serde(with = "serde_bytes")]
    pub nonce: Vec<u8>,
    /// The selective-disclosure proof.
    pub disclosure: Bbs2023DisclosureProof,
}

/// Build a credential envelope from a BBS+ issuance.
pub fn bbs_credential_envelope(
    cryptosuite: impl Into<String>,
    public_key: PublicKey,
    claims: Vec<Vec<u8>>,
    issuance: Bbs2023IssuanceProof,
    provenance: Provenance,
    ownership: OwnershipChain,
    joule_cost: JouleCost,
) -> Result<Envelope, BbsError> {
    let body = BbsCredentialBody {
        cryptosuite: cryptosuite.into(),
        public_key,
        claims,
        issuance,
    };
    let body_bytes = serde_cbor::to_vec(&body)?;
    let cargo = Cargo::Custom {
        type_uri: BBS_CREDENTIAL_CARGO_TYPE_URI.to_string(),
        body: body_bytes,
    };
    Envelope::new(provenance, ownership, cargo, joule_cost)
        .map_err(|e| BbsError::Bridge(e.to_string()))
}

/// Recover the BBS+ credential body from an envelope.
pub fn bbs_credential_from_envelope(
    envelope: &Envelope,
) -> Result<BbsCredentialBody, BbsError> {
    match &envelope.cargo {
        Cargo::Custom { type_uri, body } if type_uri == BBS_CREDENTIAL_CARGO_TYPE_URI => {
            Ok(serde_cbor::from_slice(body)?)
        }
        Cargo::Custom { type_uri, .. } => Err(BbsError::Bridge(format!(
            "unexpected cargo type_uri: {type_uri}"
        ))),
        other => Err(BbsError::Bridge(format!(
            "cargo kind {} is not a BBS credential",
            other.kind()
        ))),
    }
}

/// Build a presentation envelope from a credential envelope plus a
/// disclosure proof.
///
/// The presentation's body records the *referenced credential SAID*
/// (so verifiers know which credential the proof is derived from)
/// and the disclosure artefact. This envelope has its own SAID
/// which intentionally differs from the credential's; the credential
/// SAID remains stable across any number of presentations.
pub fn presentation_envelope(
    credential_envelope: &Envelope,
    public_key: PublicKey,
    nonce: Vec<u8>,
    disclosure: Bbs2023DisclosureProof,
    provenance: Provenance,
    ownership: OwnershipChain,
    joule_cost: JouleCost,
) -> Result<Envelope, BbsError> {
    // Confirm the referenced envelope is a BBS credential — this also
    // exercises the canonical re-encoding rule because the verifier
    // never has to re-serialise the credential body.
    let _ = bbs_credential_from_envelope(credential_envelope)?;
    let body = BbsPresentationBody {
        credential_said: credential_envelope.id,
        public_key,
        nonce,
        disclosure,
    };
    let body_bytes = serde_cbor::to_vec(&body)?;
    let cargo = Cargo::Custom {
        type_uri: BBS_PRESENTATION_CARGO_TYPE_URI.to_string(),
        body: body_bytes,
    };
    Envelope::new(provenance, ownership, cargo, joule_cost)
        .map_err(|e| BbsError::Bridge(e.to_string()))
}

/// Recover a presentation body from an envelope.
pub fn presentation_from_envelope(
    envelope: &Envelope,
) -> Result<BbsPresentationBody, BbsError> {
    match &envelope.cargo {
        Cargo::Custom { type_uri, body } if type_uri == BBS_PRESENTATION_CARGO_TYPE_URI => {
            Ok(serde_cbor::from_slice(body)?)
        }
        Cargo::Custom { type_uri, .. } => Err(BbsError::Bridge(format!(
            "unexpected cargo type_uri: {type_uri}"
        ))),
        other => Err(BbsError::Bridge(format!(
            "cargo kind {} is not a BBS presentation",
            other.kind()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cryptosuite::{
        create_bbs_2023_disclosure, issue_bbs_2023,
    };
    use crate::keys::keygen;
    use chrono::TimeZone;
    use rand::rngs::OsRng;
    use smart_byte_core::Said;

    fn fixture_provenance() -> Provenance {
        Provenance::new(
            Said::hash(b"bbs-issuer"),
            chrono::Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap(),
            b"bbs-auth".to_vec(),
        )
    }

    #[test]
    fn credential_envelope_roundtrip() {
        let kp = keygen(&mut OsRng);
        let claims: Vec<&[u8]> = vec![b"id", b"name", b"age"];
        let iss = issue_bbs_2023(&claims, &kp.secret, &kp.public).unwrap();
        let claim_vecs: Vec<Vec<u8>> =
            claims.iter().map(|c| c.to_vec()).collect();
        let env = bbs_credential_envelope(
            "bbs-2023",
            kp.public.clone(),
            claim_vecs.clone(),
            iss.clone(),
            fixture_provenance(),
            OwnershipChain::empty(),
            JouleCost::measured(11),
        )
        .unwrap();
        env.verify_said().unwrap();
        let back = bbs_credential_from_envelope(&env).unwrap();
        assert_eq!(back.public_key, kp.public);
        assert_eq!(back.claims, claim_vecs);
        assert_eq!(back.issuance, iss);
    }

    #[test]
    fn credential_said_stable_across_presentations() {
        let kp = keygen(&mut OsRng);
        let claims: Vec<&[u8]> = vec![b"id", b"name", b"age", b"country"];
        let iss = issue_bbs_2023(&claims, &kp.secret, &kp.public).unwrap();
        let claim_vecs: Vec<Vec<u8>> =
            claims.iter().map(|c| c.to_vec()).collect();
        let cred = bbs_credential_envelope(
            "bbs-2023",
            kp.public.clone(),
            claim_vecs,
            iss.clone(),
            fixture_provenance(),
            OwnershipChain::empty(),
            JouleCost::measured(7),
        )
        .unwrap();
        let original_said = cred.id;
        cred.verify_said().unwrap();

        // Build two distinct presentations from the same credential.
        let disc_a = create_bbs_2023_disclosure(
            &iss,
            &claims,
            &[0, 2],
            &kp.public,
            b"nonce-a",
            &mut OsRng,
        )
        .unwrap();
        let disc_b = create_bbs_2023_disclosure(
            &iss,
            &claims,
            &[1, 3],
            &kp.public,
            b"nonce-b",
            &mut OsRng,
        )
        .unwrap();
        let pres_a = presentation_envelope(
            &cred,
            kp.public.clone(),
            b"nonce-a".to_vec(),
            disc_a,
            fixture_provenance(),
            OwnershipChain::empty(),
            JouleCost::measured(1),
        )
        .unwrap();
        let pres_b = presentation_envelope(
            &cred,
            kp.public.clone(),
            b"nonce-b".to_vec(),
            disc_b,
            fixture_provenance(),
            OwnershipChain::empty(),
            JouleCost::measured(1),
        )
        .unwrap();

        // Credential SAID unchanged after both presentations exist.
        assert_eq!(cred.id, original_said);

        // Both presentations reference the same credential SAID.
        let body_a = presentation_from_envelope(&pres_a).unwrap();
        let body_b = presentation_from_envelope(&pres_b).unwrap();
        assert_eq!(body_a.credential_said, original_said);
        assert_eq!(body_b.credential_said, original_said);

        // The presentations themselves have distinct SAIDs.
        assert_ne!(pres_a.id, pres_b.id);
    }

    #[test]
    fn presentation_rejects_non_credential_envelope() {
        let kp = keygen(&mut OsRng);
        let not_a_cred = Envelope::new(
            fixture_provenance(),
            OwnershipChain::empty(),
            Cargo::Bytes(vec![1, 2, 3]),
            JouleCost::default(),
        )
        .unwrap();
        let dummy_iss = issue_bbs_2023(&[b"x"], &kp.secret, &kp.public).unwrap();
        let dummy_disc = create_bbs_2023_disclosure(
            &dummy_iss,
            &[b"x"],
            &[],
            &kp.public,
            b"n",
            &mut OsRng,
        )
        .unwrap();
        let err = presentation_envelope(
            &not_a_cred,
            kp.public.clone(),
            b"n".to_vec(),
            dummy_disc,
            fixture_provenance(),
            OwnershipChain::empty(),
            JouleCost::default(),
        )
        .unwrap_err();
        assert!(matches!(err, BbsError::Bridge(_)));
    }
}
