//! Integration tests against W3C VCDM 2.0 test vectors and the
//! VC-JWT / SD-JWT / Status List / cargo-bridge end-to-end paths.

use chrono::TimeZone;
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use serde_json::json;
use smart_byte_core::{JouleCost, OwnershipChain, Provenance, Said};
use smart_byte_vc::{
    BitstringStatusList, CredentialSubject, Disclosure, Holder, Issuer,
    StatusListCredential, StatusPurpose, VC_CARGO_TYPE_URI, VcBuilder,
    VerifiableCredential, VerifiablePresentation, check_status,
    issue_data_integrity, issue_sd_jwt, issue_vc_jwt, present_sd_jwt,
    vc_envelope, vc_from_envelope, verify_data_integrity, verify_sd_jwt,
    verify_vc_jwt,
};

fn load_fixture(name: &str) -> VerifiableCredential {
    let path = format!(
        "{}/tests/fixtures/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let bytes = std::fs::read(&path).expect("fixture read");
    serde_json::from_slice(&bytes).expect("fixture parse")
}

#[test]
fn vcdm_v2_example_roundtrips() {
    let vc = load_fixture("vcdm_v2_example.json");
    vc.validate_shape().expect("shape");
    let jcs_a = vc.to_jcs().expect("jcs a");
    let again: VerifiableCredential =
        serde_json::from_slice(&jcs_a).expect("reparse");
    let jcs_b = again.to_jcs().expect("jcs b");
    assert_eq!(jcs_a, jcs_b, "JCS round-trip must be stable");
    assert_eq!(vc, again, "structural round-trip");
    assert_eq!(vc.issuer.id().as_str(), "https://university.example/issuers/565049");
}

#[test]
fn vcdm_v2_data_integrity_sign_and_verify() {
    let vc = load_fixture("vcdm_v2_example.json");
    let sk = SigningKey::generate(&mut OsRng);
    let vk = sk.verifying_key();
    let signed = issue_data_integrity(
        vc,
        "did:example:issuer#keys-1".parse().expect("iri"),
        chrono::Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).single().expect("ts"),
        &sk,
    )
    .expect("issue DI");
    verify_data_integrity(&signed, &vk).expect("verify DI");
}

#[test]
fn vc_jwt_known_key_roundtrip() {
    let vc = load_fixture("vcdm_v2_example.json");
    let sk = SigningKey::from_bytes(&[7u8; 32]);
    let vk = sk.verifying_key();
    let (jwt, proof) = issue_vc_jwt(
        &vc,
        &sk,
        Some("did:example:issuer#keys-1".to_string()),
    )
    .expect("issue jwt");
    assert!(jwt.split('.').count() == 3);
    assert_eq!(proof.type_, "JwtProof2020");
    let decoded = verify_vc_jwt(&jwt, &vk).expect("verify jwt");
    assert_eq!(decoded, vc);
}

#[test]
fn sd_jwt_discloses_two_of_five() {
    let sk = SigningKey::generate(&mut OsRng);
    let vk = sk.verifying_key();
    let payload = json!({
        "given_name": "Alice",
        "family_name": "Lockhart",
        "birthdate": "1990-01-15",
        "email": "alice@example.org",
        "country": "GB",
        "iss": "did:example:issuer",
    });
    let (combined, disclosures, _proof) = issue_sd_jwt(
        &payload,
        &[
            "given_name",
            "family_name",
            "birthdate",
            "email",
            "country",
        ],
        &sk,
        Some("did:example:issuer#keys-1".to_string()),
    )
    .expect("issue sd-jwt");
    assert_eq!(disclosures.len(), 5);
    let presented = present_sd_jwt(&combined, &["given_name", "country"])
        .expect("holder present");
    let merged = verify_sd_jwt(&presented, &vk).expect("verify presented");
    assert_eq!(merged["given_name"], "Alice");
    assert_eq!(merged["country"], "GB");
    assert!(merged.get("family_name").is_none());
    assert!(merged.get("birthdate").is_none());
    assert!(merged.get("email").is_none());
}

#[test]
fn sd_jwt_disclosure_helpers_are_consistent() {
    let d = Disclosure {
        salt: "_26bc4LT-ac6q2KI6cBW5es".into(),
        name: "family_name".into(),
        value: json!("Möbius"),
    };
    let b64 = d.to_base64().expect("encode");
    // re-decoding the base64 yields the disclosure JSON array
    let bytes = base64::Engine::decode(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD,
        &b64,
    )
    .expect("b64 decode");
    let arr: serde_json::Value =
        serde_json::from_slice(&bytes).expect("disclosure json");
    assert_eq!(arr[0], "_26bc4LT-ac6q2KI6cBW5es");
    assert_eq!(arr[1], "family_name");
    assert_eq!(arr[2], "Möbius");
}

#[test]
fn status_list_revokes_index_42() {
    // Issue a fresh status list credential with 1024 indices.
    let subj = CredentialSubject {
        id: Some("https://example.com/credentials/status/3"
            .parse()
            .expect("iri")),
        claims: serde_json::Map::new(),
    };
    let status_vc = VcBuilder::new()
        .type_tag("BitstringStatusListCredential")
        .issuer(Issuer::Uri("did:example:issuer".parse().expect("iri")))
        .subject(subj)
        .build()
        .expect("status vc");
    let mut slc = StatusListCredential {
        vc: status_vc,
        purpose: StatusPurpose::revocation(),
        bitstring: BitstringStatusList::new(1024),
    };
    slc.refresh_subject().expect("initial refresh");
    let initial = slc.vc.credential_subject[0]
        .claims
        .get("encodedList")
        .and_then(|v| v.as_str())
        .expect("encodedList")
        .to_string();
    assert!(!check_status(&initial, 1024, 42).expect("check"));

    // Revoke index 42.
    slc.bitstring.set(42, true).expect("revoke");
    slc.refresh_subject().expect("refresh after revoke");
    let after = slc.vc.credential_subject[0]
        .claims
        .get("encodedList")
        .and_then(|v| v.as_str())
        .expect("encodedList")
        .to_string();
    assert!(check_status(&after, 1024, 42).expect("check 42"));
    assert!(!check_status(&after, 1024, 41).expect("check 41"));
    assert!(!check_status(&after, 1024, 43).expect("check 43"));
    assert_ne!(initial, after);
}

#[test]
fn status_list_entry_fixture_parses() {
    // The fixture carries a credentialStatus pointing at index 42.
    let vc = load_fixture("vcdm_v2_with_status.json");
    let status = vc.credential_status.as_ref().expect("status");
    assert_eq!(status.type_, "BitstringStatusListEntry");
    assert_eq!(
        status.status_purpose.as_deref(),
        Some("revocation")
    );
    assert_eq!(status.status_list_index.as_deref(), Some("42"));
}

#[test]
fn cargo_bridge_envelopes_a_vc_with_stable_said() {
    let vc = load_fixture("vcdm_v2_example.json");
    let prov = Provenance::new(
        Said::hash(b"vc-issuer"),
        chrono::Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).single().expect("ts"),
        b"vc-auth".to_vec(),
    );
    let env_a = vc_envelope(
        &vc,
        prov.clone(),
        OwnershipChain::empty(),
        JouleCost::measured(11),
    )
    .expect("envelope a");
    let env_b = vc_envelope(
        &vc,
        prov,
        OwnershipChain::empty(),
        JouleCost::measured(11),
    )
    .expect("envelope b");
    // SAID stability
    assert_eq!(env_a.id, env_b.id);
    env_a.verify_said().expect("said verify");
    // Cargo carries the VC type uri
    match &env_a.cargo {
        smart_byte_core::Cargo::Custom { type_uri, .. } => {
            assert_eq!(type_uri, VC_CARGO_TYPE_URI);
        }
        _ => panic!("expected Cargo::Custom"),
    }
    // Round-trip extraction
    let back = vc_from_envelope(&env_a).expect("from envelope");
    assert_eq!(back, vc);
}

#[test]
fn presentation_with_holder_and_credential() {
    let vc = load_fixture("vcdm_v2_example.json");
    let holder = Holder::new("did:example:alice".parse().expect("did"));
    let vp = VerifiablePresentation::new()
        .expect("vp")
        .with_holder(holder.did.clone())
        .with_credential(vc);
    vp.validate_shape().expect("shape");
    assert_eq!(vp.verifiable_credential.len(), 1);
    assert_eq!(
        vp.holder.as_ref().map(|d| d.to_string()),
        Some("did:example:alice".to_string())
    );
}
