//! End-to-end issue / present / verify integration tests.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{TimeZone, Utc};
use ciborium::value::Value as CborValue;
use smart_byte_mdl::{
    FixedTime, Issuer, IssuerKey, ItemsRequest, MobileDoc, NS_MDL,
    P256DeviceSigner, SessionTranscript, TrustAnchor, ValidityInfo, Verifier,
    present,
};

fn build_validity() -> ValidityInfo {
    ValidityInfo {
        signed: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
        valid_from: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
        valid_until: Utc.with_ymd_and_hms(2030, 1, 1, 0, 0, 0).unwrap(),
        expected_update: None,
    }
}

fn build_claims() -> BTreeMap<String, BTreeMap<String, CborValue>> {
    let mut ns: BTreeMap<String, BTreeMap<String, CborValue>> = BTreeMap::new();
    let mut inner: BTreeMap<String, CborValue> = BTreeMap::new();
    inner.insert("family_name".into(), CborValue::Text("Doe".into()));
    inner.insert("given_name".into(), CborValue::Text("Jane".into()));
    inner.insert("birth_date".into(), CborValue::Text("1990-04-12".into()));
    inner.insert("age_over_21".into(), CborValue::Bool(true));
    inner.insert("document_number".into(), CborValue::Text("Z-1234".into()));
    ns.insert(NS_MDL.into(), inner);
    ns
}

struct Setup {
    issuer_key: IssuerKey,
    device_key: IssuerKey,
    doc: MobileDoc,
}

fn issue() -> Setup {
    let issuer_key = IssuerKey::generate_es256();
    let device_key = IssuerKey::generate_es256();
    let issuer = Issuer::new(
        IssuerKey {
            alg: issuer_key.alg,
            p256: issuer_key.p256.clone(),
        },
        Vec::new(),
        build_validity(),
    );
    let doc = issuer
        .issue(
            "org.iso.18013.5.1.mDL",
            build_claims(),
            device_key.cose_public_key(),
            b"integration-seed",
        )
        .unwrap();
    Setup {
        issuer_key,
        device_key,
        doc,
    }
}

#[test]
fn issue_verify_full_disclosure() {
    let setup = issue();
    let anchor = TrustAnchor::es256(setup.issuer_key.p256.verifying_key(), None);
    let verifier = Verifier::new(
        vec![anchor],
        Arc::new(FixedTime(
            Utc.with_ymd_and_hms(2027, 1, 1, 0, 0, 0).unwrap(),
        )),
    );
    let claims = verifier.verify(&setup.doc).unwrap();
    assert_eq!(claims.doc_type, "org.iso.18013.5.1.mDL");
    let ns = &claims.claims[NS_MDL];
    assert_eq!(ns["family_name"], CborValue::Text("Doe".into()));
    assert_eq!(ns["given_name"], CborValue::Text("Jane".into()));
    assert_eq!(ns["age_over_21"], CborValue::Bool(true));
    assert_eq!(ns.len(), 5);
}

#[test]
fn selective_disclosure_reveals_only_requested_items() {
    let setup = issue();
    let signer = P256DeviceSigner {
        key: setup.device_key.p256.clone(),
    };
    let transcript =
        SessionTranscript::for_oid4vp("client-id", "https://verifier", "n", "m");
    let mut req: BTreeMap<String, Vec<String>> = BTreeMap::new();
    req.insert(
        NS_MDL.into(),
        vec!["family_name".into(), "age_over_21".into()],
    );
    let presented = present(
        &setup.doc,
        &req,
        &signer,
        &transcript,
        BTreeMap::new(),
    )
    .unwrap();

    let anchor = TrustAnchor::es256(setup.issuer_key.p256.verifying_key(), None);
    let verifier = Verifier::new(
        vec![anchor],
        Arc::new(FixedTime(
            Utc.with_ymd_and_hms(2027, 1, 1, 0, 0, 0).unwrap(),
        )),
    )
    .with_transcript(transcript);
    let claims = verifier.verify(&presented).unwrap();
    let ns = &claims.claims[NS_MDL];
    assert_eq!(ns.len(), 2);
    assert!(ns.contains_key("family_name"));
    assert!(ns.contains_key("age_over_21"));
    assert!(!ns.contains_key("birth_date"));
    assert!(!ns.contains_key("document_number"));
    assert!(!ns.contains_key("given_name"));
}

#[test]
fn tampered_issuer_signed_item_is_rejected() {
    let mut setup = issue();
    let items = setup
        .doc
        .issuer_signed
        .name_spaces
        .get_mut(NS_MDL)
        .unwrap();
    for it in items.iter_mut() {
        if it.element_identifier == "family_name" {
            it.element_value = CborValue::Text("Mallory".into());
            break;
        }
    }
    let anchor = TrustAnchor::es256(setup.issuer_key.p256.verifying_key(), None);
    let verifier = Verifier::new(
        vec![anchor],
        Arc::new(FixedTime(
            Utc.with_ymd_and_hms(2027, 1, 1, 0, 0, 0).unwrap(),
        )),
    );
    let err = verifier.verify(&setup.doc).unwrap_err();
    assert!(
        matches!(err, smart_byte_mdl::MdlError::DigestMismatch { .. }),
        "got {err:?}"
    );
}

#[test]
fn expired_validity_info_is_rejected() {
    let setup = issue();
    let anchor = TrustAnchor::es256(setup.issuer_key.p256.verifying_key(), None);
    // 2031 is after validUntil 2030.
    let verifier = Verifier::new(
        vec![anchor],
        Arc::new(FixedTime(
            Utc.with_ymd_and_hms(2031, 1, 1, 0, 0, 0).unwrap(),
        )),
    );
    let err = verifier.verify(&setup.doc).unwrap_err();
    assert!(
        matches!(err, smart_byte_mdl::MdlError::ValidityOutOfRange { .. }),
        "got {err:?}"
    );
}

#[test]
fn device_signed_with_wrong_key_is_rejected() {
    let setup = issue();
    // Sign with a wrong device key — the MSO commits to the original.
    let wrong_device = IssuerKey::generate_es256();
    let signer = P256DeviceSigner {
        key: wrong_device.p256,
    };
    let transcript =
        SessionTranscript::for_oid4vp("client-id", "https://verifier", "n", "m");
    let mut req: BTreeMap<String, Vec<String>> = BTreeMap::new();
    req.insert(NS_MDL.into(), vec!["family_name".into()]);
    let presented = present(
        &setup.doc,
        &req,
        &signer,
        &transcript,
        BTreeMap::new(),
    )
    .unwrap();
    let anchor = TrustAnchor::es256(setup.issuer_key.p256.verifying_key(), None);
    let verifier = Verifier::new(
        vec![anchor],
        Arc::new(FixedTime(
            Utc.with_ymd_and_hms(2027, 1, 1, 0, 0, 0).unwrap(),
        )),
    )
    .with_transcript(transcript);
    let err = verifier.verify(&presented).unwrap_err();
    assert!(
        matches!(err, smart_byte_mdl::MdlError::Signature(_)),
        "got {err:?}"
    );
}

#[test]
fn items_request_drives_disclosure() {
    let setup = issue();
    let mut ns: BTreeMap<String, BTreeMap<String, bool>> = BTreeMap::new();
    let mut inner: BTreeMap<String, bool> = BTreeMap::new();
    inner.insert("family_name".into(), false);
    inner.insert("age_over_21".into(), false);
    ns.insert(NS_MDL.into(), inner);
    let req = ItemsRequest {
        doc_type: "org.iso.18013.5.1.mDL".into(),
        name_spaces: ns,
    };
    let signer = P256DeviceSigner {
        key: setup.device_key.p256,
    };
    let transcript =
        SessionTranscript::for_oid4vp("client-id", "https://verifier", "n", "m");
    let presented = present(
        &setup.doc,
        &req.requested(),
        &signer,
        &transcript,
        BTreeMap::new(),
    )
    .unwrap();
    let anchor = TrustAnchor::es256(setup.issuer_key.p256.verifying_key(), None);
    let verifier = Verifier::new(
        vec![anchor],
        Arc::new(FixedTime(
            Utc.with_ymd_and_hms(2027, 1, 1, 0, 0, 0).unwrap(),
        )),
    )
    .with_transcript(transcript);
    let claims = verifier.verify(&presented).unwrap();
    let ns = &claims.claims[NS_MDL];
    assert_eq!(ns.len(), 2);
    assert!(ns.contains_key("family_name"));
    assert!(ns.contains_key("age_over_21"));
}
