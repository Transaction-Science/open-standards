//! Structural conformance to the ISO/IEC 18013-5 Annex D example mDL.
//!
//! Annex D of ISO/IEC 18013-5:2021 publishes a worked example of an mDL
//! using the `org.iso.18013.5.1` namespace. We reproduce its structure
//! deterministically here (the canonical field set, the SHA-256 digest
//! algorithm, the MSO version `1.0`, the docType `org.iso.18013.5.1.mDL`,
//! a single namespace) and assert that an issuer constructed from that
//! input produces a fully verifiable credential.
//!
//! The exact byte-for-byte signed MSO bytes from Annex D are not
//! reproduced here — Annex D's private key is fixed, but our test runs
//! over a freshly generated key so the wire bytes differ. The
//! load-bearing property under test is the structural conformance
//! between this implementation's MSO and the Annex D shape.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{TimeZone, Utc};
use ciborium::value::Value as CborValue;
use smart_byte_mdl::{
    FixedTime, Issuer, IssuerKey, MobileSecurityObject, NS_MDL, TrustAnchor,
    ValidityInfo, Verifier, mdl,
};

/// Canonical Annex-D mDL element set.
fn annex_d_claims() -> BTreeMap<String, CborValue> {
    let mut m = BTreeMap::new();
    m.insert(mdl::FAMILY_NAME.into(), CborValue::Text("Doe".into()));
    m.insert(mdl::GIVEN_NAME.into(), CborValue::Text("Jane".into()));
    m.insert(mdl::BIRTH_DATE.into(), CborValue::Text("1990-01-01".into()));
    m.insert(mdl::ISSUE_DATE.into(), CborValue::Text("2024-01-01".into()));
    m.insert(mdl::EXPIRY_DATE.into(), CborValue::Text("2029-01-01".into()));
    m.insert(mdl::ISSUING_COUNTRY.into(), CborValue::Text("US".into()));
    m.insert(
        mdl::ISSUING_AUTHORITY.into(),
        CborValue::Text("Example State DMV".into()),
    );
    m.insert(
        mdl::DOCUMENT_NUMBER.into(),
        CborValue::Text("123456789".into()),
    );
    m.insert(mdl::PORTRAIT.into(), CborValue::Bytes(vec![0xFF; 32]));
    m.insert(
        mdl::DRIVING_PRIVILEGES.into(),
        CborValue::Array(vec![CborValue::Map(vec![
            (
                CborValue::Text("vehicle_category_code".into()),
                CborValue::Text("A".into()),
            ),
            (
                CborValue::Text("issue_date".into()),
                CborValue::Text("2018-08-09".into()),
            ),
            (
                CborValue::Text("expiry_date".into()),
                CborValue::Text("2024-10-20".into()),
            ),
        ])]),
    );
    m.insert(
        mdl::UN_DISTINGUISHING_SIGN.into(),
        CborValue::Text("USA".into()),
    );
    m.insert(mdl::age_over(18), CborValue::Bool(true));
    m.insert(mdl::age_over(21), CborValue::Bool(true));
    m
}

#[test]
fn annex_d_structure_round_trips_through_issuer_and_verifier() {
    let issuer_key = IssuerKey::generate_es256();
    let device_key = IssuerKey::generate_es256();
    let validity = ValidityInfo {
        signed: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
        valid_from: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
        valid_until: Utc.with_ymd_and_hms(2029, 1, 1, 0, 0, 0).unwrap(),
        expected_update: None,
    };
    let issuer = Issuer::new(
        IssuerKey {
            alg: issuer_key.alg,
            p256: issuer_key.p256.clone(),
        },
        Vec::new(),
        validity,
    );
    let mut ns: BTreeMap<String, BTreeMap<String, CborValue>> = BTreeMap::new();
    ns.insert(NS_MDL.into(), annex_d_claims());
    let doc = issuer
        .issue(
            "org.iso.18013.5.1.mDL",
            ns,
            device_key.cose_public_key(),
            b"annex-d",
        )
        .unwrap();
    // Structural assertions from Annex D.
    assert_eq!(doc.doc_type, "org.iso.18013.5.1.mDL");
    assert!(doc.issuer_signed.name_spaces.contains_key(NS_MDL));
    let mso = MobileSecurityObject::from_cbor(&doc.issuer_signed.issuer_auth.payload).unwrap();
    assert_eq!(mso.version, "1.0");
    assert_eq!(mso.digest_algorithm, "SHA-256");
    assert_eq!(mso.doc_type, "org.iso.18013.5.1.mDL");
    assert_eq!(
        mso.value_digests[NS_MDL].len(),
        doc.issuer_signed.name_spaces[NS_MDL].len()
    );
    // Verifier accepts at a time within the validity window.
    let anchor = TrustAnchor::es256(issuer_key.p256.verifying_key(), Some("Example DMV".into()));
    let verifier = Verifier::new(
        vec![anchor],
        Arc::new(FixedTime(
            Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap(),
        )),
    );
    let claims = verifier.verify(&doc).unwrap();
    assert_eq!(
        claims.claims[NS_MDL][mdl::FAMILY_NAME],
        CborValue::Text("Doe".into())
    );
    assert_eq!(
        claims.claims[NS_MDL][&mdl::age_over(21)],
        CborValue::Bool(true)
    );
}
