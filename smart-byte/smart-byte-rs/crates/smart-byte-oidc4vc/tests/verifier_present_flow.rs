//! Verifier-side OID4VP flow, covering both PE 2.0 and DCQL.

use serde_json::Value;
use smart_byte_oidc4vc::{
    ClaimQuery, CredentialMeta, CredentialQuery, DcqlQuery, PathSegment,
    PresentationSubmission, SubmissionDescriptor, Verifier, VpAuthResponse,
    simple_definition,
};

#[test]
fn pe_v2_round_trip() {
    let def = simple_definition(
        "pdef-1",
        "id-degree",
        "$.credentialSubject.degree.name",
    );
    let verifier = Verifier::from_presentation_definition(
        "https://rp.example.com",
        "n-vp-1",
        "https://rp.example.com/cb",
        def,
    )
    .expect("verifier builds");

    let response = VpAuthResponse {
        vp_token: Value::from("ey.vp.payload.sig"),
        presentation_submission: Some(PresentationSubmission {
            id: "sub-1".into(),
            definition_id: "pdef-1".into(),
            descriptor_map: vec![SubmissionDescriptor {
                id: "id-degree".into(),
                format: "jwt_vp".into(),
                path: "$".into(),
                path_nested: None,
            }],
        }),
        state: None,
    };
    verifier
        .validate_response(&response)
        .expect("submission validates");
}

#[test]
fn pe_v2_missing_submission_rejected() {
    let def = simple_definition(
        "pdef-1",
        "id-degree",
        "$.credentialSubject.degree.name",
    );
    let verifier = Verifier::from_presentation_definition(
        "https://rp.example.com",
        "n-vp-1",
        "https://rp.example.com/cb",
        def,
    )
    .unwrap();
    let response = VpAuthResponse {
        vp_token: Value::from("ey.vp.payload.sig"),
        presentation_submission: None,
        state: None,
    };
    assert!(verifier.validate_response(&response).is_err());
}

#[test]
fn dcql_response_round_trip() {
    let query = DcqlQuery {
        credentials: vec![CredentialQuery {
            id: "deg".into(),
            format: "vc+sd-jwt".into(),
            require: true,
            meta: CredentialMeta::default(),
            claims: vec![ClaimQuery::path(vec![PathSegment::Key(
                "vct".into(),
            )])],
            claim_sets: vec![],
        }],
        credential_sets: vec![],
    };
    let verifier = Verifier::from_dcql(
        "https://rp.example.com",
        "n-vp-2",
        "https://rp.example.com/cb",
        query,
    )
    .unwrap();

    let response = VpAuthResponse {
        vp_token: serde_json::json!({"deg": "ey.sd-jwt.x"}),
        presentation_submission: None,
        state: None,
    };
    verifier
        .validate_response(&response)
        .expect("dcql response validates");
}

#[test]
fn dcql_response_with_submission_rejected() {
    let query = DcqlQuery {
        credentials: vec![CredentialQuery {
            id: "deg".into(),
            format: "vc+sd-jwt".into(),
            require: true,
            meta: CredentialMeta::default(),
            claims: vec![ClaimQuery::path(vec![PathSegment::Key(
                "vct".into(),
            )])],
            claim_sets: vec![],
        }],
        credential_sets: vec![],
    };
    let verifier = Verifier::from_dcql(
        "https://rp.example.com",
        "n-vp-2",
        "https://rp.example.com/cb",
        query,
    )
    .unwrap();
    let response = VpAuthResponse {
        vp_token: Value::from("ey.x"),
        presentation_submission: Some(PresentationSubmission {
            id: "s-1".into(),
            definition_id: "anything".into(),
            descriptor_map: vec![],
        }),
        state: None,
    };
    assert!(verifier.validate_response(&response).is_err());
}
