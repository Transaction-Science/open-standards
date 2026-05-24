//! DCQL serialisation + matching tests.

use serde_json::json;
use smart_byte_oidc4vc::{
    ClaimQuery, CredentialMeta, CredentialQuery, DcqlQuery, PathSegment,
    evaluate_claim, evaluate_credential,
};

fn pid_credential() -> serde_json::Value {
    json!({
        "vct": "https://eu.europa.ec.eudi/pid/1",
        "credentialSubject": {
            "given_name": "Alice",
            "family_name": "Citizen",
            "birth_date": "1990-04-12"
        }
    })
}

#[test]
fn query_serialises() {
    let q = DcqlQuery {
        credentials: vec![CredentialQuery {
            id: "pid".into(),
            format: "dc+sd-jwt".into(),
            require: true,
            meta: CredentialMeta {
                vct_values: vec!["https://eu.europa.ec.eudi/pid/1".into()],
                doctype_value: None,
                type_values: vec![],
            },
            claims: vec![
                ClaimQuery::path(vec![
                    PathSegment::Key("credentialSubject".into()),
                    PathSegment::Key("given_name".into()),
                ])
                .with_id("c-gn"),
                ClaimQuery::path(vec![
                    PathSegment::Key("credentialSubject".into()),
                    PathSegment::Key("family_name".into()),
                ])
                .with_id("c-fn"),
            ],
            claim_sets: vec![vec!["c-gn".into(), "c-fn".into()]],
        }],
        credential_sets: vec![],
    };
    q.validate().expect("query validates");
    let j = serde_json::to_string(&q).unwrap();
    let back: DcqlQuery = serde_json::from_str(&j).unwrap();
    assert_eq!(back, q);
}

#[test]
fn evaluates_each_claim_path() {
    let cred = pid_credential();
    let gn = ClaimQuery::path(vec![
        PathSegment::Key("credentialSubject".into()),
        PathSegment::Key("given_name".into()),
    ]);
    let v = evaluate_claim(&cred, &gn).expect("given_name resolves");
    assert_eq!(v, &serde_json::Value::from("Alice"));
}

#[test]
fn rejects_unknown_path() {
    let cred = pid_credential();
    let q = ClaimQuery::path(vec![
        PathSegment::Key("credentialSubject".into()),
        PathSegment::Key("missing".into()),
    ]);
    assert!(evaluate_claim(&cred, &q).is_none());
}

#[test]
fn evaluates_credential_with_claim_set() {
    let cred = pid_credential();
    let q = CredentialQuery {
        id: "pid".into(),
        format: "dc+sd-jwt".into(),
        require: true,
        meta: CredentialMeta::default(),
        claims: vec![
            ClaimQuery::path(vec![
                PathSegment::Key("credentialSubject".into()),
                PathSegment::Key("given_name".into()),
            ])
            .with_id("gn"),
            ClaimQuery::path(vec![
                PathSegment::Key("credentialSubject".into()),
                PathSegment::Key("does_not_exist".into()),
            ])
            .with_id("missing"),
        ],
        claim_sets: vec![vec!["gn".into()]],
    };
    let matched = evaluate_credential(&cred, &q).expect("matches set");
    assert!(matched.contains(&"gn".to_string()));
}

#[test]
fn duplicate_credential_ids_rejected() {
    let q = DcqlQuery {
        credentials: vec![
            CredentialQuery {
                id: "a".into(),
                format: "dc+sd-jwt".into(),
                require: true,
                meta: CredentialMeta::default(),
                claims: vec![],
                claim_sets: vec![],
            },
            CredentialQuery {
                id: "a".into(),
                format: "dc+sd-jwt".into(),
                require: true,
                meta: CredentialMeta::default(),
                claims: vec![],
                claim_sets: vec![],
            },
        ],
        credential_sets: vec![],
    };
    assert!(q.validate().is_err());
}
