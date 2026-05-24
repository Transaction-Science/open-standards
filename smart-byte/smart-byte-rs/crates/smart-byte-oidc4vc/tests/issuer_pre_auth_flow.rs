//! End-to-end pre-authorized-code issuance flow.
//!
//! Exercises the full wallet-side pipeline:
//!
//! 1. Issuer publishes metadata + AS metadata.
//! 2. Issuer hands the wallet a pre-authorized credential offer.
//! 3. Wallet builds a token request, the AS returns access_token +
//!    c_nonce.
//! 4. Wallet builds a credential request bearing a JWT proof, attaches
//!    a DPoP header, and the issuer returns the credential.
//! 5. Wallet notifies the issuer of acceptance.

use std::collections::HashMap;

use chrono::Utc;
use smart_byte_oidc4vc::{
    AuthorizationServerMetadata, CredentialConfiguration, CredentialOffer,
    CredentialResponse, IssuerMetadata, NotificationEvent, TokenResponse,
    WalletClient,
};

fn metadata() -> IssuerMetadata {
    let mut configs = HashMap::new();
    configs.insert(
        "UniversityDegree_SD_JWT".to_string(),
        CredentialConfiguration {
            format: "vc+sd-jwt".into(),
            scope: Some("UniversityDegree".into()),
            credential_signing_alg_values_supported: vec!["EdDSA".into()],
            proof_types_supported: HashMap::new(),
            vct: Some("https://example.com/UniversityDegree".into()),
            doctype: None,
            credential_definition: None,
        },
    );
    IssuerMetadata {
        credential_issuer: "https://issuer.example.com".into(),
        authorization_servers: vec!["https://issuer.example.com".into()],
        credential_endpoint: "https://issuer.example.com/credential".into(),
        nonce_endpoint: Some("https://issuer.example.com/nonce".into()),
        notification_endpoint: Some(
            "https://issuer.example.com/notification".into(),
        ),
        deferred_credential_endpoint: None,
        batch_credential_endpoint: None,
        credential_configurations_supported: configs,
        display: vec![],
    }
}

fn as_metadata() -> AuthorizationServerMetadata {
    AuthorizationServerMetadata {
        issuer: "https://issuer.example.com".into(),
        token_endpoint: "https://issuer.example.com/token".into(),
        authorization_endpoint: None,
        pushed_authorization_request_endpoint: None,
        grant_types_supported: vec![
            AuthorizationServerMetadata::PREAUTH_GRANT.into(),
        ],
        dpop_signing_alg_values_supported: vec!["EdDSA".into()],
        code_challenge_methods_supported: vec!["S256".into()],
        pre_authorized_grant_anonymous_access_supported: true,
    }
}

#[test]
fn end_to_end_pre_authorized_flow() {
    // 1. Offer.
    let offer = CredentialOffer::pre_authorized(
        "https://issuer.example.com",
        vec!["UniversityDegree_SD_JWT".into()],
        "pre-auth-code-1",
    );
    let link = offer.to_deep_link().expect("deep link encodes");
    assert!(link.starts_with("openid-credential-offer://"));
    let parsed = CredentialOffer::from_deep_link(&link)
        .expect("offer parses from deep link");
    assert_eq!(parsed, offer);

    // 2. Wallet.
    let mut wallet =
        WalletClient::new(parsed, metadata(), as_metadata()).expect("wallet");

    // 3. Token request (PIN-less).
    let token_request = wallet
        .build_pre_authorized_token_request(None)
        .expect("token request");
    assert_eq!(token_request.grant_type, AuthorizationServerMetadata::PREAUTH_GRANT);
    let now = Utc::now();
    let _token_dpop = wallet.build_token_dpop(now, "jti-token-1");

    // Server-side: synthesise the token response.
    let token_resp = TokenResponse {
        access_token: "at-xyz".into(),
        token_type: "DPoP".into(),
        expires_in: Some(3600),
        refresh_token: None,
        scope: None,
        c_nonce: Some("c-nonce-1".into()),
        c_nonce_expires_in: Some(120),
        authorization_details: None,
    };
    wallet.record_token_response(&token_resp);

    // 4. Credential request with DPoP.
    let credential_dpop = wallet
        .build_credential_dpop(now, "jti-cred-1")
        .expect("dpop with access token");
    assert!(credential_dpop.ath.is_some());
    let credential_request = wallet
        .build_credential_request("ey.proof.payload.sig")
        .expect("credential request");
    credential_request
        .validate()
        .expect("credential request validates");

    // Server-side response.
    let credential_resp = CredentialResponse {
        credential: Some(serde_json::json!(
            "ey.sd-jwt.header.body.sig~disc1~disc2"
        )),
        credentials: vec![],
        transaction_id: None,
        c_nonce: Some("c-nonce-2".into()),
        c_nonce_expires_in: Some(120),
        notification_id: Some("notif-1".into()),
    };
    assert!(!credential_resp.is_deferred());

    // 5. Notification.
    let notif = wallet.build_notification(
        credential_resp.notification_id.clone().unwrap(),
        NotificationEvent::CredentialAccepted,
    );
    notif.validate().expect("notification validates");
    assert_eq!(
        notif.parsed_event().expect("event parses"),
        NotificationEvent::CredentialAccepted
    );
}

#[test]
fn deferred_flow_recognised() {
    let resp = CredentialResponse {
        credential: None,
        credentials: vec![],
        transaction_id: Some("tx-99".into()),
        c_nonce: None,
        c_nonce_expires_in: None,
        notification_id: None,
    };
    assert!(resp.is_deferred());
}
