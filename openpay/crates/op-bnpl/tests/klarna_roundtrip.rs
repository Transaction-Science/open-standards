//! Klarna driver round-trip: initiate → authorize → capture → refund.

use std::collections::BTreeMap;

use op_bnpl::{
    BillingInfo, BnplAcquirer, BnplIntent, ConsumerInfo, IdempotencyKey, KlarnaAcquirer,
    KlarnaRegion, LineItem, RedirectUrls, ShippingInfo,
};
use op_core::{Currency, Money};
use reqwest::Client;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn intent() -> BnplIntent {
    BnplIntent {
        amount: Money::from_minor(15_000, Currency::EUR),
        currency: Currency::EUR,
        line_items: vec![LineItem {
            name: "Widget".into(),
            sku: Some("W-1".into()),
            quantity: 1,
            unit_price: Money::from_minor(15_000, Currency::EUR),
            total_amount: Money::from_minor(15_000, Currency::EUR),
        }],
        shipping: ShippingInfo {
            name: "Bob".into(),
            line1: "Berliner Str 1".into(),
            line2: None,
            city: "Berlin".into(),
            region: "BE".into(),
            postal_code: "10115".into(),
            country: "DE".into(),
        },
        billing: BillingInfo {
            name: "Bob".into(),
            line1: "Berliner Str 1".into(),
            line2: None,
            city: "Berlin".into(),
            region: "BE".into(),
            postal_code: "10115".into(),
            country: "DE".into(),
        },
        consumer: ConsumerInfo {
            email: "bob@example.de".into(),
            phone: None,
            given_name: Some("Bob".into()),
            family_name: Some("Mueller".into()),
            date_of_birth: None,
        },
        idempotency_key: IdempotencyKey::from("idem-klarna-1"),
        redirect_urls: RedirectUrls {
            success: "https://m/ok".into(),
            cancel: "https://m/cancel".into(),
            failure: None,
        },
        metadata: BTreeMap::new(),
    }
}

#[tokio::test]
async fn klarna_full_roundtrip() {
    let server = MockServer::start().await;

    // /payments/v1/sessions
    Mock::given(method("POST"))
        .and(path("/payments/v1/sessions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "session_id": "SESS_1",
            "client_token": "ct_abc"
        })))
        .expect(1)
        .mount(&server)
        .await;

    // /payments/v1/authorizations/{token}/order
    Mock::given(method("POST"))
        .and(path("/payments/v1/authorizations/auth_tok_1/order"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "order_id": "ORD_1",
            "authorized_amount": 15_000,
            "authorized_payment_method": { "type": "PAY_IN_3" }
        })))
        .expect(1)
        .mount(&server)
        .await;

    // /ordermanagement/v1/orders/{id}/captures
    Mock::given(method("POST"))
        .and(path("/ordermanagement/v1/orders/ORD_1/captures"))
        .respond_with(
            ResponseTemplate::new(201)
                .insert_header("Location", "/ordermanagement/v1/orders/ORD_1/captures/CAP_1"),
        )
        .expect(1)
        .mount(&server)
        .await;

    // /ordermanagement/v1/orders/{id}/refunds
    Mock::given(method("POST"))
        .and(path("/ordermanagement/v1/orders/ORD_1/refunds"))
        .respond_with(
            ResponseTemplate::new(201)
                .insert_header("Location", "/ordermanagement/v1/orders/ORD_1/refunds/REF_1"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let acquirer = KlarnaAcquirer::with_base_url(
        Client::new(),
        "u",
        "p",
        KlarnaRegion::Eu,
        server.uri(),
    );

    let session = acquirer.initiate(&intent()).await.unwrap();
    assert_eq!(session.provider_ref, "SESS_1");
    assert_eq!(session.client_token.as_deref(), Some("ct_abc"));

    let auth = acquirer.authorize(&session, "auth_tok_1").await.unwrap();
    assert_eq!(auth.provider_ref, "ORD_1");
    assert_eq!(auth.authorized_amount.minor_units, 15_000);

    let captured = acquirer.capture(&auth, None).await.unwrap();
    assert_eq!(captured.amount.minor_units, 15_000);

    let refund = acquirer
        .refund(&captured, Money::from_minor(5_000, Currency::EUR))
        .await
        .unwrap();
    assert_eq!(refund.amount.minor_units, 5_000);
}

#[tokio::test]
async fn klarna_session_4xx_maps_to_provider_rejected() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/payments/v1/sessions"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({
            "error_code": "merchant_blocked",
            "error_messages": ["merchant agreement suspended"]
        })))
        .mount(&server)
        .await;

    let acquirer = KlarnaAcquirer::with_base_url(
        Client::new(),
        "u",
        "p",
        KlarnaRegion::Eu,
        server.uri(),
    );
    let err = acquirer.initiate(&intent()).await.unwrap_err();
    match err {
        op_bnpl::Error::ProviderRejected {
            status,
            code,
            message,
        } => {
            assert_eq!(status, 403);
            assert_eq!(code, "merchant_blocked");
            assert!(message.contains("suspended"));
        }
        other => panic!("expected ProviderRejected, got {other:?}"),
    }
}
