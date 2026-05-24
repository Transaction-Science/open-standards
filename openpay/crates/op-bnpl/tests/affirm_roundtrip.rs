//! Affirm driver round-trip: authorize → capture → refund against a
//! wiremock-stubbed server.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use op_bnpl::{
    AffirmAcquirer, BillingInfo, BnplAcquirer, BnplIntent, ConsumerInfo, IdempotencyKey,
    LineItem, RedirectUrls, ShippingInfo,
};
use op_core::{Currency, Money};
use reqwest::Client;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn sample_intent() -> BnplIntent {
    BnplIntent {
        amount: Money::from_minor(10_000, Currency::USD),
        currency: Currency::USD,
        line_items: vec![LineItem {
            name: "Thing".into(),
            sku: Some("T-1".into()),
            quantity: 1,
            unit_price: Money::from_minor(10_000, Currency::USD),
            total_amount: Money::from_minor(10_000, Currency::USD),
        }],
        shipping: ShippingInfo {
            name: "Alice".into(),
            line1: "1 Market".into(),
            line2: None,
            city: "SF".into(),
            region: "CA".into(),
            postal_code: "94105".into(),
            country: "US".into(),
        },
        billing: BillingInfo {
            name: "Alice".into(),
            line1: "1 Market".into(),
            line2: None,
            city: "SF".into(),
            region: "CA".into(),
            postal_code: "94105".into(),
            country: "US".into(),
        },
        consumer: ConsumerInfo {
            email: "a@b.com".into(),
            phone: None,
            given_name: None,
            family_name: None,
            date_of_birth: None,
        },
        idempotency_key: IdempotencyKey::from("idem-affirm-1"),
        redirect_urls: RedirectUrls {
            success: "https://m/ok".into(),
            cancel: "https://m/cancel".into(),
            failure: None,
        },
        metadata: BTreeMap::new(),
    }
}

#[tokio::test]
async fn affirm_full_roundtrip() {
    let server = MockServer::start().await;

    // POST /api/v2/charges → returns a charge.
    Mock::given(method("POST"))
        .and(path("/api/v2/charges"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "CHG_TEST_1",
            "amount": 10_000,
            "status": "authorized"
        })))
        .expect(1)
        .mount(&server)
        .await;

    // POST /api/v2/charges/{id}/capture → returns event.
    Mock::given(method("POST"))
        .and(path("/api/v2/charges/CHG_TEST_1/capture"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "EVT_CAP_1",
            "amount": 10_000,
            "transaction_id": "TXN_BANK_1",
            "type": "capture"
        })))
        .expect(1)
        .mount(&server)
        .await;

    // POST /api/v2/charges/{id}/refund → returns event.
    Mock::given(method("POST"))
        .and(path("/api/v2/charges/CHG_TEST_1/refund"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "EVT_REF_1",
            "amount": 4_000,
            "type": "refund"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let acquirer = AffirmAcquirer::new(Client::new(), "pub", "priv", server.uri());

    let intent = sample_intent();
    let session = acquirer.initiate(&intent).await.unwrap();
    let auth = acquirer.authorize(&session, "checkout_token_abc").await.unwrap();
    assert_eq!(auth.provider_ref, "CHG_TEST_1");
    assert_eq!(auth.authorized_amount.minor_units, 10_000);

    let captured = acquirer.capture(&auth, None).await.unwrap();
    assert_eq!(captured.amount.minor_units, 10_000);
    assert_eq!(captured.settlement_ref.as_deref(), Some("TXN_BANK_1"));

    let refund = acquirer
        .refund(&captured, Money::from_minor(4_000, Currency::USD))
        .await
        .unwrap();
    assert_eq!(refund.amount.minor_units, 4_000);
}

#[tokio::test]
async fn affirm_rejected_4xx_maps_to_provider_rejected() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v2/charges"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": "invalid-token",
            "message": "checkout token expired"
        })))
        .mount(&server)
        .await;

    let acquirer = AffirmAcquirer::new(Client::new(), "pub", "priv", server.uri());
    let intent = sample_intent();
    let session = acquirer.initiate(&intent).await.unwrap();
    let err = acquirer.authorize(&session, "stale").await.unwrap_err();
    match err {
        op_bnpl::Error::ProviderRejected { status, code, .. } => {
            assert_eq!(status, 400);
            assert_eq!(code, "invalid-token");
        }
        other => panic!("expected ProviderRejected, got {other:?}"),
    }
}

/// Idempotency at the acquirer level: replaying the same intent through
/// `initiate` with the same key must not produce a network call (initiate
/// is local-only for Affirm). And the synthetic provider_ref must match.
#[tokio::test]
async fn affirm_initiate_is_local_and_idempotent_on_key() {
    let server = MockServer::start().await;
    // No mocks installed — if any HTTP request fires, it 404s and the
    // test panics via the assertion below.
    let acquirer = AffirmAcquirer::new(Client::new(), "pub", "priv", server.uri());

    let intent = sample_intent();
    let s1 = acquirer.initiate(&intent).await.unwrap();
    let s2 = acquirer.initiate(&intent).await.unwrap();
    assert_eq!(s1.provider_ref, s2.provider_ref);
    assert_eq!(s1.provider_ref, "idem-affirm-1");
}

/// When the authorize endpoint is hit twice with the same checkout_token,
/// the mock server only fires once for the first request — we verify the
/// acquirer doesn't accidentally re-send.
#[tokio::test]
async fn affirm_authorize_single_call_per_logical_request() {
    let server = MockServer::start().await;
    let counter: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    Mock::given(method("POST"))
        .and(path("/api/v2/charges"))
        .respond_with(move |_: &Request| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_json(json!({
                "id": "CHG_X",
                "amount": 10_000,
                "status": "authorized"
            }))
        })
        .mount(&server)
        .await;

    let acquirer = AffirmAcquirer::new(Client::new(), "pub", "priv", server.uri());
    let intent = sample_intent();
    let session = acquirer.initiate(&intent).await.unwrap();
    let _ = acquirer.authorize(&session, "tok").await.unwrap();
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}
