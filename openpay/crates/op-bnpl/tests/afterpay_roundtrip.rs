//! Afterpay driver round-trip: initiate → authorize → capture → refund.

use std::collections::BTreeMap;

use op_bnpl::{
    AfterpayAcquirer, AfterpayRegion, BillingInfo, BnplAcquirer, BnplIntent, ConsumerInfo,
    IdempotencyKey, LineItem, RedirectUrls, ShippingInfo,
};
use op_core::{Currency, Money};
use reqwest::Client;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn intent() -> BnplIntent {
    BnplIntent {
        amount: Money::from_minor(8_000, Currency::USD),
        currency: Currency::USD,
        line_items: vec![LineItem {
            name: "Lamp".into(),
            sku: None,
            quantity: 1,
            unit_price: Money::from_minor(8_000, Currency::USD),
            total_amount: Money::from_minor(8_000, Currency::USD),
        }],
        shipping: ShippingInfo {
            name: "C".into(),
            line1: "1".into(),
            line2: None,
            city: "NY".into(),
            region: "NY".into(),
            postal_code: "10001".into(),
            country: "US".into(),
        },
        billing: BillingInfo {
            name: "C".into(),
            line1: "1".into(),
            line2: None,
            city: "NY".into(),
            region: "NY".into(),
            postal_code: "10001".into(),
            country: "US".into(),
        },
        consumer: ConsumerInfo {
            email: "c@d.com".into(),
            phone: None,
            given_name: Some("C".into()),
            family_name: Some("D".into()),
            date_of_birth: Some("1990-01-01".into()),
        },
        idempotency_key: IdempotencyKey::from("idem-after-1"),
        redirect_urls: RedirectUrls {
            success: "https://m/ok".into(),
            cancel: "https://m/cancel".into(),
            failure: None,
        },
        metadata: BTreeMap::new(),
    }
}

#[tokio::test]
async fn afterpay_full_roundtrip() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v2/checkouts"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "token": "CHK_TOKEN_1",
            "redirectCheckoutUrl": "https://portal.afterpay.com/uss/checkout?token=CHK_TOKEN_1"
        })))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v2/payments/auth"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "PAY_1",
            "amount": { "amount": "80.00", "currency": "USD" },
            "status": "AUTH_APPROVED"
        })))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v2/payments/PAY_1/capture"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "PAY_1",
            "amount": { "amount": "80.00", "currency": "USD" },
            "status": "CAPTURED"
        })))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v2/payments/PAY_1/refund"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "refundId": "REF_1",
            "amount": { "amount": "30.00", "currency": "USD" }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let acquirer = AfterpayAcquirer::with_base_url(
        Client::new(),
        "merchant",
        "secret",
        AfterpayRegion::Us,
        server.uri(),
    );

    let session = acquirer.initiate(&intent()).await.unwrap();
    assert_eq!(session.provider_ref, "CHK_TOKEN_1");
    assert!(session.redirect_url.is_some());

    let auth = acquirer.authorize(&session, "CHK_TOKEN_1").await.unwrap();
    assert_eq!(auth.provider_ref, "PAY_1");
    assert_eq!(auth.authorized_amount.minor_units, 8_000);

    let captured = acquirer.capture(&auth, None).await.unwrap();
    assert_eq!(captured.amount.minor_units, 8_000);

    let refund = acquirer
        .refund(&captured, Money::from_minor(3_000, Currency::USD))
        .await
        .unwrap();
    assert_eq!(refund.refund_ref, "REF_1");
    assert_eq!(refund.amount.minor_units, 3_000);
}

#[tokio::test]
async fn afterpay_void_succeeds_on_200() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/payments/PAY_X/void"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&server)
        .await;

    let acquirer = AfterpayAcquirer::with_base_url(
        Client::new(),
        "m",
        "k",
        AfterpayRegion::Us,
        server.uri(),
    );
    let auth = op_bnpl::AuthorizedCheckout {
        provider: op_bnpl::BnplProvider::AfterpayClearpay,
        provider_ref: "PAY_X".into(),
        authorized_amount: Money::from_minor(8_000, Currency::USD),
        plan: op_bnpl::InstalmentPlan::new(
            4,
            Money::from_minor(2_000, Currency::USD),
            chrono::Utc::now(),
            op_bnpl::InstalmentInterval::Biweekly,
        ),
    };
    acquirer.void(&auth).await.unwrap();
}
