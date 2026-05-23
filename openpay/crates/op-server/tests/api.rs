//! HTTP integration tests via `tower::ServiceExt::oneshot` — no
//! sockets, no listener, just direct router invocation. Tests the
//! full handler stack: JSON parsing → store → JSON out.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use op_server::{AppState, router};
use op_webhook::{HttpTransport, MockTransport};
use serde_json::{Value, json};
use tower::util::ServiceExt;

async fn body_to_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("collect body").to_bytes();
    serde_json::from_slice(&bytes).expect("body is json")
}

fn json_request(method: &str, path: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn empty_request(method: &str, path: &str) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .body(Body::empty())
        .unwrap()
}

#[tokio::test]
async fn health_returns_ok() {
    let app = router(AppState::new_in_memory());
    let res = app.oneshot(empty_request("GET", "/health")).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn readiness_returns_counts() {
    let app = router(AppState::new_in_memory());
    let res = app
        .oneshot(empty_request("GET", "/readiness"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["status"], "ready");
    assert_eq!(body["refunds"], 0);
}

#[tokio::test]
async fn create_refund_round_trip() {
    let app = router(AppState::new_in_memory());
    let tx_id = uuid::Uuid::now_v7();
    let req = json!({
        "original_tx_id": tx_id.to_string(),
        "amount_minor": 7500,
        "currency": "USD",
        "reason": "customer_request",
        "external_id": "ext-1",
        "requested_at_unix_secs": 1_700_000_000_u64,
    });
    let res = app
        .clone()
        .oneshot(json_request("POST", "/v1/refunds", req))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["amount_minor"], 7500);
    assert_eq!(body["currency"], "USD");
    assert_eq!(body["status"], "requested");
    let refund_id = body["id"].as_str().unwrap();

    let res = app
        .oneshot(empty_request("GET", &format!("/v1/refunds/{refund_id}")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["external_id"], "ext-1");
}

#[tokio::test]
async fn refund_unknown_returns_404() {
    let app = router(AppState::new_in_memory());
    let missing = uuid::Uuid::now_v7();
    let res = app
        .oneshot(empty_request("GET", &format!("/v1/refunds/{missing}")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["code"], "not_found");
}

#[tokio::test]
async fn refund_idempotency_replay_returns_same() {
    let app = router(AppState::new_in_memory());
    let tx_id = uuid::Uuid::now_v7();
    let req = json!({
        "original_tx_id": tx_id.to_string(),
        "amount_minor": 100,
        "currency": "USD",
        "reason": "duplicate_charge",
        "external_id": "ext-idem",
        "requested_at_unix_secs": 1_700_000_000_u64,
    });
    let res1 = app
        .clone()
        .oneshot(json_request("POST", "/v1/refunds", req.clone()))
        .await
        .unwrap();
    assert_eq!(res1.status(), StatusCode::OK);
    let body1 = body_to_json(res1.into_body()).await;

    let res2 = app
        .oneshot(json_request("POST", "/v1/refunds", req))
        .await
        .unwrap();
    assert_eq!(res2.status(), StatusCode::OK);
    let body2 = body_to_json(res2.into_body()).await;
    assert_eq!(body1["id"], body2["id"]);
}

#[tokio::test]
async fn create_dispute_round_trip() {
    let app = router(AppState::new_in_memory());
    let tx_id = uuid::Uuid::now_v7();
    let req = json!({
        "original_tx_id": tx_id.to_string(),
        "amount_minor": 9999,
        "currency": "USD",
        "reason": "fraudulent",
        "network_reason_code": "10.4",
        "external_id": "disp-1",
        "opened_at_unix_secs": 1_700_000_000_u64,
    });
    let res = app
        .oneshot(json_request("POST", "/v1/disputes", req))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["status"], "chargeback");
    assert_eq!(body["network_reason_code"], "10.4");
}

#[tokio::test]
async fn open_batch_and_close() {
    let app = router(AppState::new_in_memory());
    let req = json!({
        "currency": "USD",
        "rail": "ach_nacha",
        "external_id": "batch-1",
        "opened_at_unix_secs": 1_700_000_000_u64,
    });
    let res = app
        .clone()
        .oneshot(json_request("POST", "/v1/settlement/batches", req))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    let bid = body["id"].as_str().unwrap();
    assert_eq!(body["status"], "open");

    // Add an entry.
    let tx_id = uuid::Uuid::now_v7();
    let entry = json!({
        "tx_id": tx_id.to_string(),
        "amount_minor": 12345,
    });
    let res = app
        .clone()
        .oneshot(json_request(
            "POST",
            &format!("/v1/settlement/batches/{bid}/entries"),
            entry,
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["entry_count"], 1);
    assert_eq!(body["gross_minor"], 12345);

    // Close with 1% reserve.
    let close = json!({
        "flat_rate_bps": 100,
        "max_total_bps": 10000,
        "dispute_adjustment_bps": 0,
        "closed_at_unix_secs": 1_700_001_000_u64,
    });
    let res = app
        .oneshot(json_request(
            "POST",
            &format!("/v1/settlement/batches/{bid}/close"),
            close,
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["status"], "closed");
    assert_eq!(body["reserve_minor"], 123); // 1% of 12345
}

#[tokio::test]
async fn audit_report_empty_window() {
    let app = router(AppState::new_in_memory());
    let res = app
        .oneshot(empty_request(
            "GET",
            "/v1/audit/report?start_tx=0&end_tx=100&generated_at_unix_secs=9999",
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert!(body["entries"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn subscription_create_pause_resume_cancel() {
    let app = router(AppState::new_in_memory());
    let req = json!({
        "customer_ref": "cust-1",
        "plan_name": "Pro Monthly",
        "amount_minor": 4_900,
        "currency": "USD",
        "interval": "month",
        "interval_count": 1,
        "trial_days": 14,
        "method": { "type": "vault", "token": "tok_v7_test" },
        "external_id": "sub-1",
        "now_unix_secs": 1_700_000_000_u64,
    });
    let res = app
        .clone()
        .oneshot(json_request("POST", "/v1/subscriptions", req))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["status"], "trialing");
    assert_eq!(body["plan_name"], "Pro Monthly");
    let sid = body["id"].as_str().unwrap().to_owned();

    // Pause.
    let res = app
        .clone()
        .oneshot(json_request(
            "POST",
            &format!("/v1/subscriptions/{sid}/pause"),
            json!({ "now_unix_secs": 1_700_000_100_u64 }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["status"], "paused");

    // Resume.
    let res = app
        .clone()
        .oneshot(json_request(
            "POST",
            &format!("/v1/subscriptions/{sid}/resume"),
            json!({}),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["status"], "active");

    // List by customer.
    let res = app
        .clone()
        .oneshot(empty_request(
            "GET",
            "/v1/subscriptions?customer_ref=cust-1",
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body.as_array().unwrap().len(), 1);

    // Cancel immediately.
    let res = app
        .oneshot(json_request(
            "POST",
            &format!("/v1/subscriptions/{sid}/cancel"),
            json!({ "at_period_end": false, "now_unix_secs": 1_700_000_200_u64 }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["status"], "canceled");
}

#[tokio::test]
async fn fx_quote_returns_configured_rate() {
    use op_core::Currency;
    use op_fx::StaticQuoteProvider;
    let provider: Arc<dyn op_fx::QuoteProvider> =
        Arc::new(StaticQuoteProvider::new().with_rate(Currency::EUR, Currency::USD, 1_082_500));
    let state = AppState::new_in_memory().with_fx_provider(provider);
    let app = router(state);

    let res = app
        .oneshot(empty_request("GET", "/v1/fx/quote?from=EUR&to=USD"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["from"], "EUR");
    assert_eq!(body["to"], "USD");
    assert_eq!(body["rate_ppm"], 1_082_500);
    assert_eq!(body["source"], "static");
}

#[tokio::test]
async fn fx_convert_applies_rate() {
    use op_core::Currency;
    use op_fx::StaticQuoteProvider;
    let provider: Arc<dyn op_fx::QuoteProvider> =
        Arc::new(StaticQuoteProvider::new().with_rate(Currency::EUR, Currency::USD, 1_082_500));
    let state = AppState::new_in_memory().with_fx_provider(provider);
    let app = router(state);

    let res = app
        .oneshot(json_request(
            "POST",
            "/v1/fx/convert",
            json!({
                "from": "EUR",
                "to": "USD",
                "amount_minor": 10_000,
            }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["source_amount_minor"], 10_000);
    assert_eq!(body["target_amount_minor"], 10_825);
    assert_eq!(body["rounding"], "half_even");
}

#[tokio::test]
async fn fx_quote_for_unknown_pair_returns_404() {
    let state = AppState::new_in_memory();
    let app = router(state);
    let res = app
        .oneshot(empty_request("GET", "/v1/fx/quote?from=USD&to=JPY"))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["code"], "not_found");
}

#[tokio::test]
async fn refund_creation_fires_webhook_to_subscribed_endpoint() {
    let mock = Arc::new(MockTransport::new());
    mock.push_ok();
    let transport: Arc<dyn HttpTransport> = mock.clone();
    let state = AppState::new_in_memory().with_webhook_transport(transport);
    let app = router(state);

    // Register a webhook endpoint subscribed to refund.created.
    let res = app
        .clone()
        .oneshot(json_request(
            "POST",
            "/v1/webhooks/endpoints",
            json!({
                "url": "https://example.test/refund-hook",
                "secret": "shared-secret-xyz",
                "event_filters": ["refund.created"],
            }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["status"], "active");

    // Create a refund.
    let tx_id = uuid::Uuid::now_v7();
    let res = app
        .oneshot(json_request(
            "POST",
            "/v1/refunds",
            json!({
                "original_tx_id": tx_id.to_string(),
                "amount_minor": 250,
                "currency": "USD",
                "reason": "customer_request",
                "external_id": "ext-webhook",
                "requested_at_unix_secs": 1_700_000_000_u64,
            }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // The mock transport should have seen exactly one outbound delivery.
    assert_eq!(mock.captured_count(), 1);
    let captured = mock.take_captured();
    assert_eq!(captured[0].url, "https://example.test/refund-hook");
    // Payload is the refund response JSON; verify a field is present.
    let payload_str = std::str::from_utf8(&captured[0].body).unwrap();
    assert!(payload_str.contains("\"external_id\":\"ext-webhook\""));
    assert!(payload_str.contains("\"amount_minor\":250"));
}

#[tokio::test]
async fn refund_creation_skips_disabled_endpoint() {
    let mock = Arc::new(MockTransport::new());
    let transport: Arc<dyn HttpTransport> = mock.clone();
    let state = AppState::new_in_memory().with_webhook_transport(transport);
    let app = router(state);

    let res = app
        .clone()
        .oneshot(json_request(
            "POST",
            "/v1/webhooks/endpoints",
            json!({
                "url": "https://example.test/disabled-hook",
                "secret": "s",
                "event_filters": ["refund.created"],
            }),
        ))
        .await
        .unwrap();
    let body = body_to_json(res.into_body()).await;
    let endpoint_id = body["id"].as_str().unwrap().to_owned();

    // Disable it.
    let _ = app
        .clone()
        .oneshot(empty_request(
            "POST",
            &format!("/v1/webhooks/endpoints/{endpoint_id}/disable"),
        ))
        .await
        .unwrap();

    let tx_id = uuid::Uuid::now_v7();
    let _ = app
        .oneshot(json_request(
            "POST",
            "/v1/refunds",
            json!({
                "original_tx_id": tx_id.to_string(),
                "amount_minor": 100,
                "currency": "USD",
                "reason": "duplicate_charge",
                "external_id": "ext-disabled",
                "requested_at_unix_secs": 1_700_000_000_u64,
            }),
        ))
        .await
        .unwrap();

    assert_eq!(
        mock.captured_count(),
        0,
        "disabled endpoint should not receive deliveries"
    );
}

#[tokio::test]
async fn intent_create_then_resume_after_challenge() {
    use op_core::{Currency, Money, PaymentMethod, VaultRef};
    use op_driver_sdk::DeterministicCardAcquirer;
    use op_orchestrator::{CardAdapter, Orchestrator, PolicyRouter};
    use op_rails_card::acquirer::AuthStatus;

    // Configure the acquirer so the test idempotency key triggers
    // a challenge.
    let acquirer = Arc::new(DeterministicCardAcquirer::new().with_key_override(
        "intent-3ds-1",
        AuthStatus::RequiresCustomerAction,
        None,
    ));
    let card = Arc::new(CardAdapter::new("det-card", acquirer));
    let mut orch = Orchestrator::new().with_router(Box::new(PolicyRouter::new(
        vec!["det-card".to_owned()],
        vec![],
    )));
    orch.register_adapter(card);

    let state = AppState::new_in_memory().with_orchestrator(orch);
    let app = router(state);

    // Step 1: create intent. Expect requires_customer_action.
    let res = app
        .clone()
        .oneshot(json_request(
            "POST",
            "/v1/intents",
            json!({
                "idempotency_key": "intent-3ds-1",
                "amount_minor": 1_500,
                "currency": "USD",
                "method": { "type": "vault", "token": "tok_v7_3ds" },
            }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["terminal_status"], "requires_customer_action");
    let psp_id = body["psp_payment_id"].as_str().unwrap().to_owned();
    let _ = Currency::USD;
    let _ = Money::from_minor(0, Currency::USD);
    let _ = PaymentMethod::Vault(VaultRef::new("x"));

    // Step 2: resume after challenge. Expect approved.
    let res = app
        .oneshot(json_request(
            "POST",
            "/v1/intents/resume",
            json!({
                "idempotency_key": "intent-3ds-1",
                "amount_minor": 1_500,
                "currency": "USD",
                "method": { "type": "vault", "token": "tok_v7_3ds" },
                "rail": "card",
                "driver": "det-card",
                "psp_payment_id": psp_id,
            }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_to_json(res.into_body()).await;
    assert_eq!(body["terminal_status"], "approved");
    assert_eq!(body["psp_payment_id"], psp_id);
}

#[tokio::test]
async fn bad_currency_returns_400() {
    let app = router(AppState::new_in_memory());
    let tx_id = uuid::Uuid::now_v7();
    let req = json!({
        "original_tx_id": tx_id.to_string(),
        "amount_minor": 100,
        "currency": "usd", // lowercase
        "reason": "customer_request",
        "requested_at_unix_secs": 1_700_000_000_u64,
    });
    let res = app
        .oneshot(json_request("POST", "/v1/refunds", req))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}
