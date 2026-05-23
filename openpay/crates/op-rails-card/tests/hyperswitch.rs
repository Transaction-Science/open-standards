//! Integration tests for the Hyperswitch driver.
//!
//! Uses `httpmock` to spin up a fake Hyperswitch server in-process. The
//! mock returns canonical response shapes verified from the Hyperswitch
//! V1 API reference. This exercises the full path:
//!
//!   `CardAcquirer::authorize` → JSON serialize → HTTP POST → mock
//!   → JSON parse → `AuthDecision`
//!
//! No real network. Run with `cargo test -p op-rails-card`.
//!
//! For tests against the real sandbox, gate them behind the
//! `live-sandbox` feature: `cargo test -p op-rails-card --features live-sandbox`.

use httpmock::prelude::*;
use op_core::{Currency, Money, PaymentMethod, Token, VaultRef};
use op_rails_card::{
    CardAcquirer,
    acquirer::{
        AuthRequest, AuthStatus, CaptureRequest, RefundRequest, ThreeDsMode, VoidReason,
        VoidRequest,
    },
    error::Error,
    hyperswitch::HyperswitchClient,
};

fn client_for(server: &MockServer) -> HyperswitchClient {
    HyperswitchClient::new(server.base_url(), "sk_test_dummy_key")
}

fn vault_method() -> PaymentMethod {
    PaymentMethod::Vault(VaultRef::new("tok_test_abcdef"))
}

// ---------------------------------------------------------------------------
// authorize() — success paths
// ---------------------------------------------------------------------------

#[test]
fn authorize_auto_capture_returns_settled() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/payments")
            .header("api-key", "sk_test_dummy_key")
            .header("Content-Type", "application/json");
        then.status(200)
            .header("content-type", "application/json")
            .body(
                r#"{
                "payment_id": "pay_auto_001",
                "status": "succeeded",
                "amount": 6540,
                "amount_capturable": 0,
                "amount_received": 6540,
                "currency": "USD",
                "connector": "stripe"
            }"#,
            );
    });

    let client = client_for(&server);
    let req = AuthRequest {
        amount: Money::from_minor(6540, Currency::USD),
        method: vault_method(),
        auto_capture: true,
        idempotency_key: "idem_001".into(),
        three_ds: Some(ThreeDsMode::Skip),
        metadata: None,
    };

    let decision = client.authorize(&req).expect("auth should succeed");
    mock.assert();

    assert_eq!(decision.psp_payment_id, "pay_auto_001");
    assert_eq!(decision.status, AuthStatus::Settled);
    assert_eq!(decision.raw_status, "succeeded");
    assert_eq!(decision.authorized_amount.unwrap().minor_units, 6540);
    assert!(decision.error_code.is_none());
}

#[test]
fn authorize_manual_capture_returns_authorized_awaiting_capture() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/payments");
        then.status(200).body(
            r#"{
                "payment_id": "pay_manual_001",
                "status": "requires_capture",
                "amount": 1000,
                "amount_capturable": 1000,
                "currency": "USD",
                "connector": "stripe"
            }"#,
        );
    });

    let client = client_for(&server);
    let req = AuthRequest {
        amount: Money::from_minor(1000, Currency::USD),
        method: vault_method(),
        auto_capture: false,
        idempotency_key: "idem_002".into(),
        three_ds: None,
        metadata: None,
    };

    let decision = client.authorize(&req).unwrap();
    assert_eq!(decision.status, AuthStatus::AuthorizedAwaitingCapture);
    assert_eq!(decision.raw_status, "requires_capture");
}

#[test]
fn authorize_with_3ds_redirect_returns_requires_customer_action() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/payments");
        then.status(200).body(
            r#"{
                "payment_id": "pay_3ds_001",
                "status": "requires_customer_action",
                "amount": 5000,
                "amount_capturable": 5000,
                "currency": "USD",
                "next_action": {
                    "type": "redirect_to_url",
                    "redirect_to_url": "https://hooks.stripe.com/3d_secure/xyz"
                }
            }"#,
        );
    });

    let client = client_for(&server);
    let req = AuthRequest {
        amount: Money::from_minor(5000, Currency::USD),
        method: vault_method(),
        auto_capture: true,
        idempotency_key: "idem_003".into(),
        three_ds: Some(ThreeDsMode::Required),
        metadata: None,
    };

    let decision = client.authorize(&req).unwrap();
    assert_eq!(decision.status, AuthStatus::RequiresCustomerAction);
    assert_eq!(
        decision.redirect_url.as_deref(),
        Some("https://hooks.stripe.com/3d_secure/xyz")
    );
}

// ---------------------------------------------------------------------------
// authorize() — failure paths
// ---------------------------------------------------------------------------

#[test]
fn authorize_unsupported_method_returns_error_without_network() {
    let server = MockServer::start();
    // No expectations — we should NOT hit the network for an A2A method.
    let client = client_for(&server);
    let req = AuthRequest {
        amount: Money::from_minor(100, Currency::USD),
        method: PaymentMethod::A2a(op_core::A2aKey::UsAch {
            routing: "021000021".into(),
            account: "1234".into(),
        }),
        auto_capture: true,
        idempotency_key: "idem_004".into(),
        three_ds: None,
        metadata: None,
    };
    assert!(matches!(
        client.authorize(&req),
        Err(Error::UnsupportedMethod)
    ));
}

#[test]
fn authorize_psp_4xx_maps_to_psp_rejected() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/payments");
        then.status(400).body(
            r#"{
            "error": {
                "type": "invalid_request",
                "message": "amount must be > 0",
                "code": "IR_05"
            }
        }"#,
        );
    });

    let client = client_for(&server);
    let req = AuthRequest {
        amount: Money::from_minor(0, Currency::USD),
        method: vault_method(),
        auto_capture: true,
        idempotency_key: "idem_005".into(),
        three_ds: None,
        metadata: None,
    };

    match client.authorize(&req) {
        Err(Error::PspRejected {
            status,
            code,
            message,
        }) => {
            assert_eq!(status, 400);
            assert_eq!(code, "IR_05");
            assert!(message.contains("amount must be > 0"));
        }
        other => panic!("expected PspRejected, got {other:?}"),
    }
}

#[test]
fn authorize_unknown_status_returns_unknown_status_error() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/payments");
        then.status(200).body(
            r#"{
            "payment_id": "pay_unknown_001",
            "status": "totally_made_up",
            "amount": 100,
            "amount_capturable": 100,
            "currency": "USD"
        }"#,
        );
    });

    let client = client_for(&server);
    let req = AuthRequest {
        amount: Money::from_minor(100, Currency::USD),
        method: vault_method(),
        auto_capture: true,
        idempotency_key: "idem_006".into(),
        three_ds: None,
        metadata: None,
    };

    match client.authorize(&req) {
        Err(Error::UnknownStatus(s)) => assert_eq!(s, "totally_made_up"),
        other => panic!("expected UnknownStatus, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// capture()
// ---------------------------------------------------------------------------

#[test]
fn capture_calls_correct_endpoint_with_amount() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/payments/pay_manual_001/capture")
            .json_body_obj(&serde_json::json!({"amount_to_capture": 500}));
        then.status(200).body(
            r#"{
            "payment_id": "pay_manual_001",
            "status": "partially_captured_and_capturable",
            "amount": 1000,
            "amount_capturable": 500,
            "amount_received": 500,
            "currency": "USD"
        }"#,
        );
    });

    let client = client_for(&server);
    let req = CaptureRequest {
        psp_payment_id: "pay_manual_001".into(),
        amount: Money::from_minor(500, Currency::USD),
        idempotency_key: "idem_cap_001".into(),
    };

    let decision = client.capture(&req).unwrap();
    mock.assert();
    assert_eq!(decision.status, AuthStatus::AuthorizedAwaitingCapture);
    assert_eq!(decision.psp_payment_id, "pay_manual_001");
}

#[test]
fn capture_full_transitions_to_settled() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/payments/pay_manual_002/capture");
        then.status(200).body(
            r#"{
            "payment_id": "pay_manual_002",
            "status": "succeeded",
            "amount": 1000,
            "amount_capturable": 0,
            "amount_received": 1000,
            "currency": "USD"
        }"#,
        );
    });

    let client = client_for(&server);
    let req = CaptureRequest {
        psp_payment_id: "pay_manual_002".into(),
        amount: Money::from_minor(1000, Currency::USD),
        idempotency_key: "idem_cap_002".into(),
    };
    let decision = client.capture(&req).unwrap();
    assert_eq!(decision.status, AuthStatus::Settled);
}

// ---------------------------------------------------------------------------
// void()
// ---------------------------------------------------------------------------

#[test]
fn void_calls_cancel_endpoint_with_reason() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/payments/pay_void_001/cancel")
            .json_body_obj(&serde_json::json!({
                "cancellation_reason": "requested_by_customer"
            }));
        then.status(200).body(
            r#"{
            "payment_id": "pay_void_001",
            "status": "cancelled",
            "amount": 1000,
            "amount_capturable": 0,
            "currency": "USD"
        }"#,
        );
    });

    let client = client_for(&server);
    let req = VoidRequest {
        psp_payment_id: "pay_void_001".into(),
        reason: VoidReason::RequestedByCustomer,
        idempotency_key: "idem_void_001".into(),
    };
    let decision = client.void(&req).unwrap();
    mock.assert();
    assert_eq!(decision.status, AuthStatus::HardDecline); // cancelled => HardDecline
    assert_eq!(decision.raw_status, "cancelled");
}

#[test]
fn void_fraudulent_reason_serialized() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/payments/pay_x/cancel")
            .json_body_obj(&serde_json::json!({"cancellation_reason": "fraudulent"}));
        then.status(200).body(r#"{
            "payment_id":"pay_x","status":"cancelled","amount":1,"amount_capturable":0,"currency":"USD"
        }"#);
    });

    let client = client_for(&server);
    let req = VoidRequest {
        psp_payment_id: "pay_x".into(),
        reason: VoidReason::Fraudulent,
        idempotency_key: "idem".into(),
    };
    client.void(&req).unwrap();
    mock.assert();
}

// ---------------------------------------------------------------------------
// refund()
// ---------------------------------------------------------------------------

#[test]
fn refund_posts_to_refunds_endpoint() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/refunds")
            .json_body_obj(&serde_json::json!({
                "payment_id": "pay_settled_001",
                "amount": 300,
                "reason": "customer dissatisfied",
                "merchant_refund_id": "idem_ref_001"
            }));
        then.status(200).body(
            r#"{
            "refund_id": "ref_test_001",
            "payment_id": "pay_settled_001",
            "status": "succeeded",
            "amount": 300,
            "currency": "USD"
        }"#,
        );
    });

    let client = client_for(&server);
    let req = RefundRequest {
        psp_payment_id: "pay_settled_001".into(),
        amount: Money::from_minor(300, Currency::USD),
        reason: Some("customer dissatisfied".into()),
        idempotency_key: "idem_ref_001".into(),
    };
    let decision = client.refund(&req).unwrap();
    mock.assert();
    assert_eq!(decision.psp_payment_id, "pay_settled_001");
    assert_eq!(decision.status, AuthStatus::Settled);
    assert_eq!(decision.authorized_amount.unwrap().minor_units, 300);
}

#[test]
fn refund_pending_status_is_transient() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/refunds");
        then.status(200).body(
            r#"{
            "refund_id": "ref_pend_001",
            "payment_id": "pay_y",
            "status": "pending",
            "amount": 100,
            "currency": "USD"
        }"#,
        );
    });

    let client = client_for(&server);
    let req = RefundRequest {
        psp_payment_id: "pay_y".into(),
        amount: Money::from_minor(100, Currency::USD),
        reason: None,
        idempotency_key: "i".into(),
    };
    let decision = client.refund(&req).unwrap();
    assert_eq!(decision.status, AuthStatus::Transient);
}

// ---------------------------------------------------------------------------
// EMV blob forwarding
// ---------------------------------------------------------------------------

#[test]
fn emv_payload_forwarded_in_connector_metadata() {
    let server = MockServer::start();
    let emv_blob = [0x9Fu8, 0x02, 0x06, 0, 0, 0, 0, 0x01, 0x00];
    let expected_hex = "9f0206000000000100";
    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/payments")
            .json_body_partial(format!(
                r#"{{"connector_metadata": {{"emv_tlv_hex": "{expected_hex}"}}}}"#
            ));
        then.status(200).body(
            r#"{
            "payment_id": "pay_emv_001",
            "status": "succeeded",
            "amount": 100,
            "amount_capturable": 0,
            "amount_received": 100,
            "currency": "USD"
        }"#,
        );
    });

    let client = client_for(&server);
    let req = AuthRequest {
        amount: Money::from_minor(100, Currency::USD),
        method: PaymentMethod::Emv(Token::new(emv_blob.to_vec())),
        auto_capture: true,
        idempotency_key: "idem_emv_001".into(),
        three_ds: None,
        metadata: None,
    };
    let decision = client.authorize(&req).unwrap();
    mock.assert();
    assert_eq!(decision.status, AuthStatus::Settled);
}
