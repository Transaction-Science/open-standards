//! Integration tests for `op-webhook`.
//!
//! These tests exercise the **operator-level pattern**: a ledger
//! event triggers a webhook fanout to merchant-configured endpoints,
//! with the resulting signature verifiable by the receiver using
//! standard `verify_signature` logic.
//!
//! `op-webhook` has NO compile-time dependency on `op-ledger` — the
//! integration is at the operator's layer. We build a payload that
//! looks like a ledger event but the dispatcher treats it as opaque.

use std::sync::Arc;

use op_webhook::retry::ExponentialBackoffPolicy;
use op_webhook::signing::{parse_signature_header, verify_signature};
use op_webhook::transport::MockResponse;
use op_webhook::{
    DispatchOutcome, Endpoint, EndpointStatus, HttpResponse, HttpTransport, InMemoryWebhookStore,
    MockTransport, RetryPolicy, SIGNATURE_HEADER, SignedPayload, WebhookDispatcher, WebhookEvent,
    WebhookStore,
};

/// Synthetic ledger-event payload. JSON for verisimilitude.
fn ledger_txn_posted_payload(tx_id: &str, amount: i64) -> Vec<u8> {
    format!(
        r#"{{"event":"ledger.transaction.posted","tx_id":"{tx_id}","amount_minor":{amount},"currency":"USD"}}"#,
    )
    .into_bytes()
}

fn setup_dispatcher(
    threshold: u32,
    fixed_jitter: u64,
    clock: u64,
) -> (
    Arc<InMemoryWebhookStore>,
    Arc<MockTransport>,
    WebhookDispatcher,
) {
    let store: Arc<InMemoryWebhookStore> = Arc::new(InMemoryWebhookStore::new());
    let transport: Arc<MockTransport> = Arc::new(MockTransport::new());
    let policy: Arc<dyn RetryPolicy> = Arc::new(ExponentialBackoffPolicy::deterministic(
        2,
        60,
        72 * 3600,
        threshold,
        fixed_jitter,
    ));
    let dispatcher = WebhookDispatcher::new(
        store.clone() as Arc<dyn WebhookStore>,
        transport.clone() as Arc<dyn HttpTransport>,
        policy,
    )
    .with_clock(move || clock);
    (store, transport, dispatcher)
}

// ============================================================
// Test 1: Happy-path — event delivered, signature verifies
// ============================================================

#[test]
fn fanout_happy_path_signature_verifies_on_receiver_side() {
    let (store, transport, dispatcher) = setup_dispatcher(10, 0, 1_700_000_000);
    let secret = b"whsec_merchant_acme_001".to_vec();
    let endpoint = Endpoint::new(
        "https://merchant.example.com/hooks/payments",
        secret.clone(),
        vec!["ledger.transaction.posted".into()],
    )
    .unwrap();
    let endpoint_id = endpoint.id;
    store.put_endpoint(endpoint).unwrap();

    let payload = ledger_txn_posted_payload("tx-1", 525);
    let event = WebhookEvent::new("ledger.transaction.posted", payload.clone(), 1_700_000_000);
    let outcomes = dispatcher.dispatch(event).unwrap();

    assert_eq!(outcomes.len(), 1);
    match &outcomes[0] {
        DispatchOutcome::Delivered {
            endpoint_id: eid,
            status,
        } => {
            assert_eq!(*eid, endpoint_id);
            assert_eq!(*status, 200);
        }
        o => panic!("expected Delivered, got {o:?}"),
    }

    // Simulate the merchant's signature verification.
    let captured = transport.take_captured();
    assert_eq!(captured.len(), 1);
    let req = &captured[0];
    let header = req
        .headers
        .iter()
        .find(|(k, _)| k == SIGNATURE_HEADER)
        .unwrap()
        .1
        .clone();
    // Receiver-side verification with `now` matching the dispatcher
    // clock (the dispatcher uses its own `now` as the signed
    // timestamp).
    verify_signature(&secret, &req.body, &header, 1_700_000_000, 300).unwrap();
}

// ============================================================
// Test 2: Tampering — modified body fails verification
// ============================================================

#[test]
fn modified_body_fails_signature_verification() {
    let (store, transport, dispatcher) = setup_dispatcher(10, 0, 1_700_000_000);
    let secret = b"whsec_test".to_vec();
    let endpoint =
        Endpoint::new("https://x.example/h", secret.clone(), vec!["*".to_string()]).unwrap();
    store.put_endpoint(endpoint).unwrap();
    let _ = dispatcher
        .dispatch(WebhookEvent::new("any", b"ok".to_vec(), 1_700_000_000))
        .unwrap();

    let captured = transport.take_captured();
    let req = &captured[0];
    let header = req
        .headers
        .iter()
        .find(|(k, _)| k == SIGNATURE_HEADER)
        .unwrap()
        .1
        .clone();
    // Attempt verification with a tampered body.
    let tampered_body = b"tampered";
    let r = verify_signature(&secret, tampered_body, &header, 1_700_000_000, 300);
    assert!(matches!(r, Err(op_webhook::Error::SignatureMismatch)));
}

// ============================================================
// Test 3: Replay — old timestamp outside tolerance is rejected
// ============================================================

#[test]
fn old_timestamp_outside_tolerance_is_rejected() {
    let secret = b"whsec_test";
    let body = b"event-body";
    // Signed at t=1000.
    let signed = SignedPayload::new(1000, body);
    let sig = op_webhook::signing::compute_signature(secret, &signed).unwrap();
    let header = format!("t=1000,v1={sig}");
    // Receiver's now is 10000 → delta 9000 > tolerance 300.
    let r = verify_signature(secret, body, &header, 10000, 300);
    assert!(matches!(
        r,
        Err(op_webhook::Error::TimestampOutOfTolerance { .. })
    ));
}

// ============================================================
// Test 4: Auto-disable after N consecutive failures
// ============================================================

#[test]
fn endpoint_auto_disables_after_threshold_failures() {
    let (store, transport, dispatcher) = setup_dispatcher(3, 0, 1000);
    let endpoint = Endpoint::new(
        "https://flaky.example/h",
        b"whsec".to_vec(),
        vec!["*".to_string()],
    )
    .unwrap();
    let endpoint_id = endpoint.id;
    store.put_endpoint(endpoint).unwrap();

    // 3 consecutive failures — threshold is 3.
    transport.push_5xx(503);
    transport.push_5xx(503);
    transport.push_5xx(503);

    for i in 0..3 {
        let _ = dispatcher
            .dispatch(WebhookEvent::new(
                "any",
                format!("evt-{i}").into_bytes(),
                1000,
            ))
            .unwrap();
    }

    let ep = store.get_endpoint(endpoint_id).unwrap();
    assert_eq!(ep.status, EndpointStatus::AutoDisabled);

    // Future dispatches are skipped.
    let outcomes = dispatcher
        .dispatch(WebhookEvent::new("any", b"new".to_vec(), 1000))
        .unwrap();
    // The new event matches but the endpoint is auto-disabled →
    // list_active_endpoints_for filters it out → no outcomes.
    assert!(outcomes.is_empty());
}

// ============================================================
// Test 5: Fanout — multiple endpoints, mixed outcomes
// ============================================================

#[test]
fn fanout_multiple_endpoints_each_get_signed_independently() {
    let (store, transport, dispatcher) = setup_dispatcher(10, 0, 1_700_000_000);
    let secret_a = b"whsec_a".to_vec();
    let secret_b = b"whsec_b".to_vec();
    let ep_a = Endpoint::new(
        "https://a.example/h",
        secret_a.clone(),
        vec!["*".to_string()],
    )
    .unwrap();
    let ep_b = Endpoint::new(
        "https://b.example/h",
        secret_b.clone(),
        vec!["*".to_string()],
    )
    .unwrap();
    store.put_endpoint(ep_a).unwrap();
    store.put_endpoint(ep_b).unwrap();

    let _ = dispatcher
        .dispatch(WebhookEvent::new("any", b"payload".to_vec(), 1_700_000_000))
        .unwrap();
    let captured = transport.take_captured();
    assert_eq!(captured.len(), 2);

    // Each captured request carries its own signature; each
    // signature should verify only against ITS endpoint's secret.
    for req in &captured {
        let header = req
            .headers
            .iter()
            .find(|(k, _)| k == SIGNATURE_HEADER)
            .unwrap()
            .1
            .clone();
        if req.url.contains("a.example") {
            verify_signature(&secret_a, &req.body, &header, 1_700_000_000, 300).unwrap();
            // The OTHER secret must fail.
            let r = verify_signature(&secret_b, &req.body, &header, 1_700_000_000, 300);
            assert!(matches!(r, Err(op_webhook::Error::SignatureMismatch)));
        } else if req.url.contains("b.example") {
            verify_signature(&secret_b, &req.body, &header, 1_700_000_000, 300).unwrap();
        } else {
            panic!("unexpected url: {}", req.url);
        }
    }
}

// ============================================================
// Test 6: Retry happens after scheduled delay, succeeds on second try
// ============================================================

#[test]
fn retry_succeeds_via_process_due_retries() {
    let (store, transport, dispatcher) = setup_dispatcher(10, 0, 1000);
    let secret = b"whsec".to_vec();
    let endpoint = Endpoint::new("https://x.example/h", secret, vec!["*".to_string()]).unwrap();
    let endpoint_id = endpoint.id;
    store.put_endpoint(endpoint).unwrap();

    // First attempt 5xx → retry scheduled.
    transport.push_5xx(503);
    let event = WebhookEvent::new("any", b"payload".to_vec(), 1000);
    let event_id = event.id;
    let outcomes = dispatcher.dispatch(event).unwrap();
    assert!(matches!(outcomes[0], DispatchOutcome::Retrying { .. }));

    // Advance the clock. Build a NEW dispatcher with a later clock
    // but the same store + transport. (The original dispatcher's
    // clock is fixed at 1000.)
    let later_policy: Arc<dyn RetryPolicy> = Arc::new(ExponentialBackoffPolicy::deterministic(
        2,
        60,
        72 * 3600,
        10,
        0,
    ));
    let later_dispatcher = WebhookDispatcher::new(
        store.clone() as Arc<dyn WebhookStore>,
        transport.clone() as Arc<dyn HttpTransport>,
        later_policy,
    )
    .with_clock(|| 10000);

    transport.push_ok();
    let outcomes = later_dispatcher.process_due_retries().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert!(matches!(outcomes[0], DispatchOutcome::Delivered { .. }));

    // Audit trail: two attempts for this (event, endpoint) pair.
    let attempts = store.list_attempts(event_id, endpoint_id).unwrap();
    assert!(
        attempts.len() >= 2,
        "expected at least 2 attempts, got {}",
        attempts.len()
    );
}

// ============================================================
// Test 7: Replay after re-enabling an auto-disabled endpoint
// ============================================================

#[test]
fn replay_works_after_operator_reenables_endpoint() {
    let (store, transport, dispatcher) = setup_dispatcher(2, 0, 1000);
    let endpoint = Endpoint::new(
        "https://flaky.example/h",
        b"whsec".to_vec(),
        vec!["*".to_string()],
    )
    .unwrap();
    let endpoint_id = endpoint.id;
    store.put_endpoint(endpoint).unwrap();

    // Force the endpoint to auto-disable.
    transport.push_5xx(500);
    transport.push_5xx(500);
    let event = WebhookEvent::new("a", b"x".to_vec(), 1000);
    let event_id = event.id;
    let _ = dispatcher.dispatch(event).unwrap();
    let _ = dispatcher
        .dispatch(WebhookEvent::new("b", b"y".to_vec(), 1000))
        .unwrap();
    let ep = store.get_endpoint(endpoint_id).unwrap();
    assert_eq!(ep.status, EndpointStatus::AutoDisabled);

    // Operator fixed the receiver and re-enables.
    store
        .set_endpoint_status(endpoint_id, EndpointStatus::Active)
        .unwrap();

    // Replay the first event. Queue a 200 for it.
    transport.push_ok();
    let outcome = dispatcher.replay(event_id, endpoint_id).unwrap();
    assert!(matches!(outcome, DispatchOutcome::Delivered { .. }));
}

// ============================================================
// Test 8: 4xx fails fast — no retry, dead letter on first attempt
// ============================================================

#[test]
fn four_oh_four_dead_letters_without_retry() {
    let (store, transport, dispatcher) = setup_dispatcher(10, 0, 1000);
    let endpoint = Endpoint::new(
        "https://x.example/h",
        b"whsec".to_vec(),
        vec!["*".to_string()],
    )
    .unwrap();
    let endpoint_id = endpoint.id;
    store.put_endpoint(endpoint).unwrap();

    transport.push_response(MockResponse::Response(HttpResponse {
        status: 404,
        body: b"Not found".to_vec(),
    }));
    let event = WebhookEvent::new("any", b"x".to_vec(), 1000);
    let event_id = event.id;
    let outcomes = dispatcher.dispatch(event).unwrap();
    match &outcomes[0] {
        DispatchOutcome::DeadLetter {
            endpoint_id: eid,
            reason,
        } => {
            assert_eq!(*eid, endpoint_id);
            assert_eq!(reason, "http_404");
        }
        o => panic!("expected DeadLetter, got {o:?}"),
    }

    // Audit: exactly one attempt with status Failed.
    let attempts = store.list_attempts(event_id, endpoint_id).unwrap();
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].status, op_webhook::DeliveryStatus::Failed);
}

// ============================================================
// Test 9: Stripe-style header is round-trippable
// ============================================================

#[test]
fn signature_header_is_parseable_by_stripe_style_consumers() {
    // Generate one of our headers and verify it parses into the
    // structure a Stripe-compatible consumer expects.
    let secret = b"whsec_test";
    let body = b"{\"k\":\"v\"}";
    let ts = 1_700_000_000u64;
    let sig =
        op_webhook::signing::compute_signature(secret, &SignedPayload::new(ts, body)).unwrap();
    let header = op_webhook::signing::build_signature_header(ts, &sig);
    // Parse like a Stripe consumer would.
    let (parsed_ts, sigs) = parse_signature_header(&header).unwrap();
    assert_eq!(parsed_ts, ts);
    assert_eq!(sigs.len(), 1);
    assert_eq!(sigs[0].len(), 64); // SHA-256 hex
}

// ============================================================
// Test 10: Retries audit trail accumulates correctly
// ============================================================

#[test]
fn audit_trail_records_every_attempt() {
    let (store, transport, dispatcher) = setup_dispatcher(10, 0, 1000);
    let endpoint = Endpoint::new(
        "https://x.example/h",
        b"whsec".to_vec(),
        vec!["*".to_string()],
    )
    .unwrap();
    let endpoint_id = endpoint.id;
    store.put_endpoint(endpoint).unwrap();

    let event = WebhookEvent::new("any", b"x".to_vec(), 1000);
    let event_id = event.id;
    // First attempt 5xx.
    transport.push_5xx(502);
    let _ = dispatcher.dispatch(event).unwrap();

    // Process retry later — also fails.
    let later_policy: Arc<dyn RetryPolicy> = Arc::new(ExponentialBackoffPolicy::deterministic(
        2,
        60,
        72 * 3600,
        10,
        0,
    ));
    let later = WebhookDispatcher::new(
        store.clone() as Arc<dyn WebhookStore>,
        transport.clone() as Arc<dyn HttpTransport>,
        later_policy,
    )
    .with_clock(|| 10000);
    transport.push_5xx(503);
    let _ = later.process_due_retries().unwrap();

    // Process again — succeeds.
    let later2_policy: Arc<dyn RetryPolicy> = Arc::new(ExponentialBackoffPolicy::deterministic(
        2,
        60,
        72 * 3600,
        10,
        0,
    ));
    let later2 = WebhookDispatcher::new(
        store.clone() as Arc<dyn WebhookStore>,
        transport.clone() as Arc<dyn HttpTransport>,
        later2_policy,
    )
    .with_clock(|| 20000);
    transport.push_ok();
    let _ = later2.process_due_retries().unwrap();

    let attempts = store.list_attempts(event_id, endpoint_id).unwrap();
    // We expect 3 attempts at this point: original (Failed/superseded),
    // retry-1 (Failed/superseded), retry-2 (Succeeded). The first
    // two attempts will have been marked Failed by
    // process_due_retries (the supersession trick).
    assert!(
        attempts.len() >= 3,
        "expected at least 3 attempts, got {}",
        attempts.len()
    );
    // The last one should be Succeeded.
    let last = attempts.last().unwrap();
    assert_eq!(last.status, op_webhook::DeliveryStatus::Succeeded);
}

// (end of file)
