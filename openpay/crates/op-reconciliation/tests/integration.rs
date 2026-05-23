//! Integration tests for `op-reconciliation`.
//!
//! These drive the fully-deterministic webhook path end-to-end: build
//! settlement [`WebhookEvent`]s, build the ledger window, run the
//! [`Reconciler`], and assert the [`ReconciliationReport`]. The
//! webhook source is chosen because it needs no ISO 20022 fixtures —
//! the CAMT flattening it shares a matcher with is unit-tested in
//! `op-iso20022` and `matcher.rs`.

use op_core::{Currency, Money};
use op_ledger::{AccountId, Entry, LedgerId, Transaction};
use op_reconciliation::sources::{SETTLEMENT_EVENT_TYPE, WebhookEventSource};
use op_reconciliation::{Discrepancy, Reconciler};
use op_webhook::WebhookEvent;

/// A `psp.settlement.confirmed` event with the reference payload shape.
fn settlement(
    source_id: &str,
    external_id: Option<&str>,
    minor: i64,
    currency: &str,
    direction: &str,
    at: u64,
) -> WebhookEvent {
    let mut body = serde_json::json!({
        "source_id": source_id,
        "amount_minor": minor,
        "currency": currency,
        "direction": direction,
        "posted_at_unix_secs": at,
    });
    if let Some(e) = external_id {
        body["external_id"] = serde_json::Value::String(e.to_owned());
    }
    WebhookEvent::new(
        SETTLEMENT_EVENT_TYPE,
        serde_json::to_vec(&body).unwrap(),
        at,
    )
}

fn posted(ext: &str, minor: i64, at: u64) -> Transaction {
    Transaction::new_posted(
        LedgerId::new(),
        at,
        vec![
            Entry::debit(AccountId::new(), Money::from_minor(minor, Currency::USD)),
            Entry::credit(AccountId::new(), Money::from_minor(minor, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id(ext)
}

fn pending(ext: &str, minor: i64, at: u64) -> Transaction {
    Transaction::new_pending(
        LedgerId::new(),
        at,
        vec![
            Entry::debit(AccountId::new(), Money::from_minor(minor, Currency::USD)),
            Entry::credit(AccountId::new(), Money::from_minor(minor, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id(ext)
}

#[test]
fn clean_run_when_everything_agrees() {
    let events = vec![
        settlement("psp-1", Some("ORD-1"), 5_00, "USD", "credit", 1_000),
        settlement("psp-2", Some("ORD-2"), 12_34, "USD", "credit", 1_100),
    ];
    let txs = vec![posted("ORD-1", 5_00, 1_000), posted("ORD-2", 12_34, 1_100)];

    let report = Reconciler::new(0, 10_000)
        .unwrap()
        .reconcile(&WebhookEventSource::new(&events), &txs)
        .unwrap();

    assert_eq!(report.matched, 2);
    assert_eq!(report.reconciled(), 2);
    assert!(report.is_clean());
}

#[test]
fn detects_every_discrepancy_class() {
    let events = vec![
        settlement("psp-amt", Some("ORD-AMT"), 999, "USD", "credit", 1_000),
        settlement("psp-sts", Some("ORD-STS"), 500, "USD", "credit", 1_000),
        settlement(
            "psp-orphan",
            Some("ORD-MISSING"),
            700,
            "USD",
            "credit",
            1_000,
        ),
    ];
    let txs = vec![
        posted("ORD-AMT", 1000, 1_000),       // amount differs (999 vs 1000)
        pending("ORD-STS", 500, 1_000),       // amount ok, status Pending
        posted("ORD-LEDGER-ONLY", 42, 1_000), // no statement line
    ];

    let report = Reconciler::new(0, 10_000)
        .unwrap()
        .reconcile(&WebhookEventSource::new(&events), &txs)
        .unwrap();

    assert_eq!(report.matched, 0);
    let kinds: Vec<&str> = report
        .discrepancies
        .iter()
        .map(|d| d.task_descriptor().kind)
        .collect();
    assert!(kinds.contains(&"amount_mismatch"));
    assert!(kinds.contains(&"status_mismatch"));
    assert!(kinds.contains(&"unmatched_statement")); // ORD-MISSING
    assert!(kinds.contains(&"unmatched_ledger")); // ORD-LEDGER-ONLY
}

#[test]
fn non_settlement_events_are_ignored() {
    // A stream full of unrelated webhooks must not produce lines.
    let noise = vec![
        WebhookEvent::new("payment.authorized", b"{}".to_vec(), 1),
        WebhookEvent::new("ledger.transaction.posted", b"{}".to_vec(), 2),
    ];
    let report = Reconciler::new(0, 10_000)
        .unwrap()
        .reconcile(&WebhookEventSource::new(&noise), &[])
        .unwrap();
    assert!(report.is_clean());
    assert_eq!(report.reconciled(), 0);
}

#[test]
fn malformed_settlement_payload_aborts_the_run() {
    // Right event_type, wrong body: must error, not silently drop the
    // line (which would under-report discrepancies).
    let bad = vec![WebhookEvent::new(
        SETTLEMENT_EVENT_TYPE,
        b"{\"not\":\"a settlement\"}".to_vec(),
        1,
    )];
    let err = Reconciler::new(0, 10_000)
        .unwrap()
        .reconcile(&WebhookEventSource::new(&bad), &[])
        .unwrap_err();
    assert!(matches!(
        err,
        op_reconciliation::Error::UnrecognizedWebhook(_)
    ));
}

#[test]
fn report_round_trips_through_json() {
    let events = vec![settlement(
        "psp-x",
        Some("ORD-X"),
        1,
        "USD",
        "credit",
        1_000,
    )];
    let txs = vec![pending("ORD-X", 1, 1_000)];
    let report = Reconciler::new(0, 10_000)
        .unwrap()
        .reconcile(&WebhookEventSource::new(&events), &txs)
        .unwrap();
    let json = serde_json::to_string(&report).unwrap();
    let back: op_reconciliation::ReconciliationReport = serde_json::from_str(&json).unwrap();
    assert_eq!(report, back);
    assert!(matches!(
        back.discrepancies.as_slice(),
        [Discrepancy::StatusMismatch { .. }]
    ));
}
