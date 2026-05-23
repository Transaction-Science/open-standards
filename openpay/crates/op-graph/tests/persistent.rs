//! Persistent-backend tests for `op-graph`.
//!
//! These prove that a `GraphHandle::new_persistent(path)` survives a
//! handle drop and reopen — the same single-file durability story
//! SQLite gives you, applied to property-graph data. Each test
//! opens a fresh tempdir, writes through one of the typed stores,
//! drops the handle, reopens the same path, and verifies the data
//! is still there.

use std::sync::Arc;

use op_core::{Currency, Money};
use op_ledger::{Account, AccountClass, Entry, Ledger, LedgerStore, Transaction};
use op_reconciliation::sources::{SETTLEMENT_EVENT_TYPE, WebhookEventSource};
use op_reconciliation::{Reconciler, ReconciliationStore};
use op_webhook::{DeliveryAttempt, Endpoint, WebhookEvent, WebhookStore};

use op_graph::{GraphHandle, GraphLedgerStore, GraphReconciliationStore, GraphWebhookStore};

#[test]
fn graph_handle_persists_vertex_and_edge_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("openpay.graph");

    let v_a = uuid::Uuid::new_v4();
    let v_b = uuid::Uuid::new_v4();
    let edge_id;
    {
        let h = GraphHandle::new_persistent(&path).unwrap();
        h.create_vertex(op_graph::graph::vtypes::LEDGER_ACCOUNT, v_a)
            .unwrap();
        h.create_vertex(op_graph::graph::vtypes::LEDGER_ACCOUNT, v_b)
            .unwrap();
        h.set_vertex_property(v_a, "label", serde_json::Value::String("acct-a".into()))
            .unwrap();
        let e = h
            .create_edge(v_a, op_graph::graph::etypes::LEDGER_IN_LEDGER, v_b)
            .unwrap();
        edge_id = e.id;
        // Handle drops here — file flushes.
    }

    let h2 = GraphHandle::new_persistent(&path).unwrap();
    let recovered = h2
        .get_typed_vertex(v_a, op_graph::graph::vtypes::LEDGER_ACCOUNT)
        .unwrap();
    assert_eq!(recovered.id, v_a);
    let props = h2.get_vertex_properties(v_a).unwrap();
    assert_eq!(
        props.get("label"),
        Some(&serde_json::Value::String("acct-a".into()))
    );
    let out = h2
        .out_edges(v_a, op_graph::graph::etypes::LEDGER_IN_LEDGER)
        .unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].to, v_b);
    assert_eq!(out[0].id, edge_id);
}

#[test]
fn ledger_transaction_round_trips_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ledger.graph");

    let (tid, lid, cid) = {
        let store = GraphLedgerStore::with_handle(GraphHandle::new_persistent(&path).unwrap());
        let l = Ledger::new("FY26").unwrap();
        let lid = l.id;
        store.create_ledger(l).unwrap();
        let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
        let rev = Account::new(lid, "rev", AccountClass::Revenue, Currency::USD);
        let cid = cash.id;
        let rid = rev.id;
        store.create_account(cash).unwrap();
        store.create_account(rev).unwrap();
        let t = Transaction::new_posted(
            lid,
            1_700_000_000,
            vec![
                Entry::debit(cid, Money::from_minor(525, Currency::USD)),
                Entry::credit(rid, Money::from_minor(525, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id("ORD-PERSIST-1");
        let tid = store.post_transaction(t).unwrap();
        (tid, lid, cid)
    };

    // Reopen the same file; the ledger is intact.
    let store2 = GraphLedgerStore::with_handle(GraphHandle::new_persistent(&path).unwrap());
    let recovered = store2.get_transaction(tid).unwrap();
    assert_eq!(recovered.external_id.as_deref(), Some("ORD-PERSIST-1"));
    assert_eq!(recovered.ledger_id, lid);
    // Balances are derived from the entries, which are stored as
    // edges — proving the edge graph survived.
    let bal = store2.balance(cid).unwrap();
    assert_eq!(bal.posted.minor_units, 525);
}

#[test]
fn webhook_event_and_attempt_round_trip_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("webhooks.graph");

    let (event_id, attempt_id, endpoint_id) = {
        let store = GraphWebhookStore::with_handle(GraphHandle::new_persistent(&path).unwrap());
        let endpoint = Endpoint::new(
            "https://merchant/h",
            b"whsec".to_vec(),
            vec!["*".to_string()],
        )
        .unwrap();
        let eid = endpoint.id;
        store.put_endpoint(endpoint).unwrap();
        let event = WebhookEvent::new("ledger.tx.posted", b"{\"x\":1}".to_vec(), 1_700_000_000);
        let evid = event.id;
        store.put_event(event).unwrap();
        let a = DeliveryAttempt::new_pending(evid, eid, 0, 1_700_000_001);
        let aid = a.id;
        store.put_attempt(a).unwrap();
        (evid, aid, eid)
    };

    let store2 = GraphWebhookStore::with_handle(GraphHandle::new_persistent(&path).unwrap());
    let attempt = store2.get_attempt(attempt_id).unwrap();
    assert_eq!(attempt.event_id, event_id);
    assert_eq!(attempt.endpoint_id, endpoint_id);
}

#[test]
fn reconciliation_tasks_survive_restart_and_re_record_stays_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recon.graph");

    let _shared: Arc<()> = Arc::new(());
    {
        let handle = GraphHandle::new_persistent(&path).unwrap();
        let ledger = GraphLedgerStore::with_handle(handle.clone());
        let recon = GraphReconciliationStore::with_handle(handle.clone());

        let l = Ledger::new("Acme").unwrap();
        let lid = l.id;
        ledger.create_ledger(l).unwrap();
        let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
        let rev = Account::new(lid, "rev", AccountClass::Revenue, Currency::USD);
        let cid = cash.id;
        let rid = rev.id;
        ledger.create_account(cash).unwrap();
        ledger.create_account(rev).unwrap();

        let pending = Transaction::new_pending(
            lid,
            1_000,
            vec![
                Entry::debit(cid, Money::from_minor(900, Currency::USD)),
                Entry::credit(rid, Money::from_minor(900, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id("ORD-DURABLE");
        let tid = ledger.post_transaction(pending).unwrap();
        let stored = vec![ledger.get_transaction(tid).unwrap()];

        let events = vec![WebhookEvent::new(
            SETTLEMENT_EVENT_TYPE,
            serde_json::to_vec(&serde_json::json!({
                "source_id": "psp-durable", "external_id": "ORD-DURABLE",
                "amount_minor": 900, "currency": "USD",
                "direction": "credit", "posted_at_unix_secs": 1_000
            }))
            .unwrap(),
            1_000,
        )];
        let report = Reconciler::new(0, 10_000)
            .unwrap()
            .reconcile(&WebhookEventSource::new(&events), &stored)
            .unwrap();
        let touched = recon.record_report(&report).unwrap();
        assert_eq!(touched.len(), 1);
        // Drop happens at end of scope -> file flushes.
    }

    // Reopen; the task is still there.
    let handle2 = GraphHandle::new_persistent(&path).unwrap();
    let ledger2 = GraphLedgerStore::with_handle(handle2.clone());
    let recon2 = GraphReconciliationStore::with_handle(handle2);
    let tasks = recon2.list_tasks().unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].kind, "status_mismatch");

    // Re-recording the same report after the restart is still
    // idempotent: the deterministic task_id index hits the persisted
    // vertex, so no duplicate task is created.
    let stored = ledger2
        .find_by_external_id("ORD-DURABLE")
        .unwrap()
        .map(|t| vec![t])
        .unwrap();
    let events = vec![WebhookEvent::new(
        SETTLEMENT_EVENT_TYPE,
        serde_json::to_vec(&serde_json::json!({
            "source_id": "psp-durable", "external_id": "ORD-DURABLE",
            "amount_minor": 900, "currency": "USD",
            "direction": "credit", "posted_at_unix_secs": 1_000
        }))
        .unwrap(),
        1_000,
    )];
    let report = Reconciler::new(0, 10_000)
        .unwrap()
        .reconcile(&WebhookEventSource::new(&events), &stored)
        .unwrap();
    let _ = recon2.record_report(&report).unwrap();
    assert_eq!(recon2.list_tasks().unwrap().len(), 1);
}
