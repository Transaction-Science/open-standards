//! Integration tests for `op-graph`.
//!
//! These exercise the typical operator pattern: a ledger
//! transaction is posted to a `GraphLedgerStore`, a webhook event
//! is emitted with the same payload that the orchestrator would
//! emit, and the resulting attempts are persisted to a
//! `GraphWebhookStore`. Both stores share a single `GraphHandle`
//! so the cross-domain queries work over one graph.

use std::sync::Arc;

use op_core::{Currency, Money};
use op_ledger::{Account, AccountClass, Direction, Entry, Ledger, LedgerStore, Transaction};
use op_webhook::retry::ExponentialBackoffPolicy;
use op_webhook::{
    DeliveryAttempt, Endpoint, EndpointStatus, HttpTransport, MockTransport, RetryPolicy,
    WebhookDispatcher, WebhookEvent, WebhookStore,
};

use op_graph::graph::etypes;
use op_graph::{
    GraphHandle, GraphLedgerStore, GraphReconciliationStore, GraphWebhookStore,
    accounts_touched_by_transaction, attempts_for_event, reversal_chain,
    transactions_touching_account,
};
use op_reconciliation::sources::{SETTLEMENT_EVENT_TYPE, WebhookEventSource};
use op_reconciliation::{Reconciler, ReconciliationStore};

// ============================================================
// Test 1: Ledger transaction round-trips through the graph store
// ============================================================

#[test]
fn graph_ledger_store_round_trips_a_transaction() {
    let store = GraphLedgerStore::new_in_memory();
    let l = Ledger::new("Acme").unwrap();
    let lid = l.id;
    store.create_ledger(l).unwrap();
    let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
    let rev = Account::new(lid, "rev", AccountClass::Revenue, Currency::USD);
    let cid = cash.id;
    let rid = rev.id;
    store.create_account(cash).unwrap();
    store.create_account(rev).unwrap();

    let t = Transaction::new_pending(
        lid,
        1_700_000_000,
        vec![
            Entry::debit(cid, Money::from_minor(525, Currency::USD)),
            Entry::credit(rid, Money::from_minor(525, Currency::USD)),
        ],
    )
    .unwrap();
    let tid = store.post_transaction(t).unwrap();
    let recovered = store.get_transaction(tid).unwrap();
    assert_eq!(recovered.entries.len(), 2);
    store.mark_posted(tid).unwrap();
    let bal = store.balance(cid).unwrap();
    assert_eq!(bal.posted.minor_units, 525);
}

// ============================================================
// Test 2: GraphWebhookStore round-trips an event + attempts
// ============================================================

#[test]
fn graph_webhook_store_round_trips_event_and_attempts() {
    let store = GraphWebhookStore::new_in_memory();
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
    let recovered = store.get_attempt(aid).unwrap();
    assert_eq!(recovered.event_id, evid);
    assert_eq!(recovered.endpoint_id, eid);
}

// ============================================================
// Test 3: `accounts_touched_by_transaction` returns both sides
// ============================================================

#[test]
fn accounts_touched_returns_both_sides_for_balanced_tx() {
    let store = GraphLedgerStore::new_in_memory();
    let l = Ledger::new("L").unwrap();
    let lid = l.id;
    store.create_ledger(l).unwrap();
    let a = Account::new(lid, "a", AccountClass::Asset, Currency::USD);
    let b = Account::new(lid, "b", AccountClass::Revenue, Currency::USD);
    let aid = a.id;
    let bid = b.id;
    store.create_account(a).unwrap();
    store.create_account(b).unwrap();

    let t = Transaction::new_pending(
        lid,
        0,
        vec![
            Entry::debit(aid, Money::from_minor(7, Currency::USD)),
            Entry::credit(bid, Money::from_minor(7, Currency::USD)),
        ],
    )
    .unwrap();
    let tid = store.post_transaction(t).unwrap();
    let touches = accounts_touched_by_transaction(store.handle(), tid).unwrap();
    assert_eq!(touches.len(), 2);
    assert!(
        touches
            .iter()
            .any(|t| t.account_id == aid && t.direction == Direction::Debit)
    );
    assert!(
        touches
            .iter()
            .any(|t| t.account_id == bid && t.direction == Direction::Credit)
    );
}

// ============================================================
// Test 4: Reversal chain returns full lineage
// ============================================================

#[test]
fn reversal_chain_returns_full_lineage() {
    let store = GraphLedgerStore::new_in_memory();
    let l = Ledger::new("L").unwrap();
    let lid = l.id;
    store.create_ledger(l).unwrap();
    let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
    let rev = Account::new(lid, "rev", AccountClass::Revenue, Currency::USD);
    let cid = cash.id;
    let rid = rev.id;
    store.create_account(cash).unwrap();
    store.create_account(rev).unwrap();

    // Original.
    let t1 = Transaction::new_pending(
        lid,
        0,
        vec![
            Entry::debit(cid, Money::from_minor(100, Currency::USD)),
            Entry::credit(rid, Money::from_minor(100, Currency::USD)),
        ],
    )
    .unwrap();
    let t1_id = store.post_transaction(t1).unwrap();
    // Reversal #1.
    let t2 = Transaction::new_pending(
        lid,
        0,
        vec![
            Entry::credit(cid, Money::from_minor(100, Currency::USD)),
            Entry::debit(rid, Money::from_minor(100, Currency::USD)),
        ],
    )
    .unwrap();
    let t2_id = store.post_transaction(t2).unwrap();
    store
        .handle()
        .create_edge(
            t2_id.as_uuid(),
            op_graph::graph::etypes::LEDGER_REVERSES,
            t1_id.as_uuid(),
        )
        .unwrap();
    // Re-post (re-correct).
    let t3 = Transaction::new_pending(
        lid,
        0,
        vec![
            Entry::debit(cid, Money::from_minor(100, Currency::USD)),
            Entry::credit(rid, Money::from_minor(100, Currency::USD)),
        ],
    )
    .unwrap();
    let t3_id = store.post_transaction(t3).unwrap();
    store
        .handle()
        .create_edge(
            t3_id.as_uuid(),
            op_graph::graph::etypes::LEDGER_REVERSES,
            t2_id.as_uuid(),
        )
        .unwrap();
    // Chain from the middle: should yield [t1, t2, t3].
    let chain = reversal_chain(store.handle(), t2_id).unwrap();
    assert_eq!(chain, vec![t1_id, t2_id, t3_id]);
}

// ============================================================
// Test 5: Cross-currency transaction produces per-currency edges
// ============================================================

#[test]
fn cross_currency_transaction_records_currency_per_edge() {
    // Two ledgers' worth of accounts can't share a tx (we enforce
    // single-ledger), but a single tx with mixed-currency entries
    // within one ledger is legal as long as debits balance credits
    // per currency. Test the round-trip.
    let store = GraphLedgerStore::new_in_memory();
    let l = Ledger::new("FX").unwrap();
    let lid = l.id;
    store.create_ledger(l).unwrap();
    let usd_cash = Account::new(lid, "usd_cash", AccountClass::Asset, Currency::USD);
    let usd_fee = Account::new(lid, "usd_fee", AccountClass::Expense, Currency::USD);
    let eur_cash = Account::new(lid, "eur_cash", AccountClass::Asset, Currency::EUR);
    let eur_fee = Account::new(lid, "eur_fee", AccountClass::Expense, Currency::EUR);
    let uc = usd_cash.id;
    let uf = usd_fee.id;
    let ec = eur_cash.id;
    let ef = eur_fee.id;
    store.create_account(usd_cash).unwrap();
    store.create_account(usd_fee).unwrap();
    store.create_account(eur_cash).unwrap();
    store.create_account(eur_fee).unwrap();
    let t = Transaction::new_pending(
        lid,
        0,
        vec![
            Entry::debit(uf, Money::from_minor(10, Currency::USD)),
            Entry::credit(uc, Money::from_minor(10, Currency::USD)),
            Entry::debit(ef, Money::from_minor(8, Currency::EUR)),
            Entry::credit(ec, Money::from_minor(8, Currency::EUR)),
        ],
    )
    .unwrap();
    let tid = store.post_transaction(t).unwrap();
    let touches = accounts_touched_by_transaction(store.handle(), tid).unwrap();
    assert_eq!(touches.len(), 4);
    let usd_count = touches.iter().filter(|t| t.currency_code == "USD").count();
    let eur_count = touches.iter().filter(|t| t.currency_code == "EUR").count();
    assert_eq!(usd_count, 2);
    assert_eq!(eur_count, 2);
}

// ============================================================
// Test 6: Idempotency at the graph layer
// ============================================================

#[test]
fn idempotency_returns_same_id_for_repeated_external_id() {
    let store = GraphLedgerStore::new_in_memory();
    let l = Ledger::new("L").unwrap();
    let lid = l.id;
    store.create_ledger(l).unwrap();
    let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
    let rev = Account::new(lid, "rev", AccountClass::Revenue, Currency::USD);
    let cid = cash.id;
    let rid = rev.id;
    store.create_account(cash).unwrap();
    store.create_account(rev).unwrap();
    let body = vec![
        Entry::debit(cid, Money::from_minor(42, Currency::USD)),
        Entry::credit(rid, Money::from_minor(42, Currency::USD)),
    ];
    let t1 = Transaction::new_pending(lid, 0, body.clone())
        .unwrap()
        .with_external_id("idem-1");
    let t2 = Transaction::new_pending(lid, 0, body)
        .unwrap()
        .with_external_id("idem-1");
    let id1 = store.post_transaction(t1).unwrap();
    let id2 = store.post_transaction(t2).unwrap();
    assert_eq!(id1, id2);
}

// ============================================================
// Test 7: Auto-disable flow over a graph-backed webhook store
// ============================================================

#[test]
fn auto_disable_flow_works_on_graph_webhook_store() {
    let store: Arc<GraphWebhookStore> = Arc::new(GraphWebhookStore::new_in_memory());
    let transport: Arc<MockTransport> = Arc::new(MockTransport::new());
    let policy: Arc<dyn RetryPolicy> = Arc::new(ExponentialBackoffPolicy::deterministic(
        2,
        60,
        72 * 3600,
        3,
        0,
    ));
    let dispatcher = WebhookDispatcher::new(
        store.clone() as Arc<dyn WebhookStore>,
        transport.clone() as Arc<dyn HttpTransport>,
        policy,
    )
    .with_clock(|| 1000);
    let endpoint = Endpoint::new(
        "https://flaky.example/h",
        b"whsec".to_vec(),
        vec!["*".to_string()],
    )
    .unwrap();
    let eid = endpoint.id;
    store.put_endpoint(endpoint).unwrap();
    // Threshold 3 → 3 consecutive failures should auto-disable.
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
    let ep = store.get_endpoint(eid).unwrap();
    assert_eq!(ep.status, EndpointStatus::AutoDisabled);
}

// ============================================================
// Test 8: Process_due_retries works on graph backend
// ============================================================

#[test]
fn process_due_retries_works_with_graph_store() {
    let store: Arc<GraphWebhookStore> = Arc::new(GraphWebhookStore::new_in_memory());
    let transport: Arc<MockTransport> = Arc::new(MockTransport::new());
    let policy: Arc<dyn RetryPolicy> = Arc::new(ExponentialBackoffPolicy::deterministic(
        2,
        60,
        72 * 3600,
        10,
        0,
    ));
    let dispatcher_early = WebhookDispatcher::new(
        store.clone() as Arc<dyn WebhookStore>,
        transport.clone() as Arc<dyn HttpTransport>,
        policy.clone(),
    )
    .with_clock(|| 1000);
    let endpoint = Endpoint::new(
        "https://x.example/h",
        b"whsec".to_vec(),
        vec!["*".to_string()],
    )
    .unwrap();
    store.put_endpoint(endpoint).unwrap();
    // First attempt fails → RetryScheduled.
    transport.push_5xx(503);
    let _ = dispatcher_early
        .dispatch(WebhookEvent::new("any", b"x".to_vec(), 1000))
        .unwrap();
    // Later dispatcher processes due retries; queue a 200.
    let dispatcher_late = WebhookDispatcher::new(
        store.clone() as Arc<dyn WebhookStore>,
        transport.clone() as Arc<dyn HttpTransport>,
        policy,
    )
    .with_clock(|| 99_999);
    transport.push_ok();
    let outcomes = dispatcher_late.process_due_retries().unwrap();
    assert_eq!(outcomes.len(), 1);
}

// ============================================================
// Test 9: Shared GraphHandle — one graph, two stores, cross queries
// ============================================================

#[test]
fn shared_graph_enables_cross_domain_queries() {
    let handle = GraphHandle::new_in_memory();
    let ledger = GraphLedgerStore::with_handle(handle.clone());
    let webhooks = GraphWebhookStore::with_handle(handle.clone());

    // Post a ledger tx.
    let l = Ledger::new("L").unwrap();
    let lid = l.id;
    ledger.create_ledger(l).unwrap();
    let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
    let rev = Account::new(lid, "rev", AccountClass::Revenue, Currency::USD);
    let cid = cash.id;
    let rid = rev.id;
    ledger.create_account(cash).unwrap();
    ledger.create_account(rev).unwrap();
    let t = Transaction::new_pending(
        lid,
        0,
        vec![
            Entry::debit(cid, Money::from_minor(15, Currency::USD)),
            Entry::credit(rid, Money::from_minor(15, Currency::USD)),
        ],
    )
    .unwrap();
    let tid = ledger.post_transaction(t).unwrap();

    // Emit a webhook event mirroring the tx, with a delivery attempt.
    let endpoint =
        Endpoint::new("https://x.example/h", b"s".to_vec(), vec!["*".to_string()]).unwrap();
    let endpoint_id = endpoint.id;
    webhooks.put_endpoint(endpoint).unwrap();
    let event = WebhookEvent::new(
        "ledger.tx.posted",
        format!("{{\"tx_id\":\"{tid}\"}}").into_bytes(),
        0,
    );
    let event_id = event.id;
    webhooks.put_event(event).unwrap();
    let a = DeliveryAttempt::new_pending(event_id, endpoint_id, 0, 0);
    webhooks.put_attempt(a).unwrap();

    // Cross-domain query: the graph holds BOTH the ledger tx and
    // the webhook event. Each query honors its own slice.
    let touches = accounts_touched_by_transaction(&handle, tid).unwrap();
    assert_eq!(touches.len(), 2);
    let attempts = attempts_for_event(&handle, event_id).unwrap();
    assert_eq!(attempts.len(), 1);
    let txs = transactions_touching_account(&handle, cid).unwrap();
    assert_eq!(txs, vec![tid]);
}

// ============================================================
// Test 10: Vertex/edge counts grow as expected
// ============================================================

#[test]
fn vertex_and_edge_counts_track_with_inserts() {
    let store = GraphLedgerStore::new_in_memory();
    let l = Ledger::new("L").unwrap();
    let lid = l.id;
    store.create_ledger(l).unwrap();
    assert_eq!(store.handle().vertex_count().unwrap(), 1);
    assert_eq!(store.handle().edge_count().unwrap(), 0);
    let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
    let rev = Account::new(lid, "rev", AccountClass::Revenue, Currency::USD);
    let cid = cash.id;
    let rid = rev.id;
    store.create_account(cash).unwrap();
    store.create_account(rev).unwrap();
    // 3 vertices (ledger + 2 accounts); 2 ledger_in_ledger edges.
    assert_eq!(store.handle().vertex_count().unwrap(), 3);
    assert_eq!(store.handle().edge_count().unwrap(), 2);
    let t = Transaction::new_pending(
        lid,
        0,
        vec![
            Entry::debit(cid, Money::from_minor(1, Currency::USD)),
            Entry::credit(rid, Money::from_minor(1, Currency::USD)),
        ],
    )
    .unwrap();
    store.post_transaction(t).unwrap();
    // +1 vertex (tx), +1 ledger_in_ledger edge (tx → ledger), +1
    // debit, +1 credit = 5 edges total.
    assert_eq!(store.handle().vertex_count().unwrap(), 4);
    assert_eq!(store.handle().edge_count().unwrap(), 5);
}

// ============================================================
// Test 11: Reconciliation tasks land in the SAME graph as the
//          ledger txs they're about — and re-recording is idempotent
// ============================================================

#[test]
fn reconciliation_tasks_wire_into_the_shared_ledger_graph() {
    let handle = GraphHandle::new_in_memory();
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

    // ORD-1: posted & will match a settlement line exactly (clean).
    let t1 = Transaction::new_posted(
        lid,
        1_000,
        vec![
            Entry::debit(cid, Money::from_minor(500, Currency::USD)),
            Entry::credit(rid, Money::from_minor(500, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id("ORD-1");
    let _t1id = ledger.post_transaction(t1).unwrap();

    // ORD-2: still PENDING but the bank says it settled → StatusMismatch.
    let t2 = Transaction::new_pending(
        lid,
        1_000,
        vec![
            Entry::debit(cid, Money::from_minor(900, Currency::USD)),
            Entry::credit(rid, Money::from_minor(900, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id("ORD-2");
    let t2id = ledger.post_transaction(t2).unwrap();

    // Reconcile the webhook stream against the stored ledger window.
    let events = vec![
        WebhookEvent::new(
            SETTLEMENT_EVENT_TYPE,
            serde_json::to_vec(&serde_json::json!({
                "source_id": "psp-1", "external_id": "ORD-1",
                "amount_minor": 500, "currency": "USD",
                "direction": "credit", "posted_at_unix_secs": 1_000
            }))
            .unwrap(),
            1_000,
        ),
        WebhookEvent::new(
            SETTLEMENT_EVENT_TYPE,
            serde_json::to_vec(&serde_json::json!({
                "source_id": "psp-2", "external_id": "ORD-2",
                "amount_minor": 900, "currency": "USD",
                "direction": "credit", "posted_at_unix_secs": 1_000
            }))
            .unwrap(),
            1_000,
        ),
    ];
    let stored = vec![
        ledger.get_transaction(_t1id).unwrap(),
        ledger.get_transaction(t2id).unwrap(),
    ];
    let report = Reconciler::new(0, 10_000)
        .unwrap()
        .reconcile(&WebhookEventSource::new(&events), &stored)
        .unwrap();
    assert_eq!(report.matched, 1); // ORD-1
    assert_eq!(report.discrepancies.len(), 1); // ORD-2 status mismatch

    // Persist into the SAME graph.
    let ids = recon.record_report(&report).unwrap();
    assert_eq!(ids.len(), 1);

    let tasks = recon.list_tasks().unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].kind, "status_mismatch");

    // The task vertex is wired to the ACTUAL ledger tx vertex (t2)
    // living in the shared graph: there is an inbound task_about edge
    // on t2's vertex.
    let about = handle.in_edges(t2id.0, etypes::TASK_ABOUT).unwrap();
    assert_eq!(about.len(), 1);

    // Re-recording the identical report is a no-op: no duplicate task.
    let ids2 = recon.record_report(&report).unwrap();
    assert_eq!(ids2, ids);
    assert_eq!(recon.list_tasks().unwrap().len(), 1);
}
