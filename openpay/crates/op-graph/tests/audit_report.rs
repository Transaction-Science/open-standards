//! Audit report end-to-end test.
//!
//! Walks the production-shape flow:
//!   - Orchestrator runs an intent through a driver (records
//!     `rail_attempt` with `external_id_hint`).
//!   - Operator posts a ledger transaction with that same key.
//!   - Reconciliation finds a discrepancy and records a task.
//!   - `AuditReport::for_window` joins all three back into one
//!     row per ledger tx.

use std::sync::Arc;

use op_core::{Currency, Money, PaymentMethod, RailKind, VaultRef};
use op_ledger::{Account, AccountClass, Entry, Ledger, LedgerStore, Transaction};
use op_orchestrator::engine::{AdapterResult, RailAdapter};
use op_orchestrator::{
    AttemptOutcome, IdempotencyKey, Orchestrator, PaymentIntent, PolicyRouter, RailTelemetry,
    RoutingSignals,
};
use op_reconciliation::{Discrepancy, ReconciliationReport, ReconciliationStore};

use op_graph::{
    AuditReport, GraphHandle, GraphLedgerStore, GraphRailTelemetry, GraphReconciliationStore,
};

const DRIVER: &str = "audited_psp";

#[derive(Clone)]
struct AlwaysApproveAdapter;

impl RailAdapter for AlwaysApproveAdapter {
    fn rail(&self) -> RailKind {
        RailKind::Card
    }
    fn driver(&self) -> &str {
        DRIVER
    }
    fn attempt(&self, _: &PaymentIntent, _: usize) -> AdapterResult {
        AdapterResult {
            outcome: AttemptOutcome::Success,
            psp_payment_id: Some("psp-1".into()),
            uetr: None,
        }
    }
}

#[test]
fn audit_report_joins_attempt_ledger_and_reconciliation() {
    let handle = GraphHandle::new_in_memory();
    let telemetry: Arc<GraphRailTelemetry> =
        Arc::new(GraphRailTelemetry::with_handle(handle.clone()));
    let ledger = GraphLedgerStore::with_handle(handle.clone());
    let recon = GraphReconciliationStore::with_handle(handle.clone());

    let router = PolicyRouter::new(vec![DRIVER.to_owned()], vec![])
        .with_signals(telemetry.clone() as Arc<dyn RoutingSignals>);
    let mut orch = Orchestrator::new()
        .with_router(Box::new(router))
        .with_telemetry(telemetry.clone() as Arc<dyn RailTelemetry>);
    orch.register_adapter(Arc::new(AlwaysApproveAdapter));

    // Set up books.
    let l = Ledger::new("Acme").unwrap();
    let lid = l.id;
    ledger.create_ledger(l).unwrap();
    let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
    let rev = Account::new(lid, "rev", AccountClass::Revenue, Currency::USD);
    let cid = cash.id;
    let rid = rev.id;
    ledger.create_account(cash).unwrap();
    ledger.create_account(rev).unwrap();

    // 1. Orchestrator runs an intent — records the rail_attempt.
    let intent_key = "audit-intent-1";
    let intent = PaymentIntent::new(
        IdempotencyKey::new(intent_key),
        Money::from_minor(7_500, Currency::USD),
        PaymentMethod::Vault(VaultRef::new("tok_v7_x")),
    );
    let outcome = orch.run(&intent).unwrap();
    assert_eq!(outcome.attempts.len(), 1);

    // Snapshot tx_count before posting the ledger tx so the
    // window can include only this one tx.
    let start = handle.tx_count();

    // 2. Operator posts the ledger tx with the same idempotency
    //    key as external_id.
    let tx = Transaction::new_posted(
        lid,
        1_000,
        vec![
            Entry::debit(cid, Money::from_minor(7_500, Currency::USD)),
            Entry::credit(rid, Money::from_minor(7_500, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id(intent_key);
    let tx_id = ledger.post_transaction(tx).unwrap();
    let end = handle.tx_count();

    // 3. Reconciliation finds something off about it.
    let report = ReconciliationReport {
        window: (0, 10_000),
        matched: 0,
        fuzzy_matched: 0,
        discrepancies: vec![Discrepancy::UnmatchedLedger {
            tx_id,
            external_id: Some(intent_key.into()),
        }],
        matched_pairs: Vec::new(),
    };
    recon.record_report(&report).unwrap();

    // 4. Audit report for the window covering this tx joins
    //    everything.
    let report = AuditReport::for_window(&handle, start, end, 9_999).unwrap();
    assert_eq!(report.entries.len(), 1);
    let entry = &report.entries[0];
    assert_eq!(entry.tx_id, tx_id);
    assert_eq!(entry.external_id.as_deref(), Some(intent_key));
    assert_eq!(entry.status, "posted");
    assert_eq!(entry.settled_amount_minor, Some(7_500));
    assert_eq!(entry.currency_code.as_deref(), Some("USD"));
    assert_eq!(entry.rail.as_deref(), Some("Card"));
    assert_eq!(entry.driver.as_deref(), Some(DRIVER));
    assert_eq!(entry.reconciliation_task_ids.len(), 1);
    assert!(report.generated_at_unix_secs == 9_999);
}

#[test]
fn empty_window_yields_empty_report() {
    let handle = GraphHandle::new_in_memory();
    let r = AuditReport::for_window(&handle, 0, 100, 0).unwrap();
    assert!(r.entries.is_empty());
}

#[test]
fn inverted_window_yields_empty_report() {
    let handle = GraphHandle::new_in_memory();
    let r = AuditReport::for_window(&handle, 100, 50, 0).unwrap();
    assert!(r.entries.is_empty());
}

#[test]
fn compact_returns_ok_on_in_memory_handle() {
    // The compaction API is a no-op for the in-memory backend
    // (Minigraf returns Ok). Verify it doesn't error.
    let handle = GraphHandle::new_in_memory();
    handle.compact().unwrap();
}
