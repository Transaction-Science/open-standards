//! Phase 19: reconciliation-density routing signal.
//!
//! Closes a longer loop than Phase 18:
//!
//! ```text
//! Orchestrator records rail_attempt[external_id_hint = idempotency_key]
//!   ↓
//! Operator posts a ledger_tx with the same external_id
//!   ↓
//! Reconciliation finds a discrepancy on that tx and records a task
//!   ↓
//! Router queries `discrepancy_score(rail, driver)` → 1 task /
//!   1 attempt = 1.0 — pushes the driver back even though every
//!   HTTP attempt against that driver returned OK.
//! ```
//!
//! This is the "noisy reconciliation" case Phase 18 explicitly
//! deferred and Phase 19 ships.

use std::sync::Arc;

use op_core::{Currency, Money, PaymentMethod, RailKind, VaultRef};
use op_ledger::{Account, AccountClass, Entry, Ledger, LedgerStore, Transaction, TransactionId};
use op_orchestrator::engine::{AdapterResult, RailAdapter};
use op_orchestrator::{
    AttemptOutcome, IdempotencyKey, Orchestrator, PaymentIntent, PolicyRouter, RailTelemetry,
    RoutingSignals,
};
use op_reconciliation::{Discrepancy, ReconciliationReport, ReconciliationStore};

use op_graph::{GraphHandle, GraphLedgerStore, GraphRailTelemetry, GraphReconciliationStore};

const DRIVER_DIRTY: &str = "dirty_psp";
const DRIVER_CLEAN: &str = "clean_psp";

#[derive(Clone)]
struct AlwaysApproveAdapter {
    driver: &'static str,
}

impl RailAdapter for AlwaysApproveAdapter {
    fn rail(&self) -> RailKind {
        RailKind::Card
    }
    fn driver(&self) -> &str {
        self.driver
    }
    fn attempt(&self, _: &PaymentIntent, _: usize) -> AdapterResult {
        AdapterResult {
            outcome: AttemptOutcome::Success,
            psp_payment_id: Some(format!("{}_psp_id", self.driver)),
            uetr: None,
        }
    }
}

fn vault_intent(key: &str) -> PaymentIntent {
    PaymentIntent::new(
        IdempotencyKey::new(key),
        Money::from_minor(5_000, Currency::USD),
        PaymentMethod::Vault(VaultRef::new("tok_v7_test")),
    )
}

/// Operator-side helper: after an attempt succeeded on `(rail,
/// driver)` for `intent_key`, post a balanced ledger transaction
/// keyed on the same idempotency key. This is the join key
/// `discrepancy_score` walks.
fn post_ledger_tx_for_intent(
    ledger: &GraphLedgerStore,
    lid: op_ledger::LedgerId,
    cash: op_ledger::AccountId,
    rev: op_ledger::AccountId,
    intent_key: &str,
    amount_minor: i64,
    effective_at: u64,
) -> TransactionId {
    let tx = Transaction::new_posted(
        lid,
        effective_at,
        vec![
            Entry::debit(cash, Money::from_minor(amount_minor, Currency::USD)),
            Entry::credit(rev, Money::from_minor(amount_minor, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id(intent_key);
    ledger.post_transaction(tx).unwrap()
}

#[test]
fn reconciliation_discrepancy_pushes_driver_back_even_when_http_is_clean() {
    // One shared graph: telemetry, ledger, and reconciliation all
    // sit in the same fact store, so the join works.
    let handle = GraphHandle::new_in_memory();
    let telemetry: Arc<GraphRailTelemetry> =
        Arc::new(GraphRailTelemetry::with_handle(handle.clone()));
    let ledger = GraphLedgerStore::with_handle(handle.clone());
    let recon = GraphReconciliationStore::with_handle(handle.clone());

    // Standard chart of accounts.
    let l = Ledger::new("Acme").unwrap();
    let lid = l.id;
    ledger.create_ledger(l).unwrap();
    let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
    let rev = Account::new(lid, "rev", AccountClass::Revenue, Currency::USD);
    let cash_id = cash.id;
    let rev_id = rev.id;
    ledger.create_account(cash).unwrap();
    ledger.create_account(rev).unwrap();

    // The router lists `dirty_psp` FIRST. Both adapters always
    // approve at the HTTP level — the only signal that can re-order
    // them is the reconciliation discrepancy we'll plant.
    let router = PolicyRouter::new(
        vec![DRIVER_DIRTY.to_owned(), DRIVER_CLEAN.to_owned()],
        vec![],
    )
    .with_signals(telemetry.clone() as Arc<dyn RoutingSignals>);
    let mut orch = Orchestrator::new()
        .with_router(Box::new(router))
        .with_telemetry(telemetry.clone() as Arc<dyn RailTelemetry>);
    orch.register_adapter(Arc::new(AlwaysApproveAdapter {
        driver: DRIVER_DIRTY,
    }));
    orch.register_adapter(Arc::new(AlwaysApproveAdapter {
        driver: DRIVER_CLEAN,
    }));

    // First intent: static order [dirty, clean]; dirty succeeds at
    // HTTP on first try. One attempt, recorded against dirty.
    let first = orch.run(&vault_intent("intent-1")).unwrap();
    assert_eq!(first.attempts.len(), 1);
    assert_eq!(first.attempts[0].driver, DRIVER_DIRTY);

    // Operator posts the ledger transaction with the same
    // idempotency key as its external_id. This is the natural
    // production pattern; the telemetry's external_id_hint matches.
    let tx_id = post_ledger_tx_for_intent(&ledger, lid, cash_id, rev_id, "intent-1", 5_000, 1_000);
    let _ = (rev_id, tx_id);

    // Reconciliation finds a discrepancy on that ledger tx. We
    // record one task pointing at tx_id via the trait — that builds
    // the task_about edge GraphRailTelemetry walks.
    let report = ReconciliationReport {
        window: (0, 10_000),
        matched: 0,
        fuzzy_matched: 0,
        discrepancies: vec![Discrepancy::UnmatchedLedger {
            tx_id,
            external_id: Some("intent-1".into()),
        }],
        matched_pairs: Vec::new(),
    };
    recon.record_report(&report).unwrap();

    // Discrepancy_score = 1 task / 1 attempt for dirty; 0 for
    // clean (no attempts). With the combined-score router,
    // dirty now ranks BEHIND clean.
    let dirty_score = telemetry.discrepancy_score_at(RailKind::Card, DRIVER_DIRTY, 2_000);
    let clean_score = telemetry.discrepancy_score_at(RailKind::Card, DRIVER_CLEAN, 2_000);
    assert!(
        dirty_score > 0.0,
        "expected dirty score > 0, got {dirty_score}"
    );
    assert_eq!(clean_score, 0.0);

    // Second intent: combined score pushes dirty back, clean is
    // tried first and succeeds on the first attempt.
    let second = orch.run(&vault_intent("intent-2")).unwrap();
    assert_eq!(
        second.attempts.len(),
        1,
        "expected one attempt on the clean driver, got {:?}",
        second.attempts
    );
    assert_eq!(second.attempts[0].driver, DRIVER_CLEAN);
}

#[test]
fn empty_reconciliation_state_yields_zero_discrepancy_score() {
    let telemetry = GraphRailTelemetry::new_in_memory();
    telemetry.record_attempt(
        RailKind::Card,
        DRIVER_DIRTY,
        op_orchestrator::AttemptResultClass::Approved,
        1_000,
        Some("intent-zero"),
        None,
    );
    // No ledger_tx with that external_id; no reconciliation tasks.
    let score = telemetry.discrepancy_score_at(RailKind::Card, DRIVER_DIRTY, 1_100);
    assert_eq!(score, 0.0);
}

#[test]
fn driver_with_no_attempts_yields_zero_discrepancy_score() {
    let telemetry = GraphRailTelemetry::new_in_memory();
    // A driver we've never seen has nothing to compute against.
    let score = telemetry.discrepancy_score_at(RailKind::Card, "ghost_driver", 1_000);
    assert_eq!(score, 0.0);
}
