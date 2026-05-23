//! Cross-store persistence: one `.graph` file holds refund,
//! dispute, settlement, and idempotency state; reopening the file
//! recovers everything.

use op_core::{Currency, Money};
use op_dispute::reason::DisputeReason;
use op_dispute::{Dispute, DisputeStore};
use op_graph::{
    GraphDisputeStore, GraphHandle, GraphIdempotencyStore, GraphRefundStore, GraphSettlementStore,
};
use op_ledger::TransactionId;
use op_orchestrator::{
    Attempt, AttemptOutcome, IdempotencyKey, IdempotencyStore, OrchestrationOutcome, TerminalStatus,
};
use op_refund::reason::RefundReason;
use op_refund::{Refund, RefundStore};
use op_settlement::{Batch, PayoutRail, SettlementStore};
use tempfile::tempdir;

#[test]
fn refund_dispute_settlement_idempotency_all_persist_on_one_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("openpay.graph");

    // Phase 1: write through fresh handle.
    let (refund_id, dispute_id, batch_id, idem_key) = {
        let handle = GraphHandle::new_persistent(&path).unwrap();
        let refunds = GraphRefundStore::with_handle(handle.clone());
        let disputes = GraphDisputeStore::with_handle(handle.clone());
        let settlement = GraphSettlementStore::with_handle(handle.clone());
        let idempotency = GraphIdempotencyStore::with_handle(handle.clone());

        // Refund.
        let r = Refund::new(
            TransactionId::new(),
            Money::from_minor(500, Currency::USD),
            RefundReason::CustomerRequest,
            1_000,
        )
        .unwrap()
        .with_external_id("ref-persist");
        let refund_id = refunds.create_refund(r).unwrap();

        // Dispute.
        let d = Dispute::new(
            TransactionId::new(),
            Money::from_minor(700, Currency::USD),
            DisputeReason::Fraudulent,
            2_000,
        )
        .unwrap()
        .with_external_id("disp-persist");
        let dispute_id = disputes.create_dispute(d).unwrap();

        // Settlement.
        let b = Batch::open(Currency::USD, PayoutRail::AchNacha, 3_000)
            .with_external_id("batch-persist");
        let batch_id = settlement.create_batch(b).unwrap();

        // Idempotency.
        let key = IdempotencyKey::new("idem-persist");
        let prior = idempotency.reserve(&key, "sig-1");
        assert!(prior.is_none(), "first reserve should be empty");
        idempotency.commit(
            &key,
            &OrchestrationOutcome {
                terminal_status: TerminalStatus::Approved,
                attempts: vec![Attempt {
                    rail: op_core::RailKind::Card,
                    driver: "test".into(),
                    outcome: AttemptOutcome::Success,
                }],
                rail_used: Some(op_core::RailKind::Card),
                psp_payment_id: Some("psp_x".into()),
                uetr: None,
            },
        );

        handle.compact().unwrap();
        (refund_id, dispute_id, batch_id, key)
    };

    // Phase 2: drop the original handle, reopen, verify recovery.
    let handle = GraphHandle::new_persistent(&path).unwrap();
    let refunds = GraphRefundStore::with_handle(handle.clone());
    let disputes = GraphDisputeStore::with_handle(handle.clone());
    let settlement = GraphSettlementStore::with_handle(handle.clone());
    let idempotency = GraphIdempotencyStore::with_handle(handle.clone());

    let r = refunds.get_refund(refund_id).unwrap();
    assert_eq!(r.external_id.as_deref(), Some("ref-persist"));
    assert_eq!(r.amount, Money::from_minor(500, Currency::USD));

    let d = disputes.get_dispute(dispute_id).unwrap();
    assert_eq!(d.external_id.as_deref(), Some("disp-persist"));
    assert_eq!(d.reason, DisputeReason::Fraudulent);

    let b = settlement.get_batch(batch_id).unwrap();
    assert_eq!(b.external_id.as_deref(), Some("batch-persist"));
    assert_eq!(b.rail, PayoutRail::AchNacha);

    let rec = idempotency.reserve(&idem_key, "sig-1").unwrap();
    assert_eq!(rec.body_signature, "sig-1");
    let outcome = rec.outcome.expect("committed outcome should persist");
    assert_eq!(outcome.terminal_status, TerminalStatus::Approved);
    assert_eq!(outcome.psp_payment_id.as_deref(), Some("psp_x"));
}
