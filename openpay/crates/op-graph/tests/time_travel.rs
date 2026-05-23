//! Bi-temporal time-travel tests for `LedgerHistory` on
//! `GraphLedgerStore`.
//!
//! These exercise Phase 17's core promise: the graph backend keeps
//! every prior fact, so an operator (or auditor) can ask "what did
//! the books look like at point X?" and get an answer.

use op_core::{Currency, Money};
use op_graph::{GraphHandle, GraphLedgerStore};
use op_ledger::{
    Account, AccountClass, Entry, Ledger, LedgerHistory, LedgerStore, Status, Transaction,
};

fn fixture() -> (
    GraphLedgerStore,
    GraphHandle,
    op_ledger::LedgerId,
    op_ledger::AccountId,
    op_ledger::AccountId,
) {
    let handle = GraphHandle::new_in_memory();
    let store = GraphLedgerStore::with_handle(handle.clone());
    let l = Ledger::new("Acme").unwrap();
    let lid = l.id;
    store.create_ledger(l).unwrap();
    let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
    let rev = Account::new(lid, "rev", AccountClass::Revenue, Currency::USD);
    let cid = cash.id;
    let rid = rev.id;
    store.create_account(cash).unwrap();
    store.create_account(rev).unwrap();
    (store, handle, lid, cid, rid)
}

#[test]
fn balance_as_of_before_reversal_shows_pre_reversal_state() {
    let (store, handle, lid, cid, rid) = fixture();

    // Post an initial $5.00 transaction.
    let tx = Transaction::new_posted(
        lid,
        1_000,
        vec![
            Entry::debit(cid, Money::from_minor(500, Currency::USD)),
            Entry::credit(rid, Money::from_minor(500, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id("ORD-1");
    store.post_transaction(tx).unwrap();

    // Snapshot AFTER the initial post but BEFORE the reversal.
    let snap = handle.tx_count();

    // Now post a reversing entry of $5.00 (credit cash, debit rev).
    let reversal = Transaction::new_posted(
        lid,
        2_000,
        vec![
            Entry::credit(cid, Money::from_minor(500, Currency::USD)),
            Entry::debit(rid, Money::from_minor(500, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id("ORD-1-REVERSE");
    store.post_transaction(reversal).unwrap();

    // Present-time balance on cash is zero (the reversal cancelled it).
    let now = store.balance(cid).unwrap();
    assert_eq!(now.posted.minor_units, 0);

    // Time-travel: at the snapshot, cash was up by $5 (debit-normal,
    // posted only the first tx's debit).
    let then = store.balance_as_of(cid, snap).unwrap();
    assert_eq!(then.posted.minor_units, 500);
}

#[test]
fn transaction_as_of_returns_pending_status_before_mark_posted() {
    let (store, handle, lid, cid, rid) = fixture();

    let tx = Transaction::new_pending(
        lid,
        1_000,
        vec![
            Entry::debit(cid, Money::from_minor(525, Currency::USD)),
            Entry::credit(rid, Money::from_minor(525, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id("ORD-PENDING");
    let tid = store.post_transaction(tx).unwrap();

    // Snapshot while the tx is still pending.
    let snap = handle.tx_count();

    // Then settle it.
    store.mark_posted(tid).unwrap();

    // Present: posted.
    let now = store.get_transaction(tid).unwrap();
    assert_eq!(now.status, Status::Posted);

    // Then: pending — the historical view at `snap` predates the
    // `mark_posted`.
    let then = store.transaction_as_of(tid, snap).unwrap();
    assert_eq!(then.status, Status::Pending);
    assert_eq!(then.external_id.as_deref(), Some("ORD-PENDING"));
    assert_eq!(then.entries.len(), 2);
}

#[test]
fn balance_as_of_through_pending_then_posted_transition() {
    // Pending tx contributes to `pending` balance but not `posted`.
    // After `mark_posted` it contributes to both. Snapshot in
    // between and verify that.
    let (store, handle, lid, cid, rid) = fixture();
    let tx = Transaction::new_pending(
        lid,
        1_000,
        vec![
            Entry::debit(cid, Money::from_minor(1_000, Currency::USD)),
            Entry::credit(rid, Money::from_minor(1_000, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id("ORD-FLOW");
    let tid = store.post_transaction(tx).unwrap();
    let snap_pending = handle.tx_count();
    store.mark_posted(tid).unwrap();
    let snap_posted = handle.tx_count();

    let b_pending = store.balance_as_of(cid, snap_pending).unwrap();
    assert_eq!(b_pending.posted.minor_units, 0);
    assert_eq!(b_pending.pending.minor_units, 1_000);

    let b_posted = store.balance_as_of(cid, snap_posted).unwrap();
    assert_eq!(b_posted.posted.minor_units, 1_000);
    assert_eq!(b_posted.pending.minor_units, 1_000);
}

#[test]
fn transaction_as_of_unknown_returns_not_found() {
    let (store, handle, _, _, _) = fixture();
    let snap = handle.tx_count();
    let fake = op_ledger::TransactionId::from_uuid(uuid::Uuid::new_v4());
    let err = store.transaction_as_of(fake, snap).unwrap_err();
    assert!(matches!(err, op_ledger::Error::TransactionNotFound(_)));
}

#[test]
fn time_travel_survives_persistence_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.graph");

    let (cid, snap) = {
        let handle = GraphHandle::new_persistent(&path).unwrap();
        let store = GraphLedgerStore::with_handle(handle.clone());
        let l = Ledger::new("Acme").unwrap();
        let lid = l.id;
        store.create_ledger(l).unwrap();
        let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
        let rev = Account::new(lid, "rev", AccountClass::Revenue, Currency::USD);
        let cid = cash.id;
        let rid = rev.id;
        store.create_account(cash).unwrap();
        store.create_account(rev).unwrap();

        store
            .post_transaction(
                Transaction::new_posted(
                    lid,
                    1_000,
                    vec![
                        Entry::debit(cid, Money::from_minor(400, Currency::USD)),
                        Entry::credit(rid, Money::from_minor(400, Currency::USD)),
                    ],
                )
                .unwrap()
                .with_external_id("FIRST"),
            )
            .unwrap();
        let snap = handle.tx_count();
        store
            .post_transaction(
                Transaction::new_posted(
                    lid,
                    2_000,
                    vec![
                        Entry::debit(cid, Money::from_minor(300, Currency::USD)),
                        Entry::credit(rid, Money::from_minor(300, Currency::USD)),
                    ],
                )
                .unwrap()
                .with_external_id("SECOND"),
            )
            .unwrap();
        (cid, snap)
    };

    // Reopen and time-travel: the saved snapshot still points at
    // the same moment in the persisted history.
    let store2 = GraphLedgerStore::with_handle(GraphHandle::new_persistent(&path).unwrap());
    let now = store2.balance(cid).unwrap();
    assert_eq!(now.posted.minor_units, 700);
    let then = store2.balance_as_of(cid, snap).unwrap();
    assert_eq!(then.posted.minor_units, 400);
}
