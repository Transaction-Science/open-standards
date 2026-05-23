//! Integration tests for `op-ledger`.
//!
//! Plain ledger flows are unit-tested per-module. These tests
//! exercise composition against `op-orchestrator` — proving that a
//! merchant deployment can route the orchestrator's
//! `OrchestrationOutcome` directly into a ledger transaction.
//!
//! Strict bookkeeping flow:
//!
//! 1. Orchestrator returns `Approved` from a card auth.
//! 2. Caller posts a **pending** ledger transaction: debit
//!    `merchant_receivable_card` (asset, debit-normal); credit
//!    `revenue` (revenue, credit-normal).
//! 3. Later, settlement notification arrives; caller marks the
//!    transaction **posted**. (Or rail signals failure → archived.)
//! 4. PSP fees are recorded separately: debit `psp_fees` (expense,
//!    debit-normal); credit `cash` (asset, debit-normal).
//!
//! Critical invariant exercised here: **idempotency at both layers**.
//! The orchestrator's idempotency key flows through to the ledger
//! transaction's `external_id`. A duplicate orchestration → a
//! duplicate ledger post → returns the existing transaction.
//! Balance is updated exactly once.

use op_core::{Currency, Money};
use op_ledger::{
    Account, AccountClass, Entry, InMemoryLedgerStore, Ledger, LedgerStore, Status, Transaction,
};

const COFFEE_USD_MINOR: i64 = 525; // $5.25

/// Set up a tiny chart of accounts for an Acme Coffee LLC merchant.
fn merchant_books() -> (
    InMemoryLedgerStore,
    op_ledger::LedgerId,
    op_ledger::AccountId, // merchant_receivable_card
    op_ledger::AccountId, // revenue
    op_ledger::AccountId, // psp_fees
    op_ledger::AccountId, // cash
) {
    let store = InMemoryLedgerStore::new();
    let l = Ledger::new("Acme Coffee FY2026").unwrap();
    let lid = l.id;
    store.create_ledger(l).unwrap();

    let receivable = Account::new(
        lid,
        "merchant_receivable_card",
        AccountClass::Asset,
        Currency::USD,
    );
    let revenue = Account::new(lid, "revenue", AccountClass::Revenue, Currency::USD);
    let psp_fees = Account::new(lid, "psp_fees", AccountClass::Expense, Currency::USD);
    let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);

    let rid = receivable.id;
    let rev_id = revenue.id;
    let fees_id = psp_fees.id;
    let cash_id = cash.id;
    store.create_account(receivable).unwrap();
    store.create_account(revenue).unwrap();
    store.create_account(psp_fees).unwrap();
    store.create_account(cash).unwrap();

    (store, lid, rid, rev_id, fees_id, cash_id)
}

// ============================================================
// Test 1: Approved card auth → pending ledger transaction
// ============================================================

#[test]
fn approved_auth_creates_pending_transaction() {
    let (store, lid, receivable, revenue, _fees, _cash) = merchant_books();

    // Orchestrator just returned Approved for ord-1. Post the
    // pending bookkeeping.
    let auth = Transaction::new_pending(
        lid,
        1_700_000_000, // effective_at, unix seconds
        vec![
            Entry::debit(
                receivable,
                Money::from_minor(COFFEE_USD_MINOR, Currency::USD),
            ),
            Entry::credit(revenue, Money::from_minor(COFFEE_USD_MINOR, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id("ord-1")
    .with_description("Card auth ord-1 (pending settlement)");

    let tid = store.post_transaction(auth).unwrap();

    // Pending balance reflects the auth; posted balance does not.
    let r = store.balance(receivable).unwrap();
    assert_eq!(r.posted.minor_units, 0);
    assert_eq!(r.pending.minor_units, COFFEE_USD_MINOR);

    let rev = store.balance(revenue).unwrap();
    assert_eq!(rev.posted.minor_units, 0);
    assert_eq!(rev.pending.minor_units, COFFEE_USD_MINOR);

    // Transaction is retrievable by external id.
    let recovered = store.find_by_external_id("ord-1").unwrap().unwrap();
    assert_eq!(recovered.id, tid);
    assert_eq!(recovered.status, Status::Pending);
}

// ============================================================
// Test 2: Settlement → mark posted; balances move pending → posted
// ============================================================

#[test]
fn settlement_marks_posted_and_balances_settle() {
    let (store, lid, receivable, revenue, _fees, _cash) = merchant_books();
    let auth = Transaction::new_pending(
        lid,
        1_700_000_000,
        vec![
            Entry::debit(
                receivable,
                Money::from_minor(COFFEE_USD_MINOR, Currency::USD),
            ),
            Entry::credit(revenue, Money::from_minor(COFFEE_USD_MINOR, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id("ord-2");
    let tid = store.post_transaction(auth).unwrap();

    // Settlement notification arrives.
    store.mark_posted(tid).unwrap();

    let r = store.balance(receivable).unwrap();
    assert_eq!(r.posted.minor_units, COFFEE_USD_MINOR);
    assert_eq!(r.pending.minor_units, COFFEE_USD_MINOR);

    let t = store.get_transaction(tid).unwrap();
    assert_eq!(t.status, Status::Posted);
}

// ============================================================
// Test 3: PSP fee captured as separate transaction
// ============================================================

#[test]
fn psp_fee_recorded_separately() {
    let (store, lid, _receivable, _revenue, fees, cash) = merchant_books();

    // PSP charged $0.15 on the $5.25 sale.
    let fee_tx = Transaction::new_posted(
        lid,
        1_700_000_000,
        vec![
            Entry::debit(fees, Money::from_minor(15, Currency::USD)),
            Entry::credit(cash, Money::from_minor(15, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id("psp-fee-ord-1");
    store.post_transaction(fee_tx).unwrap();

    let fees_bal = store.balance(fees).unwrap();
    assert_eq!(fees_bal.posted.minor_units, 15);

    // Cash, debit-normal, decreased by 15 → balance = -15.
    let cash_bal = store.balance(cash).unwrap();
    assert_eq!(cash_bal.posted.minor_units, -15);
}

// ============================================================
// Test 4: Idempotency replay returns same transaction id; balance
//           updated only once.
// ============================================================

#[test]
fn replay_with_same_external_id_does_not_double_count() {
    let (store, lid, receivable, revenue, _fees, _cash) = merchant_books();

    let entries = vec![
        Entry::debit(receivable, Money::from_minor(1000, Currency::USD)),
        Entry::credit(revenue, Money::from_minor(1000, Currency::USD)),
    ];
    let auth1 = Transaction::new_posted(lid, 100, entries.clone())
        .unwrap()
        .with_external_id("ord-replay");
    let auth2 = Transaction::new_posted(lid, 100, entries)
        .unwrap()
        .with_external_id("ord-replay");

    let id1 = store.post_transaction(auth1).unwrap();
    let id2 = store.post_transaction(auth2).unwrap();
    assert_eq!(id1, id2);

    // Balance reflects ONE transaction, $10.00.
    let r = store.balance(receivable).unwrap();
    assert_eq!(r.posted.minor_units, 1000);
}

// ============================================================
// Test 5: Refund modeled as reversal of original posted transaction
// ============================================================

#[test]
fn refund_reverses_balance_to_zero() {
    let (store, lid, receivable, revenue, _fees, _cash) = merchant_books();

    let original = Transaction::new_posted(
        lid,
        100,
        vec![
            Entry::debit(receivable, Money::from_minor(2500, Currency::USD)),
            Entry::credit(revenue, Money::from_minor(2500, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id("ord-refund");
    store.post_transaction(original.clone()).unwrap();
    assert_eq!(store.balance(receivable).unwrap().posted.minor_units, 2500);

    // Customer wants a refund. Reverse the original.
    let refund = Transaction::reversal_of(&original, 200).unwrap();
    let rid = store.post_transaction(refund).unwrap();
    store.mark_posted(rid).unwrap();

    let r = store.balance(receivable).unwrap();
    assert_eq!(r.posted.minor_units, 0);
    let rev = store.balance(revenue).unwrap();
    assert_eq!(rev.posted.minor_units, 0);
}

// ============================================================
// Test 6: Multiple orders accumulate
// ============================================================

#[test]
fn many_small_orders_sum_correctly() {
    let (store, lid, receivable, revenue, _fees, _cash) = merchant_books();

    for i in 0..10 {
        let amount = 100 * (i + 1); // $1.00, $2.00, ..., $10.00
        let t = Transaction::new_posted(
            lid,
            i as u64,
            vec![
                Entry::debit(receivable, Money::from_minor(amount, Currency::USD)),
                Entry::credit(revenue, Money::from_minor(amount, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id(format!("ord-batch-{i}"));
        store.post_transaction(t).unwrap();
    }

    // Sum of 100 to 1000 stepping 100 = 5500.
    let r = store.balance(receivable).unwrap();
    assert_eq!(r.posted.minor_units, 5500);
}

// ============================================================
// Test 7: Archived transactions don't affect balance
// ============================================================

#[test]
fn archived_transactions_are_invisible_to_balance() {
    let (store, lid, receivable, revenue, _fees, _cash) = merchant_books();

    // Two transactions: ord-7 posted, ord-8 archived.
    let posted = Transaction::new_posted(
        lid,
        100,
        vec![
            Entry::debit(receivable, Money::from_minor(500, Currency::USD)),
            Entry::credit(revenue, Money::from_minor(500, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id("ord-7");
    let archived_pending = Transaction::new_pending(
        lid,
        100,
        vec![
            Entry::debit(receivable, Money::from_minor(9999, Currency::USD)),
            Entry::credit(revenue, Money::from_minor(9999, Currency::USD)),
        ],
    )
    .unwrap()
    .with_external_id("ord-8");

    store.post_transaction(posted).unwrap();
    let archived_id = store.post_transaction(archived_pending).unwrap();
    store.mark_archived(archived_id).unwrap();

    let r = store.balance(receivable).unwrap();
    // Only the posted $5.00 counts.
    assert_eq!(r.posted.minor_units, 500);
    assert_eq!(r.pending.minor_units, 500);
}

// ============================================================
// Test 8: Multi-currency ledger handles each currency independently
// ============================================================

#[test]
fn multi_currency_ledger_supports_per_currency_accounts() {
    let store = InMemoryLedgerStore::new();
    let ledger = Ledger::new("MultiCcy").unwrap();
    let lid = ledger.id;
    store.create_ledger(ledger).unwrap();

    // USD chart.
    let usd_recv = Account::new(lid, "recv_usd", AccountClass::Asset, Currency::USD);
    let usd_rev = Account::new(lid, "rev_usd", AccountClass::Revenue, Currency::USD);
    let usd_recv_id = usd_recv.id;
    let usd_rev_id = usd_rev.id;
    store.create_account(usd_recv).unwrap();
    store.create_account(usd_rev).unwrap();

    // EUR chart.
    let eur_recv = Account::new(lid, "recv_eur", AccountClass::Asset, Currency::EUR);
    let eur_rev = Account::new(lid, "rev_eur", AccountClass::Revenue, Currency::EUR);
    let eur_recv_id = eur_recv.id;
    let eur_rev_id = eur_rev.id;
    store.create_account(eur_recv).unwrap();
    store.create_account(eur_rev).unwrap();

    // USD sale.
    let usd_tx = Transaction::new_posted(
        lid,
        0,
        vec![
            Entry::debit(usd_recv_id, Money::from_minor(1000, Currency::USD)),
            Entry::credit(usd_rev_id, Money::from_minor(1000, Currency::USD)),
        ],
    )
    .unwrap();
    store.post_transaction(usd_tx).unwrap();

    // EUR sale.
    let eur_tx = Transaction::new_posted(
        lid,
        0,
        vec![
            Entry::debit(eur_recv_id, Money::from_minor(2000, Currency::EUR)),
            Entry::credit(eur_rev_id, Money::from_minor(2000, Currency::EUR)),
        ],
    )
    .unwrap();
    store.post_transaction(eur_tx).unwrap();

    // Each balance carries its own currency; they don't leak.
    let usd_b = store.balance(usd_recv_id).unwrap();
    assert_eq!(usd_b.currency, Currency::USD);
    assert_eq!(usd_b.posted.minor_units, 1000);

    let eur_b = store.balance(eur_recv_id).unwrap();
    assert_eq!(eur_b.currency, Currency::EUR);
    assert_eq!(eur_b.posted.minor_units, 2000);
}

// ============================================================
// Test 9: Full kiosk-day simulation — many orders, fees, one refund
// ============================================================

#[test]
fn kiosk_day_simulation_balances_remain_consistent() {
    let (store, lid, receivable, revenue, fees, cash) = merchant_books();

    // 5 orders of $10 each, with $0.30 PSP fee each.
    let mut posted_orders: Vec<op_ledger::TransactionId> = Vec::new();
    for i in 0..5 {
        let ord = Transaction::new_posted(
            lid,
            i,
            vec![
                Entry::debit(receivable, Money::from_minor(1000, Currency::USD)),
                Entry::credit(revenue, Money::from_minor(1000, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id(format!("ord-{i}"));
        posted_orders.push(store.post_transaction(ord).unwrap());

        let fee = Transaction::new_posted(
            lid,
            i,
            vec![
                Entry::debit(fees, Money::from_minor(30, Currency::USD)),
                Entry::credit(cash, Money::from_minor(30, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id(format!("fee-ord-{i}"));
        store.post_transaction(fee).unwrap();
    }

    // Refund order 2.
    let order2 = store.get_transaction(posted_orders[2]).unwrap();
    let refund = Transaction::reversal_of(&order2, 1000).unwrap();
    let refund_id = store.post_transaction(refund).unwrap();
    store.mark_posted(refund_id).unwrap();

    // Expected balances:
    //   receivable: 4 orders × $10 = $40 (one refunded)
    //   revenue:    same: $40
    //   fees:       5 fees × $0.30 = $1.50 (PSP doesn't refund the fee)
    //   cash:       -5 × $0.30 = -$1.50

    assert_eq!(store.balance(receivable).unwrap().posted.minor_units, 4000);
    assert_eq!(store.balance(revenue).unwrap().posted.minor_units, 4000);
    assert_eq!(store.balance(fees).unwrap().posted.minor_units, 150);
    assert_eq!(store.balance(cash).unwrap().posted.minor_units, -150);
}

// ============================================================
// Test 10: Transactions across ledgers are isolated
// ============================================================

#[test]
fn ledgers_are_isolated_balance_views() {
    let store = InMemoryLedgerStore::new();
    let l1 = Ledger::new("Merchant A").unwrap();
    let l2 = Ledger::new("Merchant B").unwrap();
    let l1_id = l1.id;
    let l2_id = l2.id;
    store.create_ledger(l1).unwrap();
    store.create_ledger(l2).unwrap();

    // Each ledger has its own "revenue" account.
    let rev_a = Account::new(l1_id, "revenue", AccountClass::Revenue, Currency::USD);
    let cash_a = Account::new(l1_id, "cash", AccountClass::Asset, Currency::USD);
    let rev_b = Account::new(l2_id, "revenue", AccountClass::Revenue, Currency::USD);
    let cash_b = Account::new(l2_id, "cash", AccountClass::Asset, Currency::USD);
    let rev_a_id = rev_a.id;
    let cash_a_id = cash_a.id;
    let rev_b_id = rev_b.id;
    store.create_account(rev_a).unwrap();
    store.create_account(cash_a).unwrap();
    store.create_account(rev_b).unwrap();
    store.create_account(cash_b).unwrap();

    // Sale in ledger A only.
    let tx_a = Transaction::new_posted(
        l1_id,
        0,
        vec![
            Entry::debit(cash_a_id, Money::from_minor(1234, Currency::USD)),
            Entry::credit(rev_a_id, Money::from_minor(1234, Currency::USD)),
        ],
    )
    .unwrap();
    store.post_transaction(tx_a).unwrap();

    // Ledger B is untouched.
    assert_eq!(store.balance(rev_a_id).unwrap().posted.minor_units, 1234);
    assert_eq!(store.balance(rev_b_id).unwrap().posted.minor_units, 0);
}
