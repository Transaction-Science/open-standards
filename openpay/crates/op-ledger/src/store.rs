//! Pluggable storage backend.
//!
//! The trait surface allows operators to substitute Postgres,
//! TigerBeetle, or any other backing store while the public API
//! stays the same. The [`InMemoryLedgerStore`] is the reference
//! implementation; it's fine for tests and single-process kiosks
//! but **not** for multi-instance production deployments (lacks
//! durability, lacks cross-instance consistency).
//!
//! ## Atomicity
//!
//! Every method that mutates state must be **atomic** with respect
//! to concurrent callers. The in-memory store achieves this via a
//! coarse `Mutex<Inner>`; production stores use database
//! transactions.
//!
//! ## Idempotency
//!
//! [`Self::post_transaction`] is idempotent by `external_id`:
//! re-posting a transaction with the same `external_id` returns
//! the **existing** transaction iff the body matches; otherwise
//! returns [`Error::IdempotencyMismatch`].

use std::collections::HashMap;
use std::sync::Mutex;

use crate::account::{Account, AccountId};
use crate::balance::Balance;
use crate::entry::Direction;
use crate::error::{Error, Result};
use crate::ledger::{Ledger, LedgerId};
use crate::transaction::{Status, Transaction, TransactionId};
use op_core::Money;

/// The pluggable store interface.
///
/// All methods are sync. Async wrappers (tokio etc.) belong in
/// adapter crates downstream.
pub trait LedgerStore: Send + Sync {
    /// Create a ledger and return its id.
    fn create_ledger(&self, ledger: Ledger) -> Result<LedgerId>;

    /// Look up a ledger by id.
    fn get_ledger(&self, id: LedgerId) -> Result<Ledger>;

    /// Create an account in a ledger.
    fn create_account(&self, account: Account) -> Result<AccountId>;

    /// Look up an account.
    fn get_account(&self, id: AccountId) -> Result<Account>;

    /// Post a transaction. Validates currency-per-account,
    /// idempotency, and cross-ledger constraints in addition to
    /// the double-entry invariant validated at `Transaction`
    /// construction time.
    ///
    /// Returns the persisted transaction id. If a transaction with
    /// the same `external_id` already exists with a matching body,
    /// returns the existing id.
    fn post_transaction(&self, transaction: Transaction) -> Result<TransactionId>;

    /// Look up a transaction.
    fn get_transaction(&self, id: TransactionId) -> Result<Transaction>;

    /// Look up a transaction by `external_id`. Returns `None` if no
    /// transaction with that id exists.
    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Transaction>>;

    /// Transition a pending transaction to posted. The store
    /// applies the same terminal-state checks as
    /// `Transaction::post`.
    fn mark_posted(&self, id: TransactionId) -> Result<()>;

    /// Transition a pending transaction to archived.
    fn mark_archived(&self, id: TransactionId) -> Result<()>;

    /// Compute the current balance of an account.
    fn balance(&self, id: AccountId) -> Result<Balance>;
}

// ============================================================
// InMemoryLedgerStore
// ============================================================

#[derive(Default)]
struct Inner {
    ledgers: HashMap<LedgerId, Ledger>,
    accounts: HashMap<AccountId, Account>,
    transactions: HashMap<TransactionId, Transaction>,
    /// external_id → transaction_id reverse index for idempotency.
    by_external_id: HashMap<String, TransactionId>,
}

/// In-process store. NOT for multi-instance production.
#[derive(Default)]
pub struct InMemoryLedgerStore {
    inner: Mutex<Inner>,
}

impl InMemoryLedgerStore {
    /// Construct empty.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of ledgers tracked (for diagnostics).
    #[must_use]
    pub fn ledger_count(&self) -> usize {
        self.inner.lock().expect("poisoned").ledgers.len()
    }

    /// Number of accounts tracked.
    #[must_use]
    pub fn account_count(&self) -> usize {
        self.inner.lock().expect("poisoned").accounts.len()
    }

    /// Number of transactions tracked.
    #[must_use]
    pub fn transaction_count(&self) -> usize {
        self.inner.lock().expect("poisoned").transactions.len()
    }

    /// Verify that the same entries (set-equality on (account, direction, amount))
    /// appear in both transactions. Used for idempotency-mismatch
    /// detection.
    fn same_body(a: &Transaction, b: &Transaction) -> bool {
        if a.ledger_id != b.ledger_id {
            return false;
        }
        if a.entries.len() != b.entries.len() {
            return false;
        }
        // Order-independent comparison.
        let mut a_sorted = a.entries.clone();
        let mut b_sorted = b.entries.clone();
        a_sorted.sort_by_key(|e| (e.account_id, e.direction, e.amount.minor_units));
        b_sorted.sort_by_key(|e| (e.account_id, e.direction, e.amount.minor_units));
        a_sorted == b_sorted
    }
}

impl LedgerStore for InMemoryLedgerStore {
    fn create_ledger(&self, ledger: Ledger) -> Result<LedgerId> {
        let mut g = self.inner.lock().expect("poisoned");
        let id = ledger.id;
        g.ledgers.insert(id, ledger);
        Ok(id)
    }

    fn get_ledger(&self, id: LedgerId) -> Result<Ledger> {
        let g = self.inner.lock().expect("poisoned");
        g.ledgers
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::LedgerNotFound(id.to_string()))
    }

    fn create_account(&self, account: Account) -> Result<AccountId> {
        let mut g = self.inner.lock().expect("poisoned");
        if !g.ledgers.contains_key(&account.ledger_id) {
            return Err(Error::LedgerNotFound(account.ledger_id.to_string()));
        }
        let id = account.id;
        g.accounts.insert(id, account);
        Ok(id)
    }

    fn get_account(&self, id: AccountId) -> Result<Account> {
        let g = self.inner.lock().expect("poisoned");
        g.accounts
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::AccountNotFound(id.to_string()))
    }

    #[tracing::instrument(
        name = "ledger.post_transaction",
        skip(self, transaction),
        fields(
            tx_id = %transaction.id,
            ledger_id = %transaction.ledger_id,
            external_id = transaction.external_id.as_deref().unwrap_or(""),
            entries = transaction.entries.len(),
        ),
    )]
    fn post_transaction(&self, transaction: Transaction) -> Result<TransactionId> {
        let mut g = self.inner.lock().expect("poisoned");

        // 1. Ledger must exist.
        if !g.ledgers.contains_key(&transaction.ledger_id) {
            return Err(Error::LedgerNotFound(transaction.ledger_id.to_string()));
        }

        // 2. Idempotency by external_id.
        if let Some(ext) = &transaction.external_id
            && let Some(existing_id) = g.by_external_id.get(ext).copied()
        {
            let existing = g
                .transactions
                .get(&existing_id)
                .cloned()
                .ok_or_else(|| Error::TransactionNotFound(existing_id.to_string()))?;
            if !Self::same_body(&existing, &transaction) {
                return Err(Error::IdempotencyMismatch);
            }
            return Ok(existing_id);
        }

        // 3. Validate every entry: account exists, belongs to the
        //    same ledger, currency matches.
        for entry in &transaction.entries {
            let account = g
                .accounts
                .get(&entry.account_id)
                .cloned()
                .ok_or_else(|| Error::AccountNotFound(entry.account_id.to_string()))?;
            if account.ledger_id != transaction.ledger_id {
                return Err(Error::CrossLedgerEntry {
                    account_id: entry.account_id.to_string(),
                    account_ledger: account.ledger_id.to_string(),
                    expected_ledger: transaction.ledger_id.to_string(),
                });
            }
            if account.currency != entry.amount.currency {
                return Err(Error::CurrencyMismatch {
                    entry_currency: entry.amount.currency.code().to_owned(),
                    account_currency: account.currency.code().to_owned(),
                    account_id: entry.account_id.to_string(),
                });
            }
        }

        // 4. Persist.
        let id = transaction.id;
        if let Some(ext) = &transaction.external_id {
            g.by_external_id.insert(ext.clone(), id);
        }
        g.transactions.insert(id, transaction);
        Ok(id)
    }

    fn get_transaction(&self, id: TransactionId) -> Result<Transaction> {
        let g = self.inner.lock().expect("poisoned");
        g.transactions
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::TransactionNotFound(id.to_string()))
    }

    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Transaction>> {
        let g = self.inner.lock().expect("poisoned");
        Ok(g.by_external_id
            .get(external_id)
            .and_then(|id| g.transactions.get(id))
            .cloned())
    }

    fn mark_posted(&self, id: TransactionId) -> Result<()> {
        let mut g = self.inner.lock().expect("poisoned");
        let t = g
            .transactions
            .get_mut(&id)
            .ok_or_else(|| Error::TransactionNotFound(id.to_string()))?;
        t.post()
    }

    fn mark_archived(&self, id: TransactionId) -> Result<()> {
        let mut g = self.inner.lock().expect("poisoned");
        let t = g
            .transactions
            .get_mut(&id)
            .ok_or_else(|| Error::TransactionNotFound(id.to_string()))?;
        t.archive()
    }

    fn balance(&self, account_id: AccountId) -> Result<Balance> {
        let g = self.inner.lock().expect("poisoned");
        let account = g
            .accounts
            .get(&account_id)
            .ok_or_else(|| Error::AccountNotFound(account_id.to_string()))?;
        let currency = account.currency;
        let normal = account.normal_balance;

        // Walk every transaction's entries; tally per status.
        // (Posted contributes to posted+pending; pending contributes
        // to pending only; archived contributes to nothing.)
        let mut posted_debits: i64 = 0;
        let mut posted_credits: i64 = 0;
        let mut pending_debits: i64 = 0;
        let mut pending_credits: i64 = 0;

        for t in g.transactions.values() {
            for entry in &t.entries {
                if entry.account_id != account_id {
                    continue;
                }
                match (t.status, entry.direction) {
                    (Status::Posted, Direction::Debit) => {
                        posted_debits = posted_debits
                            .checked_add(entry.amount.minor_units)
                            .ok_or(op_core::Error::Overflow)?;
                    }
                    (Status::Posted, Direction::Credit) => {
                        posted_credits = posted_credits
                            .checked_add(entry.amount.minor_units)
                            .ok_or(op_core::Error::Overflow)?;
                    }
                    (Status::Pending, Direction::Debit) => {
                        pending_debits = pending_debits
                            .checked_add(entry.amount.minor_units)
                            .ok_or(op_core::Error::Overflow)?;
                    }
                    (Status::Pending, Direction::Credit) => {
                        pending_credits = pending_credits
                            .checked_add(entry.amount.minor_units)
                            .ok_or(op_core::Error::Overflow)?;
                    }
                    (Status::Archived, _) => { /* skip */ }
                }
            }
        }

        let posted_amt = sign_by_normal(normal, posted_debits, posted_credits)?;
        // Pending balance includes posted + pending.
        let total_debits = posted_debits
            .checked_add(pending_debits)
            .ok_or(op_core::Error::Overflow)?;
        let total_credits = posted_credits
            .checked_add(pending_credits)
            .ok_or(op_core::Error::Overflow)?;
        let pending_amt = sign_by_normal(normal, total_debits, total_credits)?;

        Ok(Balance {
            currency,
            posted: Money::from_minor(posted_amt, currency),
            pending: Money::from_minor(pending_amt, currency),
        })
    }
}

/// Sign the (debits − credits) or (credits − debits) appropriately
/// for the account's normal balance.
///
/// For a debit-normal account: balance = debits − credits.
/// For a credit-normal account: balance = credits − debits.
fn sign_by_normal(normal: crate::account::NormalBalance, debits: i64, credits: i64) -> Result<i64> {
    use crate::account::NormalBalance::*;
    let result = match normal {
        Debit => debits.checked_sub(credits),
        Credit => credits.checked_sub(debits),
    };
    result.ok_or_else(|| op_core::Error::Overflow.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account::AccountClass;
    use crate::entry::Entry;
    use crate::ledger::Ledger;
    use op_core::Currency;

    fn setup() -> (InMemoryLedgerStore, LedgerId, AccountId, AccountId) {
        let store = InMemoryLedgerStore::new();
        let ledger = Ledger::new("test").unwrap();
        let lid = ledger.id;
        store.create_ledger(ledger).unwrap();

        // Asset: debit-normal. Revenue: credit-normal.
        let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
        let revenue = Account::new(lid, "revenue", AccountClass::Revenue, Currency::USD);
        let cash_id = cash.id;
        let rev_id = revenue.id;
        store.create_account(cash).unwrap();
        store.create_account(revenue).unwrap();

        (store, lid, cash_id, rev_id)
    }

    #[test]
    fn empty_account_has_zero_balance() {
        let (store, _lid, cash, _rev) = setup();
        let b = store.balance(cash).unwrap();
        assert!(b.is_zero());
    }

    #[test]
    fn posted_transaction_updates_both_views() {
        let (store, lid, cash, rev) = setup();
        // Customer paid $10.00 for coffee. Debit cash, credit revenue.
        let t = Transaction::new_posted(
            lid,
            100,
            vec![
                Entry::debit(cash, Money::from_minor(1000, Currency::USD)),
                Entry::credit(rev, Money::from_minor(1000, Currency::USD)),
            ],
        )
        .unwrap();
        store.post_transaction(t).unwrap();

        let cash_balance = store.balance(cash).unwrap();
        assert_eq!(cash_balance.posted.minor_units, 1000);
        assert_eq!(cash_balance.pending.minor_units, 1000);

        let rev_balance = store.balance(rev).unwrap();
        assert_eq!(rev_balance.posted.minor_units, 1000);
        assert_eq!(rev_balance.pending.minor_units, 1000);
    }

    #[test]
    fn pending_transaction_only_affects_pending_view() {
        let (store, lid, cash, rev) = setup();
        let t = Transaction::new_pending(
            lid,
            100,
            vec![
                Entry::debit(cash, Money::from_minor(2500, Currency::USD)),
                Entry::credit(rev, Money::from_minor(2500, Currency::USD)),
            ],
        )
        .unwrap();
        store.post_transaction(t).unwrap();

        let cash_balance = store.balance(cash).unwrap();
        assert_eq!(cash_balance.posted.minor_units, 0);
        assert_eq!(cash_balance.pending.minor_units, 2500);
    }

    #[test]
    fn mark_posted_moves_pending_to_posted() {
        let (store, lid, cash, rev) = setup();
        let t = Transaction::new_pending(
            lid,
            100,
            vec![
                Entry::debit(cash, Money::from_minor(500, Currency::USD)),
                Entry::credit(rev, Money::from_minor(500, Currency::USD)),
            ],
        )
        .unwrap();
        let tid = store.post_transaction(t).unwrap();

        store.mark_posted(tid).unwrap();
        let bal = store.balance(cash).unwrap();
        assert_eq!(bal.posted.minor_units, 500);
        assert_eq!(bal.pending.minor_units, 500);
    }

    #[test]
    fn mark_archived_removes_from_both_views() {
        let (store, lid, cash, rev) = setup();
        let t = Transaction::new_pending(
            lid,
            100,
            vec![
                Entry::debit(cash, Money::from_minor(500, Currency::USD)),
                Entry::credit(rev, Money::from_minor(500, Currency::USD)),
            ],
        )
        .unwrap();
        let tid = store.post_transaction(t).unwrap();
        // Before archive: pending balance = 500.
        assert_eq!(store.balance(cash).unwrap().pending.minor_units, 500);

        store.mark_archived(tid).unwrap();
        // After archive: both 0.
        let b = store.balance(cash).unwrap();
        assert!(b.is_zero());
    }

    #[test]
    fn idempotency_returns_existing_id_for_matching_body() {
        let (store, lid, cash, rev) = setup();
        let entries = vec![
            Entry::debit(cash, Money::from_minor(700, Currency::USD)),
            Entry::credit(rev, Money::from_minor(700, Currency::USD)),
        ];
        let t1 = Transaction::new_posted(lid, 100, entries.clone())
            .unwrap()
            .with_external_id("ord-7");
        let t2 = Transaction::new_posted(lid, 100, entries)
            .unwrap()
            .with_external_id("ord-7");

        let id1 = store.post_transaction(t1).unwrap();
        let id2 = store.post_transaction(t2).unwrap();
        assert_eq!(id1, id2, "second post must return existing id");

        // And the rail was only "charged" once.
        let bal = store.balance(cash).unwrap();
        assert_eq!(
            bal.posted.minor_units, 700,
            "balance must reflect ONE transaction, not two"
        );
    }

    #[test]
    fn idempotency_mismatch_rejects_different_body() {
        let (store, lid, cash, rev) = setup();
        let t1 = Transaction::new_posted(
            lid,
            100,
            vec![
                Entry::debit(cash, Money::from_minor(500, Currency::USD)),
                Entry::credit(rev, Money::from_minor(500, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id("ord-mismatch");
        let t2 = Transaction::new_posted(
            lid,
            100,
            vec![
                Entry::debit(cash, Money::from_minor(9999, Currency::USD)),
                Entry::credit(rev, Money::from_minor(9999, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id("ord-mismatch");

        store.post_transaction(t1).unwrap();
        let r = store.post_transaction(t2);
        assert!(matches!(r, Err(Error::IdempotencyMismatch)));
    }

    #[test]
    fn cross_ledger_entry_rejected() {
        let store = InMemoryLedgerStore::new();
        let ledger_a = Ledger::new("a").unwrap();
        let ledger_b = Ledger::new("b").unwrap();
        let lid_a = ledger_a.id;
        let lid_b = ledger_b.id;
        store.create_ledger(ledger_a).unwrap();
        store.create_ledger(ledger_b).unwrap();

        let acct_a = Account::new(lid_a, "a", AccountClass::Asset, Currency::USD);
        let acct_b = Account::new(lid_b, "b", AccountClass::Asset, Currency::USD);
        let aid_a = acct_a.id;
        let aid_b = acct_b.id;
        store.create_account(acct_a).unwrap();
        store.create_account(acct_b).unwrap();

        // A transaction in ledger_a referencing an account from
        // ledger_b should be rejected.
        let t = Transaction::new_posted(
            lid_a,
            0,
            vec![
                Entry::debit(aid_a, Money::from_minor(100, Currency::USD)),
                Entry::credit(aid_b, Money::from_minor(100, Currency::USD)),
            ],
        )
        .unwrap();
        let r = store.post_transaction(t);
        assert!(matches!(r, Err(Error::CrossLedgerEntry { .. })));
    }

    #[test]
    fn currency_mismatch_rejected() {
        let store = InMemoryLedgerStore::new();
        let ledger = Ledger::new("test").unwrap();
        let lid = ledger.id;
        store.create_ledger(ledger).unwrap();

        let usd_acct = Account::new(lid, "usd", AccountClass::Asset, Currency::USD);
        let eur_acct = Account::new(lid, "eur", AccountClass::Asset, Currency::EUR);
        let usd_id = usd_acct.id;
        let eur_id = eur_acct.id;
        store.create_account(usd_acct).unwrap();
        store.create_account(eur_acct).unwrap();

        // A USD entry against an EUR account must be rejected.
        // We can't construct the transaction with mixed currency on
        // the same account because the balanced-check passes (USD
        // debits == USD credits, EUR debits == EUR credits), so the
        // store must catch the per-entry mismatch.
        let t = Transaction::new_posted(
            lid,
            0,
            vec![
                Entry::debit(usd_id, Money::from_minor(100, Currency::USD)),
                Entry::credit(eur_id, Money::from_minor(100, Currency::USD)),
            ],
        )
        .unwrap();
        let r = store.post_transaction(t);
        assert!(matches!(r, Err(Error::CurrencyMismatch { .. })));
    }

    #[test]
    fn unknown_account_rejected() {
        let (store, lid, cash, _rev) = setup();
        // Reference an account that exists in NO ledger.
        let phantom = AccountId::new();
        let t = Transaction::new_posted(
            lid,
            0,
            vec![
                Entry::debit(phantom, Money::from_minor(100, Currency::USD)),
                Entry::credit(cash, Money::from_minor(100, Currency::USD)),
            ],
        )
        .unwrap();
        let r = store.post_transaction(t);
        assert!(matches!(r, Err(Error::AccountNotFound(_))));
    }

    #[test]
    fn unknown_ledger_rejected_for_transaction() {
        let store = InMemoryLedgerStore::new();
        let phantom_lid = LedgerId::new();
        let a1 = AccountId::new();
        let a2 = AccountId::new();
        let t = Transaction::new_posted(
            phantom_lid,
            0,
            vec![
                Entry::debit(a1, Money::from_minor(1, Currency::USD)),
                Entry::credit(a2, Money::from_minor(1, Currency::USD)),
            ],
        )
        .unwrap();
        let r = store.post_transaction(t);
        assert!(matches!(r, Err(Error::LedgerNotFound(_))));
    }

    #[test]
    fn unknown_account_for_balance_lookup() {
        let store = InMemoryLedgerStore::new();
        let r = store.balance(AccountId::new());
        assert!(matches!(r, Err(Error::AccountNotFound(_))));
    }

    #[test]
    fn find_by_external_id_returns_some_for_known_id() {
        let (store, lid, cash, rev) = setup();
        let t = Transaction::new_posted(
            lid,
            0,
            vec![
                Entry::debit(cash, Money::from_minor(100, Currency::USD)),
                Entry::credit(rev, Money::from_minor(100, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id("ord-9");
        store.post_transaction(t).unwrap();

        let found = store.find_by_external_id("ord-9").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().external_id.as_deref(), Some("ord-9"));
    }

    #[test]
    fn find_by_external_id_returns_none_for_unknown() {
        let (store, _, _, _) = setup();
        let found = store.find_by_external_id("missing").unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn reversal_flow_zeroes_the_balance() {
        let (store, lid, cash, rev) = setup();
        let original = Transaction::new_posted(
            lid,
            100,
            vec![
                Entry::debit(cash, Money::from_minor(500, Currency::USD)),
                Entry::credit(rev, Money::from_minor(500, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id("ord-rev");
        store.post_transaction(original.clone()).unwrap();
        assert_eq!(store.balance(cash).unwrap().posted.minor_units, 500);

        // Reverse it. The reversal is constructed as pending; we
        // post it then mark it posted.
        let rev_tx = Transaction::reversal_of(&original, 200).unwrap();
        let rev_id = store.post_transaction(rev_tx).unwrap();
        store.mark_posted(rev_id).unwrap();

        // Balance is back to zero.
        let bal = store.balance(cash).unwrap();
        assert_eq!(bal.posted.minor_units, 0);
        assert_eq!(bal.pending.minor_units, 0);
    }

    #[test]
    fn debit_normal_balance_sign() {
        let (store, lid, cash, rev) = setup();
        // Debit cash $100. Since cash is debit-normal, balance is
        // POSITIVE.
        let t = Transaction::new_posted(
            lid,
            0,
            vec![
                Entry::debit(cash, Money::from_minor(100, Currency::USD)),
                Entry::credit(rev, Money::from_minor(100, Currency::USD)),
            ],
        )
        .unwrap();
        store.post_transaction(t).unwrap();
        assert_eq!(store.balance(cash).unwrap().posted.minor_units, 100);
        // Revenue is credit-normal, also positive.
        assert_eq!(store.balance(rev).unwrap().posted.minor_units, 100);
    }

    #[test]
    fn diagnostic_counters() {
        let (store, _lid, _cash, _rev) = setup();
        assert_eq!(store.ledger_count(), 1);
        assert_eq!(store.account_count(), 2);
        assert_eq!(store.transaction_count(), 0);
    }

    #[test]
    fn account_creation_in_missing_ledger_rejected() {
        let store = InMemoryLedgerStore::new();
        let phantom_lid = LedgerId::new();
        let a = Account::new(phantom_lid, "x", AccountClass::Asset, Currency::USD);
        let r = store.create_account(a);
        assert!(matches!(r, Err(Error::LedgerNotFound(_))));
    }
}
