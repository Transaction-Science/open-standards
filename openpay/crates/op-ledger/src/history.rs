//! [`LedgerHistory`] — time-travel queries for stores with a
//! bi-temporal substrate.
//!
//! Not every [`LedgerStore`](crate::LedgerStore) keeps history:
//! [`InMemoryLedgerStore`](crate::InMemoryLedgerStore) is a snapshot
//! of "now," like a SQL row that gets updated in place. A store
//! built on a bi-temporal fact log (e.g. the graph-backed store in
//! `op-graph` over Minigraf's append-only EAV facts) *does* keep
//! every prior value of every property, so it can answer "what did
//! the books look like at point X."
//!
//! This trait exposes that capability without forcing every store
//! to implement it. Operators check for the trait before relying on
//! it (or pick a store implementation that's documented as
//! providing it).
//!
//! ## Reference point: the transaction counter
//!
//! All [`LedgerHistory`] methods take a `tx_count: u64`. This is a
//! monotonic counter the underlying store advances by 1 on each
//! write. Operators **snapshot** it right after a write they want
//! to time-travel against later:
//!
//! ```text
//! let id = store.post_transaction(tx)?;
//! let snap = store.tx_count();          // tx_count() lives on
//!                                       // GraphLedgerStore;
//!                                       // see op-graph.
//! // ... days later ...
//! let bal_then = store.balance_as_of(account, snap)?;
//! ```
//!
//! The counter is opaque: don't try to subtract two counts and
//! interpret the result as a wall-clock duration. Use it only to
//! identify a *moment* in the store's history.

use crate::account::AccountId;
use crate::balance::Balance;
use crate::error::Result;
use crate::transaction::{Transaction, TransactionId};

/// Time-travel reads against a store whose substrate retains history.
///
/// The first two methods take the opaque
/// [`tx_count`](crate::LedgerHistory::balance_as_of). The `_at_time`
/// variants accept a wall-clock unix-seconds value — operators and
/// auditors think in dates, not opaque counters. The named-checkpoint
/// helpers let an operator stash and recall a counter as a single
/// keyword (`"Q4-2025-close"` → `42_999`).
pub trait LedgerHistory {
    /// The balance of `account` as it stood at counter `tx_count`.
    ///
    /// Pending and posted are computed from the entries that were
    /// in force at that moment; a transaction posted between
    /// `tx_count` and now does not contribute to the historical
    /// balance.
    ///
    /// # Errors
    /// Backend-specific. The caller-supplied `tx_count` need not
    /// correspond to a real past moment; values past the current
    /// `tx_count` simply return the present state.
    fn balance_as_of(&self, account: AccountId, tx_count: u64) -> Result<Balance>;

    /// The transaction with `id` as it stood at counter `tx_count`.
    ///
    /// Properties that change over a transaction's lifetime (notably
    /// [`Status`](crate::Status) — `Pending` → `Posted`/`Archived`)
    /// reflect their historical value. Entries themselves are
    /// immutable once posted, so the entry list matches the present.
    ///
    /// # Errors
    /// `Error::TransactionNotFound` if the transaction didn't exist
    /// at `tx_count` (was created later, or never existed).
    fn transaction_as_of(&self, id: TransactionId, tx_count: u64) -> Result<Transaction>;

    /// Wall-clock variant of [`balance_as_of`](Self::balance_as_of).
    /// Returns the balance as it stood *just after* the most-recent
    /// transaction whose `effective_at_unix_secs` is at-or-before
    /// `at_unix_secs`. If no such transaction exists, returns a
    /// zero balance in the account's currency.
    ///
    /// # Errors
    /// Backend-specific.
    fn balance_as_of_time(&self, account: AccountId, at_unix_secs: u64) -> Result<Balance>;

    /// Wall-clock variant of [`transaction_as_of`](Self::transaction_as_of).
    /// Returns the transaction's state at the counter snapshotted
    /// after the most-recent qualifying transaction (see
    /// [`balance_as_of_time`](Self::balance_as_of_time) for the
    /// time-anchoring rule).
    ///
    /// # Errors
    /// `Error::TransactionNotFound` if the tx didn't exist at
    /// `at_unix_secs`.
    fn transaction_as_of_time(&self, id: TransactionId, at_unix_secs: u64) -> Result<Transaction>;

    /// Persist `name → current_tx_count` in the store. Re-saving
    /// the same name overwrites the previous mapping. Idempotent
    /// across processes when the store is durable.
    ///
    /// # Errors
    /// Backend-specific persistence failure.
    fn save_checkpoint(&self, name: &str) -> Result<u64>;

    /// Look up a previously-saved checkpoint. `None` if no such
    /// name was saved.
    ///
    /// # Errors
    /// Backend-specific read failure.
    fn tx_count_at_checkpoint(&self, name: &str) -> Result<Option<u64>>;

    /// The set of [`TransactionId`]s that were posted in the
    /// inclusive counter window `[start_tx, end_tx]` — i.e. whose
    /// `posted_at_tx_count` is in that range. Useful for "show me
    /// every booking made between two snapshots" replay queries.
    ///
    /// # Errors
    /// Backend-specific. Returns an empty list if `end_tx <
    /// start_tx`.
    fn replay_window(&self, start_tx: u64, end_tx: u64) -> Result<Vec<TransactionId>>;
}
