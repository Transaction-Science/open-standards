//! Ledger transactions.
//!
//! A [`Transaction`] is a balanced set of [`Entry`] rows that the
//! ledger applies atomically. It has a [`Status`] lifecycle:
//!
//! ```text
//!    Pending  ──post()──►  Posted  (terminal)
//!       │
//!       └──archive()──►  Archived (terminal)
//! ```
//!
//! Once `Posted` or `Archived`, a transaction is immutable. To
//! correct a posted transaction, post a new one constructed via
//! [`Transaction::reversal_of`].

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::entry::{Direction, Entry};
use crate::error::{Error, Result};

/// Opaque transaction id.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TransactionId(pub Uuid);

impl TransactionId {
    /// Generate a fresh id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Wrap an existing UUID.
    #[must_use]
    pub fn from_uuid(u: Uuid) -> Self {
        Self(u)
    }

    /// The underlying UUID.
    #[must_use]
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for TransactionId {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for TransactionId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// Lifecycle state of a transaction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Status {
    /// Authorized but not yet settled. Counts toward pending balance
    /// only.
    Pending,
    /// Settled. Counts toward both pending and posted balances.
    /// Immutable once entered.
    Posted,
    /// Cancelled before posting (e.g. card auth voided). Does not
    /// contribute to any balance. Immutable once entered.
    Archived,
}

impl Status {
    /// True if the status is terminal (no further state transitions
    /// allowed).
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Posted | Self::Archived)
    }
}

/// A balanced double-entry transaction.
///
/// Construct via [`Transaction::new_pending`] or
/// [`Transaction::new_posted`]; both validate the double-entry
/// invariant at construction time. Reversals via
/// [`Transaction::reversal_of`].
///
/// The `external_id` is the caller's idempotency key. Two
/// transactions with the same `external_id` are treated as the same
/// logical event by the [`LedgerStore`](crate::LedgerStore).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transaction {
    /// Stable id.
    pub id: TransactionId,

    /// Which ledger this transaction belongs to.
    pub ledger_id: crate::ledger::LedgerId,

    /// Current status.
    pub status: Status,

    /// Caller-supplied idempotency key. Used by the store to dedupe.
    pub external_id: Option<String>,

    /// Human-readable description (`"Card auth ORD-123"`).
    pub description: Option<String>,

    /// Caller-supplied effective timestamp (unix epoch seconds).
    /// When the transaction is considered to have happened in the
    /// real world. Distinct from the implementation's clock.
    pub effective_at_unix_secs: u64,

    /// The balanced entries.
    pub entries: Vec<Entry>,

    /// Free-form metadata.
    pub metadata: Vec<(String, String)>,
}

impl Transaction {
    /// Construct a new pending transaction. Validates the
    /// double-entry invariant per currency.
    ///
    /// # Errors
    /// - [`Error::TooFewEntries`] if fewer than 2 entries.
    /// - [`Error::Unbalanced`] if debits ≠ credits for any currency.
    pub fn new_pending(
        ledger_id: crate::ledger::LedgerId,
        effective_at_unix_secs: u64,
        entries: Vec<Entry>,
    ) -> Result<Self> {
        Self::construct(ledger_id, Status::Pending, effective_at_unix_secs, entries)
    }

    /// Construct a new posted transaction. Validates the
    /// double-entry invariant per currency.
    ///
    /// # Errors
    /// See [`new_pending`](Self::new_pending).
    pub fn new_posted(
        ledger_id: crate::ledger::LedgerId,
        effective_at_unix_secs: u64,
        entries: Vec<Entry>,
    ) -> Result<Self> {
        Self::construct(ledger_id, Status::Posted, effective_at_unix_secs, entries)
    }

    fn construct(
        ledger_id: crate::ledger::LedgerId,
        status: Status,
        effective_at_unix_secs: u64,
        entries: Vec<Entry>,
    ) -> Result<Self> {
        if entries.len() < 2 {
            return Err(Error::TooFewEntries(entries.len()));
        }
        Self::validate_balanced(&entries)?;
        Ok(Self {
            id: TransactionId::new(),
            ledger_id,
            status,
            external_id: None,
            description: None,
            effective_at_unix_secs,
            entries,
            metadata: Vec::new(),
        })
    }

    /// Verify the double-entry invariant: for every currency
    /// appearing in `entries`, sum of debit amounts == sum of credit
    /// amounts.
    fn validate_balanced(entries: &[Entry]) -> Result<()> {
        // (debit_total, credit_total) per currency code.
        let mut tallies: HashMap<String, (i64, i64)> = HashMap::new();
        for e in entries {
            let cur = e.amount.currency.code().to_owned();
            let entry = tallies.entry(cur).or_insert((0, 0));
            match e.direction {
                Direction::Debit => {
                    entry.0 = entry
                        .0
                        .checked_add(e.amount.minor_units)
                        .ok_or(op_core::Error::Overflow)?;
                }
                Direction::Credit => {
                    entry.1 = entry
                        .1
                        .checked_add(e.amount.minor_units)
                        .ok_or(op_core::Error::Overflow)?;
                }
            }
        }
        for (cur, (debits, credits)) in tallies {
            if debits != credits {
                return Err(Error::Unbalanced {
                    currency: cur,
                    debits,
                    credits,
                });
            }
        }
        Ok(())
    }

    /// Builder: set an external id.
    #[must_use]
    pub fn with_external_id(mut self, id: impl Into<String>) -> Self {
        self.external_id = Some(id.into());
        self
    }

    /// Builder: set a description.
    #[must_use]
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }

    /// Builder: append a metadata pair.
    #[must_use]
    pub fn with_metadata(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.metadata.push((k.into(), v.into()));
        self
    }

    /// Construct the reversal of a posted transaction.
    ///
    /// Returns a new pending transaction whose entries are the
    /// originals with directions flipped. Same amounts, same
    /// accounts, opposite sides. The result is automatically
    /// balanced because the original was.
    ///
    /// The new transaction carries an `external_id` derived from
    /// the original's id with a `:reversal` suffix so the
    /// idempotency store doesn't collide.
    ///
    /// # Errors
    /// [`Error::TerminalState`] is NOT returned — reversal works on
    /// posted transactions (that's the whole point). It would
    /// however be silly to reverse a pending or archived
    /// transaction; we don't forbid it programmatically but
    /// document the convention.
    pub fn reversal_of(original: &Transaction, effective_at_unix_secs: u64) -> Result<Self> {
        let entries: Vec<Entry> = original.entries.iter().map(Entry::reverse).collect();
        let mut t = Self::new_pending(original.ledger_id, effective_at_unix_secs, entries)?;
        let parent_id = original
            .external_id
            .clone()
            .unwrap_or_else(|| original.id.to_string());
        t.external_id = Some(format!("{parent_id}:reversal"));
        t.description = Some(format!("Reversal of {}", original.id));
        t.metadata
            .push(("reverses".into(), original.id.to_string()));
        Ok(t)
    }

    /// Transition a pending transaction to posted.
    ///
    /// # Errors
    /// [`Error::TerminalState`] if the transaction is already in a
    /// terminal state.
    pub fn post(&mut self) -> Result<()> {
        match self.status {
            Status::Pending => {
                self.status = Status::Posted;
                Ok(())
            }
            terminal => Err(Error::TerminalState {
                id: self.id.to_string(),
                state: terminal,
            }),
        }
    }

    /// Transition a pending transaction to archived.
    ///
    /// # Errors
    /// [`Error::TerminalState`] if the transaction is already in a
    /// terminal state.
    pub fn archive(&mut self) -> Result<()> {
        match self.status {
            Status::Pending => {
                self.status = Status::Archived;
                Ok(())
            }
            terminal => Err(Error::TerminalState {
                id: self.id.to_string(),
                state: terminal,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account::AccountId;
    use crate::ledger::LedgerId;
    use op_core::{Currency, Money};

    fn aid() -> AccountId {
        AccountId::new()
    }
    fn lid() -> LedgerId {
        LedgerId::new()
    }

    fn balanced_pair() -> Vec<Entry> {
        let a = aid();
        let b = aid();
        let m = Money::from_minor(1000, Currency::USD);
        vec![Entry::debit(a, m), Entry::credit(b, m)]
    }

    #[test]
    fn status_is_terminal_for_posted_and_archived() {
        assert!(!Status::Pending.is_terminal());
        assert!(Status::Posted.is_terminal());
        assert!(Status::Archived.is_terminal());
    }

    #[test]
    fn balanced_pair_constructs() {
        let t = Transaction::new_pending(lid(), 0, balanced_pair()).unwrap();
        assert_eq!(t.status, Status::Pending);
        assert_eq!(t.entries.len(), 2);
    }

    #[test]
    fn too_few_entries_rejected() {
        let one = vec![Entry::debit(aid(), Money::from_minor(1, Currency::USD))];
        let r = Transaction::new_pending(lid(), 0, one);
        assert!(matches!(r, Err(Error::TooFewEntries(1))));
        let none = vec![];
        let r2 = Transaction::new_pending(lid(), 0, none);
        assert!(matches!(r2, Err(Error::TooFewEntries(0))));
    }

    #[test]
    fn unbalanced_rejected() {
        let a = aid();
        let b = aid();
        let entries = vec![
            Entry::debit(a, Money::from_minor(1000, Currency::USD)),
            Entry::credit(b, Money::from_minor(500, Currency::USD)),
        ];
        let r = Transaction::new_pending(lid(), 0, entries);
        match r {
            Err(Error::Unbalanced {
                currency,
                debits,
                credits,
            }) => {
                assert_eq!(currency, "USD");
                assert_eq!(debits, 1000);
                assert_eq!(credits, 500);
            }
            other => panic!("expected Unbalanced, got {other:?}"),
        }
    }

    #[test]
    fn multi_currency_balanced_per_currency() {
        // USD: 1000 debit, 1000 credit. EUR: 500 debit, 500 credit.
        let entries = vec![
            Entry::debit(aid(), Money::from_minor(1000, Currency::USD)),
            Entry::credit(aid(), Money::from_minor(1000, Currency::USD)),
            Entry::debit(aid(), Money::from_minor(500, Currency::EUR)),
            Entry::credit(aid(), Money::from_minor(500, Currency::EUR)),
        ];
        let r = Transaction::new_pending(lid(), 0, entries);
        assert!(r.is_ok());
    }

    #[test]
    fn multi_currency_unbalanced_in_one_currency_rejected() {
        // USD balanced; EUR not.
        let entries = vec![
            Entry::debit(aid(), Money::from_minor(1000, Currency::USD)),
            Entry::credit(aid(), Money::from_minor(1000, Currency::USD)),
            Entry::debit(aid(), Money::from_minor(500, Currency::EUR)),
            Entry::credit(aid(), Money::from_minor(400, Currency::EUR)),
        ];
        let r = Transaction::new_pending(lid(), 0, entries);
        assert!(matches!(r, Err(Error::Unbalanced { .. })));
    }

    #[test]
    fn cross_currency_no_invariant_across_currencies() {
        // USD entry alone — debits don't equal credits IN THAT currency
        // → rejected. The cross-currency case proves the invariant is
        // per-currency, not aggregate.
        let entries = vec![
            Entry::debit(aid(), Money::from_minor(100, Currency::USD)),
            Entry::credit(aid(), Money::from_minor(100, Currency::EUR)),
        ];
        let r = Transaction::new_pending(lid(), 0, entries);
        assert!(matches!(r, Err(Error::Unbalanced { .. })));
    }

    #[test]
    fn post_transitions_pending_to_posted() {
        let mut t = Transaction::new_pending(lid(), 0, balanced_pair()).unwrap();
        t.post().unwrap();
        assert_eq!(t.status, Status::Posted);
    }

    #[test]
    fn post_twice_rejected() {
        let mut t = Transaction::new_pending(lid(), 0, balanced_pair()).unwrap();
        t.post().unwrap();
        let r = t.post();
        assert!(matches!(r, Err(Error::TerminalState { .. })));
    }

    #[test]
    fn archive_then_post_rejected() {
        let mut t = Transaction::new_pending(lid(), 0, balanced_pair()).unwrap();
        t.archive().unwrap();
        let r = t.post();
        assert!(matches!(r, Err(Error::TerminalState { .. })));
    }

    #[test]
    fn new_posted_skips_pending() {
        let t = Transaction::new_posted(lid(), 0, balanced_pair()).unwrap();
        assert_eq!(t.status, Status::Posted);
    }

    #[test]
    fn external_id_round_trips() {
        let t = Transaction::new_pending(lid(), 0, balanced_pair())
            .unwrap()
            .with_external_id("ord-42");
        assert_eq!(t.external_id.as_deref(), Some("ord-42"));
    }

    #[test]
    fn metadata_accumulates() {
        let t = Transaction::new_pending(lid(), 0, balanced_pair())
            .unwrap()
            .with_metadata("k1", "v1")
            .with_metadata("k2", "v2");
        assert_eq!(t.metadata.len(), 2);
        assert_eq!(t.metadata[0], ("k1".into(), "v1".into()));
    }

    #[test]
    fn reversal_flips_each_entry_direction() {
        let original = Transaction::new_posted(lid(), 100, balanced_pair())
            .unwrap()
            .with_external_id("ord-1");
        let rev = Transaction::reversal_of(&original, 200).unwrap();

        assert_eq!(rev.status, Status::Pending);
        assert_eq!(rev.entries.len(), original.entries.len());
        for (orig, rev_entry) in original.entries.iter().zip(rev.entries.iter()) {
            assert_eq!(rev_entry.direction, orig.direction.opposite());
            assert_eq!(rev_entry.amount, orig.amount);
            assert_eq!(rev_entry.account_id, orig.account_id);
        }
    }

    #[test]
    fn reversal_external_id_suffix_avoids_collision() {
        let original = Transaction::new_posted(lid(), 0, balanced_pair())
            .unwrap()
            .with_external_id("ord-1");
        let rev = Transaction::reversal_of(&original, 0).unwrap();
        assert_eq!(rev.external_id.as_deref(), Some("ord-1:reversal"));
    }

    #[test]
    fn reversal_without_external_id_uses_id() {
        let original = Transaction::new_posted(lid(), 0, balanced_pair()).unwrap();
        let rev = Transaction::reversal_of(&original, 0).unwrap();
        assert_eq!(
            rev.external_id.as_deref(),
            Some(format!("{}:reversal", original.id).as_str())
        );
    }

    #[test]
    fn reversal_metadata_includes_parent() {
        let original = Transaction::new_posted(lid(), 0, balanced_pair()).unwrap();
        let rev = Transaction::reversal_of(&original, 0).unwrap();
        let reverses = rev
            .metadata
            .iter()
            .find(|(k, _)| k == "reverses")
            .expect("expected 'reverses' metadata");
        assert_eq!(reverses.1, original.id.to_string());
    }

    #[test]
    fn reversal_of_reversal_equals_original_directions() {
        let original = Transaction::new_posted(lid(), 0, balanced_pair()).unwrap();
        let rev = Transaction::reversal_of(&original, 0).unwrap();
        let rev_rev = Transaction::reversal_of(&rev, 0).unwrap();
        for (orig, rr) in original.entries.iter().zip(rev_rev.entries.iter()) {
            assert_eq!(rr.direction, orig.direction);
        }
    }

    #[test]
    fn transaction_id_display() {
        let u = Uuid::new_v4();
        let id = TransactionId::from_uuid(u);
        assert_eq!(format!("{id}"), u.to_string());
    }
}
