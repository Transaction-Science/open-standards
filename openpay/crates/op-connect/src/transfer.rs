//! Internal-ledger transfers between connected accounts.
//!
//! No payment-rail involvement; this is purely a ledger entry move. The
//! classic use cases:
//!
//! - Marketplace re-allocating an already-collected payout among
//!   sub-merchants after a dispute settlement.
//! - Platform refunding a fee leg back to a sub-merchant.
//! - Reserve releases at the end of a holdback period (see
//!   [`crate::payout::PayoutSchedule::reserve_pct`]).
//!
//! The ledger here is a minimal local model — not the full `op-ledger`
//! double-entry machine. Operators wire to `op-ledger` in production;
//! this surface keeps op-connect free-standing for tests and kiosk
//! deployments. The two shapes are 1:1 compatible at the entry level.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use op_core::{Currency, Money};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::account::AccountId;
use crate::error::{Error, Result};

/// Strongly-typed transfer identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransferId(pub String);

impl TransferId {
    /// Mint a fresh `trsf_<uuidv4>` identifier.
    #[must_use]
    pub fn new() -> Self {
        Self(format!("trsf_{}", Uuid::new_v4().simple()))
    }
}

impl Default for TransferId {
    fn default() -> Self {
        Self::new()
    }
}

/// A single ledger entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// Unique identifier.
    pub id: TransferId,
    /// Source account (debited).
    pub from: AccountId,
    /// Destination account (credited).
    pub to: AccountId,
    /// Amount transferred.
    pub amount: Money,
    /// Posting time.
    pub posted_at: DateTime<Utc>,
}

/// Minimal connected-account ledger.
///
/// Tracks per-account balances and an append-only entry log. Production
/// callers replace this with `op-ledger` (which adds double-entry
/// transactions, pending vs posted, account classes, etc.); this
/// surface stays in op-connect so the crate's tests don't require an
/// op-ledger dependency loop.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ledger {
    balances: BTreeMap<AccountId, Money>,
    entries: Vec<LedgerEntry>,
}

impl Ledger {
    /// Fresh empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed an account with an opening balance.
    pub fn credit_opening(&mut self, acct: &AccountId, amount: Money) {
        self.balances
            .entry(acct.clone())
            .and_modify(|m| {
                if let Ok(sum) = m.checked_add(amount) {
                    *m = sum;
                }
            })
            .or_insert(amount);
    }

    /// Read balance for an account in the given currency.
    ///
    /// Returns zero (in `currency`) if the account is unknown or if the
    /// recorded balance is in a different currency.
    #[must_use]
    pub fn balance(&self, acct: &AccountId, currency: Currency) -> Money {
        self.balances
            .get(acct)
            .copied()
            .filter(|m| m.currency == currency)
            .unwrap_or_else(|| Money::zero(currency))
    }

    /// Number of posted entries.
    #[must_use]
    pub fn entries_len(&self) -> usize {
        self.entries.len()
    }

    /// Iterate over the entry log.
    pub fn entries(&self) -> impl Iterator<Item = &LedgerEntry> {
        self.entries.iter()
    }
}

/// Move funds from one connected account to another.
///
/// Pure ledger entry — no rail submission. The source account must
/// hold sufficient balance in the requested currency, and both
/// accounts must already exist in the ledger (call
/// [`Ledger::credit_opening`] to seed).
///
/// # Errors
/// - [`Error::AccountNotFound`] if `from` has no balance entry.
/// - [`Error::CurrencyMismatch`] if balances are in a different currency.
/// - [`Error::InvalidSplit`] (re-used for negative-balance protection)
///   if the source would go negative.
pub fn transfer(
    from: &AccountId,
    to: &AccountId,
    amount: Money,
    ledger: &mut Ledger,
) -> Result<TransferId> {
    if amount.minor_units < 0 {
        return Err(Error::InvalidSplit {
            reason: "transfer amount must be non-negative".into(),
        });
    }

    let from_balance = ledger
        .balances
        .get(from)
        .copied()
        .ok_or_else(|| Error::AccountNotFound(from.0.clone()))?;
    if from_balance.currency != amount.currency {
        return Err(Error::CurrencyMismatch(format!(
            "source account currency {} != transfer currency {}",
            from_balance.currency, amount.currency
        )));
    }
    let new_from = from_balance.checked_sub(amount).map_err(|_| Error::Overflow)?;
    if new_from.minor_units < 0 {
        return Err(Error::InvalidSplit {
            reason: format!("insufficient balance in {}", from.0),
        });
    }

    let to_balance = ledger
        .balances
        .get(to)
        .copied()
        .unwrap_or_else(|| Money::zero(amount.currency));
    if to_balance.currency != amount.currency {
        return Err(Error::CurrencyMismatch(format!(
            "destination account currency {} != transfer currency {}",
            to_balance.currency, amount.currency
        )));
    }
    let new_to = to_balance.checked_add(amount).map_err(|_| Error::Overflow)?;

    ledger.balances.insert(from.clone(), new_from);
    ledger.balances.insert(to.clone(), new_to);

    let entry = LedgerEntry {
        id: TransferId::new(),
        from: from.clone(),
        to: to.clone(),
        amount,
        posted_at: Utc::now(),
    };
    let id = entry.id.clone();
    ledger.entries.push(entry);
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_transfer_moves_balance() {
        let a = AccountId("acct_a".into());
        let b = AccountId("acct_b".into());
        let mut led = Ledger::new();
        led.credit_opening(&a, Money::from_minor(10_000, Currency::USD));
        led.credit_opening(&b, Money::from_minor(0, Currency::USD));

        transfer(&a, &b, Money::from_minor(2_500, Currency::USD), &mut led)
            .expect("ok");
        assert_eq!(
            led.balance(&a, Currency::USD),
            Money::from_minor(7_500, Currency::USD)
        );
        assert_eq!(
            led.balance(&b, Currency::USD),
            Money::from_minor(2_500, Currency::USD)
        );
        assert_eq!(led.entries_len(), 1);
    }

    #[test]
    fn insufficient_balance_blocks() {
        let a = AccountId("acct_a".into());
        let b = AccountId("acct_b".into());
        let mut led = Ledger::new();
        led.credit_opening(&a, Money::from_minor(100, Currency::USD));
        led.credit_opening(&b, Money::from_minor(0, Currency::USD));

        let err = transfer(&a, &b, Money::from_minor(200, Currency::USD), &mut led)
            .expect_err("insufficient");
        assert!(matches!(err, Error::InvalidSplit { .. }));
    }
}
