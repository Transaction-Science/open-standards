//! Ledger entries — the individual debit / credit rows.
//!
//! Every transaction has 2+ entries. An entry says "move `amount`
//! to/from `account_id` on the `direction` side." The transaction's
//! double-entry invariant says debits == credits per currency.

use serde::{Deserialize, Serialize};

use op_core::Money;

use crate::account::AccountId;

/// Which side of double-entry this entry goes on.
///
/// Distinct from [`NormalBalance`](crate::NormalBalance):
/// `Direction` describes a single entry's side; `NormalBalance`
/// describes which side increases the account.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Direction {
    /// Left side of double-entry.
    Debit,
    /// Right side of double-entry.
    Credit,
}

impl Direction {
    /// The opposite direction. Used for reversals.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Debit => Self::Credit,
            Self::Credit => Self::Debit,
        }
    }

    /// True if this direction increases an account with the given
    /// normal balance. (Debit-normal accounts grow on debits;
    /// credit-normal accounts grow on credits.)
    #[must_use]
    pub fn increases(self, normal: crate::account::NormalBalance) -> bool {
        matches!(
            (self, normal),
            (Self::Debit, crate::account::NormalBalance::Debit)
                | (Self::Credit, crate::account::NormalBalance::Credit)
        )
    }
}

/// A single debit or credit entry.
///
/// `amount` must be positive (zero entries are pointless and likely
/// programmer error; we accept them in the constructor for
/// flexibility but the balanced-transaction invariant rejects
/// zero-only transactions).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    /// Account this entry affects.
    pub account_id: AccountId,

    /// Direction (debit or credit).
    pub direction: Direction,

    /// Amount + currency. Currency must match the account's
    /// currency (verified when the transaction is created).
    pub amount: Money,
}

impl Entry {
    /// Construct.
    #[must_use]
    pub fn new(account_id: AccountId, direction: Direction, amount: Money) -> Self {
        Self {
            account_id,
            direction,
            amount,
        }
    }

    /// Convenience: a debit entry.
    #[must_use]
    pub fn debit(account_id: AccountId, amount: Money) -> Self {
        Self::new(account_id, Direction::Debit, amount)
    }

    /// Convenience: a credit entry.
    #[must_use]
    pub fn credit(account_id: AccountId, amount: Money) -> Self {
        Self::new(account_id, Direction::Credit, amount)
    }

    /// Construct the entry that reverses this one.
    ///
    /// Used by [`Transaction::reversal_of`](crate::transaction::Transaction::reversal_of).
    #[must_use]
    pub fn reverse(&self) -> Self {
        Self {
            account_id: self.account_id,
            direction: self.direction.opposite(),
            amount: self.amount,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account::NormalBalance;
    use op_core::Currency;

    fn aid() -> AccountId {
        AccountId::new()
    }

    #[test]
    fn opposite_swaps_direction() {
        assert_eq!(Direction::Debit.opposite(), Direction::Credit);
        assert_eq!(Direction::Credit.opposite(), Direction::Debit);
        assert_eq!(Direction::Debit.opposite().opposite(), Direction::Debit);
    }

    #[test]
    fn increases_for_debit_normal_account() {
        assert!(Direction::Debit.increases(NormalBalance::Debit));
        assert!(!Direction::Credit.increases(NormalBalance::Debit));
    }

    #[test]
    fn increases_for_credit_normal_account() {
        assert!(Direction::Credit.increases(NormalBalance::Credit));
        assert!(!Direction::Debit.increases(NormalBalance::Credit));
    }

    #[test]
    fn debit_helper() {
        let a = aid();
        let m = Money::from_minor(100, Currency::USD);
        let e = Entry::debit(a, m);
        assert_eq!(e.direction, Direction::Debit);
        assert_eq!(e.account_id, a);
        assert_eq!(e.amount.minor_units, 100);
    }

    #[test]
    fn credit_helper() {
        let a = aid();
        let m = Money::from_minor(100, Currency::USD);
        let e = Entry::credit(a, m);
        assert_eq!(e.direction, Direction::Credit);
    }

    #[test]
    fn reverse_swaps_direction_preserves_amount_and_account() {
        let a = aid();
        let m = Money::from_minor(500, Currency::EUR);
        let e = Entry::debit(a, m);
        let r = e.reverse();
        assert_eq!(r.direction, Direction::Credit);
        assert_eq!(r.account_id, a);
        assert_eq!(r.amount, m);
    }

    #[test]
    fn double_reverse_is_identity() {
        let e = Entry::debit(aid(), Money::from_minor(100, Currency::USD));
        let rr = e.reverse().reverse();
        assert_eq!(rr.direction, e.direction);
        assert_eq!(rr.amount, e.amount);
        assert_eq!(rr.account_id, e.account_id);
    }
}
