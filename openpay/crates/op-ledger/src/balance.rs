//! Balance — what an account holds at a point in time.
//!
//! Three balance views per Modern Treasury convention:
//!
//! - **Posted** — sum of posted transactions only. The settled state.
//! - **Pending** — sum of posted AND pending transactions. The
//!   "balance after everything in flight settles" projection.
//! - **Available** — context-dependent; for OpenPay's reference
//!   model it equals `posted` (we don't model holds beyond pending).
//!
//! Balances are derived (never stored). They're computed by walking
//! the ledger's entries; the [`LedgerStore`](crate::LedgerStore)
//! does the iteration.
//!
//! ## Why a struct instead of `(i64, i64)`?
//!
//! Type-safety. `Balance::pending` and `Balance::posted` are clearly
//! labeled; you can't accidentally pass posted where pending is
//! expected. The struct also carries the currency so the caller
//! doesn't have to refetch the account to format the amount.

use op_core::{Currency, Money};

/// Balance for a single account at a point in time.
///
/// All three views carry the account's currency. They're always in
/// the SAME currency (one account → one currency, fixed at
/// creation).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Balance {
    /// Currency. Always matches the account's currency.
    pub currency: Currency,

    /// Sum of posted-transaction entries, signed by the account's
    /// normal balance so it's natural-positive for a healthy account.
    pub posted: Money,

    /// Sum of posted-and-pending entries, signed the same way.
    pub pending: Money,
}

impl Balance {
    /// Construct a zero balance in the given currency.
    #[must_use]
    pub fn zero(currency: Currency) -> Self {
        Self {
            currency,
            posted: Money::from_minor(0, currency),
            pending: Money::from_minor(0, currency),
        }
    }

    /// True if both views are zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.posted.is_zero() && self.pending.is_zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_constructs_zero_money_for_each_view() {
        let b = Balance::zero(Currency::USD);
        assert!(b.is_zero());
        assert_eq!(b.posted.minor_units, 0);
        assert_eq!(b.pending.minor_units, 0);
        assert_eq!(b.currency, Currency::USD);
    }

    #[test]
    fn non_zero_pending_is_not_zero() {
        let b = Balance {
            currency: Currency::USD,
            posted: Money::from_minor(0, Currency::USD),
            pending: Money::from_minor(100, Currency::USD),
        };
        assert!(!b.is_zero());
    }

    #[test]
    fn equality_requires_all_three_fields() {
        let a = Balance::zero(Currency::USD);
        let b = Balance::zero(Currency::USD);
        assert_eq!(a, b);
        let c = Balance::zero(Currency::EUR);
        assert_ne!(a, c);
    }
}
