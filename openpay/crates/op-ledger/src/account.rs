//! Ledger accounts.
//!
//! An account is the unit of balance in the ledger. Examples for an
//! OpenPay deployment:
//!
//! - `merchant_receivable` (asset, debit-normal) — money customers
//!   owe the merchant from approved-but-not-yet-settled card auths.
//! - `cash` (asset, debit-normal) — money in the merchant's bank
//!   account.
//! - `psp_payable` (liability, credit-normal) — money the merchant
//!   owes the PSP for processing fees.
//! - `revenue` (revenue, credit-normal) — money earned.
//! - `refunds` (contra-revenue / expense, debit-normal) — money
//!   refunded to customers.
//!
//! Operators choose the chart of accounts that matches their
//! business — we don't prescribe one. The crate just enforces that
//! transactions balance and that entries respect each account's
//! currency.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use op_core::Currency;

/// Opaque account id. UUID v4 under the hood.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AccountId(pub Uuid);

impl AccountId {
    /// Generate a fresh id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Wrap an existing UUID (useful for tests + replay).
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

impl Default for AccountId {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for AccountId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// Which side of the double-entry equation increases this account.
///
/// In accounting terms:
///
/// - **Asset, Expense** accounts are **debit-normal** — debits
///   increase them, credits decrease them.
/// - **Liability, Equity, Revenue** accounts are **credit-normal** —
///   credits increase them, debits decrease them.
///
/// This affects only the balance derivation: the ledger computes
/// `(sum of credits − sum of debits)` for credit-normal accounts,
/// and `(sum of debits − sum of credits)` for debit-normal
/// accounts. Either way the resulting "balance" is the natural
/// positive number you'd report.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormalBalance {
    /// Debits increase the account (assets, expenses).
    Debit,
    /// Credits increase the account (liabilities, equity, revenue).
    Credit,
}

/// Accounting classification, informational only.
///
/// We carry this so operators can group accounts by class for
/// reporting, but the ledger itself doesn't enforce GAAP / IFRS
/// rules — that's the operator's job downstream. The
/// `NormalBalance` for each class is the conventional one and the
/// crate uses it as a sensible default if the operator doesn't
/// override.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccountClass {
    /// Things the entity owns. Debit-normal.
    Asset,
    /// Things the entity owes. Credit-normal.
    Liability,
    /// Owner's stake. Credit-normal.
    Equity,
    /// Income from operations. Credit-normal.
    Revenue,
    /// Cost of operations. Debit-normal.
    Expense,
}

impl AccountClass {
    /// Conventional normal balance for this class.
    #[must_use]
    pub const fn conventional_normal_balance(self) -> NormalBalance {
        match self {
            Self::Asset | Self::Expense => NormalBalance::Debit,
            Self::Liability | Self::Equity | Self::Revenue => NormalBalance::Credit,
        }
    }
}

/// A single account in a ledger.
///
/// Once constructed, the `currency` and `normal_balance` are fixed.
/// The `name` and `metadata` can be edited (but doing so is a
/// caller-side concern; the store interface doesn't include
/// edit-in-place because rename-and-archive is the auditable path).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Account {
    /// Stable id.
    pub id: AccountId,

    /// Which ledger this account belongs to. Entries against this
    /// account are only valid in transactions posted to the same
    /// ledger.
    pub ledger_id: crate::ledger::LedgerId,

    /// Human-readable name (e.g. `"merchant_receivable_usd"`).
    pub name: String,

    /// Conventional class. Informational.
    pub class: AccountClass,

    /// Which side increases the balance.
    pub normal_balance: NormalBalance,

    /// ISO 4217 currency. Fixed at creation.
    pub currency: Currency,

    /// Optional external reference (the operator's own id for this
    /// account, e.g. a database row id).
    pub external_id: Option<String>,
}

impl Account {
    /// Construct with the conventional `NormalBalance` for the given
    /// class.
    #[must_use]
    pub fn new(
        ledger_id: crate::ledger::LedgerId,
        name: impl Into<String>,
        class: AccountClass,
        currency: Currency,
    ) -> Self {
        Self {
            id: AccountId::new(),
            ledger_id,
            name: name.into(),
            class,
            normal_balance: class.conventional_normal_balance(),
            currency,
            external_id: None,
        }
    }

    /// Builder: set an external id.
    #[must_use]
    pub fn with_external_id(mut self, id: impl Into<String>) -> Self {
        self.external_id = Some(id.into());
        self
    }

    /// Builder: override the `NormalBalance` (rare — only for
    /// non-conventional accounts like contra-asset).
    #[must_use]
    pub fn with_normal_balance(mut self, nb: NormalBalance) -> Self {
        self.normal_balance = nb;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::LedgerId;

    fn lid() -> LedgerId {
        LedgerId::new()
    }

    #[test]
    fn conventional_normal_balances() {
        assert_eq!(
            AccountClass::Asset.conventional_normal_balance(),
            NormalBalance::Debit
        );
        assert_eq!(
            AccountClass::Expense.conventional_normal_balance(),
            NormalBalance::Debit
        );
        assert_eq!(
            AccountClass::Liability.conventional_normal_balance(),
            NormalBalance::Credit
        );
        assert_eq!(
            AccountClass::Equity.conventional_normal_balance(),
            NormalBalance::Credit
        );
        assert_eq!(
            AccountClass::Revenue.conventional_normal_balance(),
            NormalBalance::Credit
        );
    }

    #[test]
    fn new_uses_class_normal_balance() {
        let a = Account::new(lid(), "cash", AccountClass::Asset, Currency::USD);
        assert_eq!(a.normal_balance, NormalBalance::Debit);
        let r = Account::new(lid(), "revenue", AccountClass::Revenue, Currency::USD);
        assert_eq!(r.normal_balance, NormalBalance::Credit);
    }

    #[test]
    fn override_normal_balance() {
        // Contra-asset accounts (allowance for doubtful receivables)
        // have credit-normal even though they're conventionally
        // classified Asset.
        let a = Account::new(lid(), "allowance", AccountClass::Asset, Currency::USD)
            .with_normal_balance(NormalBalance::Credit);
        assert_eq!(a.normal_balance, NormalBalance::Credit);
        assert_eq!(a.class, AccountClass::Asset);
    }

    #[test]
    fn external_id_builder() {
        let a = Account::new(lid(), "cash", AccountClass::Asset, Currency::USD)
            .with_external_id("db-row-42");
        assert_eq!(a.external_id.as_deref(), Some("db-row-42"));
    }

    #[test]
    fn fresh_accounts_have_distinct_ids() {
        let lid = lid();
        let a = Account::new(lid, "a", AccountClass::Asset, Currency::USD);
        let b = Account::new(lid, "b", AccountClass::Asset, Currency::USD);
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn account_id_round_trip_via_uuid() {
        let u = Uuid::new_v4();
        let id = AccountId::from_uuid(u);
        assert_eq!(id.as_uuid(), u);
    }

    #[test]
    fn account_id_display_is_uuid_string() {
        let u = Uuid::new_v4();
        let id = AccountId::from_uuid(u);
        assert_eq!(format!("{id}"), u.to_string());
    }
}
