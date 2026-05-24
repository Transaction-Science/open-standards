//! Account Information Service Provider (AISP) surface.
//!
//! Implements the vendor-neutral version of:
//!
//! - UK Open Banking R/W v3.1 § Accounts/Balances/Transactions APIs
//! - Berlin Group NextGenPSD2 § 5.2–5.6 (account-information service)
//! - STET PSD2 § Accounts / Balances / Transactions
//! - Australia CDR `cds-au/v1/banking/*`
//! - FDX `accounts`, `transactions`, `balances`
//! - SGFinDex `Accounts` aggregation
//!
//! The shapes here are the union of those six; the per-standard
//! binding modules ([`crate::uk_ob`], [`crate::berlin_group`], etc.)
//! decide which subset of fields the wire payload carries.

use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::fapi::OAuth2Token;

/// Opaque consent identifier minted by the ASPSP at consent creation.
///
/// UK OBIE: `Data.ConsentId`. Berlin Group: `consentId` header.
/// STET: `consentId`. CDR: `arrangementId`. FDX: `consentId`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConsentId(pub String);

/// Account type per ISO 20022 `CashAccountType4Code`, narrowed to the
/// values open-banking standards actually use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccountType {
    /// Personal current / checking account.
    Personal,
    /// Business current account.
    Business,
    /// Card (credit) account.
    Card,
    /// Loan / mortgage account.
    Loan,
    /// Savings / e-money account.
    Savings,
}

/// Refinement of [`AccountType`] used by FDX and some CDR fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccountSubtype {
    /// FDX `CHECKING` / OBIE `CurrentAccount`.
    Checking,
    /// FDX `SAVINGS` / OBIE `Savings`.
    Savings,
    /// FDX `MONEY_MARKET`.
    MoneyMarket,
    /// FDX `CD` (certificate of deposit).
    CertificateOfDeposit,
    /// FDX `CREDIT_CARD`.
    CreditCard,
    /// FDX `MORTGAGE`.
    Mortgage,
    /// FDX `LOAN` (non-mortgage).
    Loan,
    /// Anything outside the enumerated subtypes; carry the wire string.
    Other(String),
}

/// Balance type per ISO 20022 `BalanceType13Code`.
///
/// Open Banking standards expose a subset; this enum is that subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BalanceType {
    /// `ClosingBooked` — most-recent booked balance at the close
    /// of the previous business day.
    ClosingBooked,
    /// `InterimAvailable` — current available balance (booked +
    /// authorised holds).
    InterimAvailable,
    /// `Expected` — booked + anticipated forward dated items.
    Expected,
    /// `OpeningBooked` — opening booked balance for the current period.
    OpeningBooked,
}

/// Whether a balance / transaction line is a credit or debit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionCredit {
    /// Money into the account.
    Credit,
    /// Money out of the account.
    Debit,
}

/// Posting status of a transaction. UK OBIE distinguishes
/// `Booked` from `Pending`; Berlin Group adds `Information`
/// (informational entries that do not change the balance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionStatus {
    /// Posted to the booked balance.
    Booked,
    /// Authorised hold, not yet booked.
    Pending,
    /// Informational only (e.g. fee preview).
    Information,
}

/// A balance line for an account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Balance {
    /// Which balance flavour this line represents.
    pub balance_type: BalanceType,
    /// Signed amount, sign carried in [`Self::credit_debit`].
    pub amount: Money,
    /// Sign of the balance.
    pub credit_debit: TransactionCredit,
    /// RFC 3339 timestamp of the snapshot.
    pub as_of: time::OffsetDateTime,
}

/// A transaction line from an account's ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transaction {
    /// ASPSP-assigned transaction identifier.
    pub id: String,
    /// Posting status.
    pub status: TransactionStatus,
    /// Signed amount, sign in [`Self::credit_debit`].
    pub amount: Money,
    /// Credit or debit direction.
    pub credit_debit: TransactionCredit,
    /// RFC 3339 timestamp at which the transaction posted (or, for
    /// pending, was authorised).
    pub booking_date: time::OffsetDateTime,
    /// Operator-facing description / merchant name.
    pub description: String,
    /// Optional remittance information (free-text reference printed
    /// on the statement).
    pub remittance: Option<String>,
}

/// An account exposed via the AISP service.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Account {
    /// ASPSP-assigned, scoped to a single consent.
    pub id: String,
    /// IBAN / sort-code+account-no / routing+account, etc.
    /// The wire format is binding-specific; we carry the raw string.
    pub identifier: String,
    /// Display nickname or product name.
    pub nickname: Option<String>,
    /// Account type (current / savings / card / loan).
    pub account_type: AccountType,
    /// Refined subtype for FDX-style classifications.
    pub subtype: Option<AccountSubtype>,
    /// Currency the account is denominated in.
    pub currency: op_core::Currency,
}

/// AISP service trait — vendor-neutral account-information surface.
///
/// Implemented by each binding ([`crate::uk_ob`], [`crate::berlin_group`],
/// [`crate::stet`], [`crate::cdr`], [`crate::fdx`]). Each binding
/// translates the request into its standard's wire format and parses
/// the response back into these shapes.
///
/// **Async strategy.** This trait is intentionally synchronous: the
/// crate ships no HTTP client, and the bindings have no I/O of their
/// own. Operators wrap a real driver in an `async` wrapper at the
/// call site. Keeping the trait sync means it stays object-safe and
/// composable without an executor dependency.
pub trait AccountInfoService: Send + Sync {
    /// List the accounts a consent covers.
    fn accounts(&self, consent: &ConsentId, token: &OAuth2Token) -> Result<Vec<Account>>;

    /// Fetch the current balance(s) for an account.
    ///
    /// `account_id` is the ASPSP-scoped identifier returned by
    /// [`Self::accounts`].
    fn balances(
        &self,
        consent: &ConsentId,
        token: &OAuth2Token,
        account_id: &str,
    ) -> Result<Vec<Balance>>;

    /// Fetch transactions for an account within an optional time range.
    fn transactions(
        &self,
        consent: &ConsentId,
        token: &OAuth2Token,
        account_id: &str,
        from: Option<time::OffsetDateTime>,
        to: Option<time::OffsetDateTime>,
    ) -> Result<Vec<Transaction>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    #[test]
    fn balance_round_trips_through_serde() {
        let b = Balance {
            balance_type: BalanceType::InterimAvailable,
            amount: Money::from_minor(12_345, Currency::GBP),
            credit_debit: TransactionCredit::Credit,
            as_of: time::OffsetDateTime::UNIX_EPOCH,
        };
        let json = serde_json::to_string(&b).expect("ser");
        let back: Balance = serde_json::from_str(&json).expect("de");
        assert_eq!(b, back);
    }

    #[test]
    fn account_subtype_other_carries_string() {
        let s = AccountSubtype::Other("LINE_OF_CREDIT".into());
        let json = serde_json::to_string(&s).expect("ser");
        assert!(json.contains("LINE_OF_CREDIT"));
    }
}
