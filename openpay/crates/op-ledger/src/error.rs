//! Ledger error type.
//!
//! Distinct from `op-core::Error` and `op-orchestrator::Error` —
//! ledger violations have their own taxonomy. Most are
//! invariant-preservation errors (the ledger refuses an operation
//! that would corrupt its state); a few are lookup errors.

use thiserror::Error;

/// Crate-local result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// All failure modes for ledger operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    /// The unbalanced-entries check failed for a transaction. The
    /// sum of debits did not equal the sum of credits for the named
    /// currency.
    #[error("transaction unbalanced for {currency}: debits={debits} credits={credits}")]
    Unbalanced {
        /// ISO 4217 currency code where the imbalance occurred.
        currency: String,
        /// Total debit amount in minor units.
        debits: i64,
        /// Total credit amount in minor units.
        credits: i64,
    },

    /// Transaction was submitted with fewer than two entries — a
    /// single-entry transaction cannot satisfy double-entry.
    #[error("transaction must have at least two entries (got {0})")]
    TooFewEntries(usize),

    /// An entry's currency did not match the account's currency.
    #[error(
        "entry currency {entry_currency} does not match account currency {account_currency} for account {account_id}"
    )]
    CurrencyMismatch {
        /// The currency the entry was denominated in.
        entry_currency: String,
        /// The currency the account was created with.
        account_currency: String,
        /// The mismatching account's id.
        account_id: String,
    },

    /// An entry referenced an account that doesn't exist in this
    /// ledger.
    #[error("account not found: {0}")]
    AccountNotFound(String),

    /// An entry referenced an account in a different ledger.
    #[error("account {account_id} belongs to ledger {account_ledger}, not {expected_ledger}")]
    CrossLedgerEntry {
        /// The account that was referenced.
        account_id: String,
        /// The ledger the account actually belongs to.
        account_ledger: String,
        /// The ledger the transaction was being posted to.
        expected_ledger: String,
    },

    /// A transaction with this `external_id` was posted before, with
    /// a different body. Per Adyen / Stripe convention this is a
    /// 409-style error — never resolved by retry. The caller must
    /// pick a new `external_id`.
    #[error("external_id reused with a different transaction body")]
    IdempotencyMismatch,

    /// Attempted to modify a transaction in a state that doesn't
    /// permit modification. Posted transactions are immutable;
    /// archived transactions can be reversed but not edited.
    #[error("transaction {id} is in terminal state ({state:?}); modification refused")]
    TerminalState {
        /// The transaction id.
        id: String,
        /// The state it is in.
        state: crate::transaction::Status,
    },

    /// Attempted to look up a transaction that doesn't exist.
    #[error("transaction not found: {0}")]
    TransactionNotFound(String),

    /// Attempted to look up a ledger that doesn't exist.
    #[error("ledger not found: {0}")]
    LedgerNotFound(String),

    /// Invalid input to a ledger constructor (empty name, etc.).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Inner core error (typically a `Money` overflow during balance
    /// aggregation).
    #[error("core error")]
    Core(#[from] op_core::Error),
}
