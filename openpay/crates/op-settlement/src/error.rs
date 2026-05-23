//! Settlement error taxonomy.

use thiserror::Error;

/// Sealed error type for the crate.
#[derive(Debug, Error)]
pub enum Error {
    /// Caller-supplied input violated a precondition.
    #[error("invalid: {0}")]
    Invalid(String),

    /// Lookup by id missed.
    #[error("not found: {0}")]
    NotFound(String),

    /// Tried a batch lifecycle transition the state machine rejects
    /// (`Closed → Open`, `Paid → Paying`, etc.).
    #[error("invalid batch transition: {from} → {to}")]
    InvalidTransition {
        /// Current status.
        from: String,
        /// Attempted next status.
        to: String,
    },

    /// Idempotency replay where the body changed.
    #[error("idempotency mismatch on external id `{0}`")]
    IdempotencyMismatch(String),

    /// Currency mismatch when adding a tx to a batch.
    #[error("currency mismatch: batch is {batch}, tx is {tx}")]
    CurrencyMismatch {
        /// The batch's currency code.
        batch: String,
        /// The transaction's currency code.
        tx: String,
    },

    /// Batch was empty when payout-file generation was attempted.
    #[error("batch has no entries; nothing to pay out")]
    EmptyBatch,

    /// Bubbled-up core error (overflow, money arithmetic).
    #[error(transparent)]
    Core(#[from] op_core::Error),

    /// Bubbled-up iso20022 codec / builder error.
    #[error(transparent)]
    Iso20022(#[from] op_iso20022::Error),
}

/// `Result<T, Error>` shorthand.
pub type Result<T> = core::result::Result<T, Error>;
