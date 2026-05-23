//! Sealed error type.

use thiserror::Error;

/// Crate `Result` alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// Caller-supplied input violated a precondition.
    #[error("invalid: {0}")]
    Invalid(String),

    /// Lookup missed.
    #[error("subscription not found: {0}")]
    NotFound(String),

    /// State-machine transition refused.
    #[error("invalid transition: subscription {id} is in {state}")]
    InvalidTransition {
        /// Subscription id.
        id: String,
        /// Current state.
        state: String,
    },

    /// Idempotency replay with mismatched body.
    #[error("idempotency mismatch on external_id `{0}`")]
    IdempotencyMismatch(String),

    /// Currency mismatch between plan amount and an operation
    /// (e.g. proration credit in a different currency).
    #[error("currency mismatch: {a} vs {b}")]
    CurrencyMismatch {
        /// One side.
        a: String,
        /// Other side.
        b: String,
    },

    /// Forwarded `op-core` error.
    #[error(transparent)]
    Core(#[from] op_core::Error),
}
