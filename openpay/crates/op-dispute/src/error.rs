//! Sealed error type for the dispute domain.

use thiserror::Error;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// Body failed an invariant check.
    #[error("invalid dispute: {0}")]
    Invalid(String),

    /// Dispute not found.
    #[error("dispute not found: {0}")]
    NotFound(String),

    /// Illegal state transition (e.g. submitting evidence after the
    /// dispute is already resolved).
    #[error("invalid transition: dispute {id} is in state {state:?}")]
    InvalidTransition {
        /// Dispute id.
        id: String,
        /// State that blocked the transition.
        state: String,
    },

    /// Idempotency token reused with a different body.
    #[error("idempotency mismatch: external_id {0} reused with different body")]
    IdempotencyMismatch(String),

    /// `op-core` error pass-through.
    #[error(transparent)]
    Core(#[from] op_core::Error),
}
