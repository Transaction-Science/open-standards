//! Sealed error type for the refund domain.

use thiserror::Error;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// A refund body failed an invariant check.
    #[error("invalid refund: {0}")]
    Invalid(String),

    /// The refund was not found in the store.
    #[error("refund not found: {0}")]
    NotFound(String),

    /// The requested state transition is illegal for the current
    /// state (e.g. trying to settle a Declined refund).
    #[error("invalid transition: refund {id} is in terminal state {state:?}")]
    InvalidTransition {
        /// Refund id.
        id: String,
        /// The state that blocked the transition.
        state: String,
    },

    /// An `external_id` was reused with a different body. The
    /// caller is asserting two different refunds under the same
    /// idempotency key.
    #[error("idempotency mismatch: external_id {0} reused with different body")]
    IdempotencyMismatch(String),

    /// The refund amount exceeded what's allowed (negative, or
    /// summing to more than the original transaction).
    #[error("refund amount {refund_minor} exceeds allowed {allowed_minor}")]
    AmountExceeded {
        /// The refund's amount in minor units.
        refund_minor: i64,
        /// The remaining refundable amount on the original tx.
        allowed_minor: i64,
    },

    /// Crossed an `op-core` error (currency, overflow).
    #[error(transparent)]
    Core(#[from] op_core::Error),
}
