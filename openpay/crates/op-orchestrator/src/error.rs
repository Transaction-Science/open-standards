//! Orchestrator error type.
//!
//! Distinct from `op-core::Error` — orchestration failures have
//! their own taxonomy: routing produced no eligible rail, every
//! rail failed terminally, fraud declined, etc.
//!
//! Inner errors from the layered crates are preserved via `#[from]`
//! so callers can pattern-match all the way down if they need to,
//! but typical use is `match err { Error::FraudDeclined(_) => ... }`.

use thiserror::Error;

/// Crate-local result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Orchestration-level error.
#[derive(Debug, Error)]
pub enum Error {
    /// Fraud scorer rejected the payment.
    #[error("fraud declined: {reason}")]
    FraudDeclined {
        /// Short reason code from the scorer (e.g. `"velocity"`,
        /// `"high-risk-bin"`).
        reason: String,
    },

    /// Fraud scorer flagged for human review. Orchestrator does not
    /// proceed; caller must handle.
    #[error("fraud review required: {reason}")]
    FraudReviewRequired {
        /// Reason code.
        reason: String,
    },

    /// Router could not produce any eligible rail for the intent.
    #[error("no eligible rail: {reason}")]
    NoEligibleRail {
        /// Why no rail was eligible (e.g. `"amount-too-large for any rail"`,
        /// `"country not supported"`).
        reason: String,
    },

    /// Every rail tried failed with a terminal (non-retriable) error.
    /// The vector lists the attempts in order.
    #[error("all rails exhausted after {attempt_count} attempts")]
    AllRailsExhausted {
        /// Number of attempts made.
        attempt_count: usize,
    },

    /// The idempotency key has been seen before, but with a
    /// different request body. Per Adyen / Stripe practice, this is
    /// a 409 / conflict — never resolved by retry.
    #[error("idempotency key reused with different request body")]
    IdempotencyMismatch,

    /// Circuit breaker is open for every eligible rail. Caller
    /// should back off and retry later.
    #[error("all eligible rails have open circuit breakers")]
    AllCircuitsOpen,

    /// Inner vault error (e.g. token expired, lookup failed).
    #[error("vault error")]
    Vault(#[from] op_vault::Error),

    /// Inner fraud-scoring error (model load failure, NaN score).
    #[error("fraud error")]
    Fraud(#[from] op_fraud::Error),

    /// Inner core error (Money overflow, RailKind/PaymentMethod
    /// mismatch).
    #[error("core error")]
    Core(#[from] op_core::Error),
}
