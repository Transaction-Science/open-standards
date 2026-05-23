//! Sealed error type for webhook operations.

use thiserror::Error;

/// Crate-local result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// All failure modes for webhook operations.
#[derive(Debug, Error)]
pub enum Error {
    /// The endpoint id wasn't found in the store.
    #[error("endpoint not found: {0}")]
    EndpointNotFound(String),

    /// The event id wasn't found in the store.
    #[error("event not found: {0}")]
    EventNotFound(String),

    /// The delivery attempt id wasn't found in the store.
    #[error("delivery attempt not found: {0}")]
    AttemptNotFound(String),

    /// HTTP transport returned an error (network failure, DNS, TLS,
    /// etc.). These are retryable.
    #[error("transport error: {0}")]
    Transport(String),

    /// The endpoint URL failed validation (not http(s), malformed,
    /// etc.).
    #[error("invalid endpoint URL: {0}")]
    InvalidUrl(String),

    /// The endpoint is disabled (auto-disabled after chronic
    /// failures, or operator-disabled). Re-enable before
    /// dispatching.
    #[error("endpoint {0} is disabled")]
    EndpointDisabled(String),

    /// The retry budget is exhausted. Either max_attempts or
    /// max_age_secs has been reached. The attempt is recorded as
    /// [`DeliveryStatus::Failed`](crate::DeliveryStatus::Failed).
    #[error("retry budget exhausted after {attempts} attempts")]
    RetryExhausted {
        /// Number of attempts made.
        attempts: u32,
    },

    /// Signature verification (used by the verifier side, e.g. a
    /// merchant verifying our payloads) failed. Caller cannot
    /// distinguish from "valid but truncated" — both fail closed.
    #[error("signature verification failed")]
    SignatureMismatch,

    /// The signature header was missing, malformed, or used an
    /// unknown scheme.
    #[error("malformed signature header: {0}")]
    MalformedSignature(String),

    /// The signed timestamp is outside the tolerance window
    /// (default 5 minutes). Older than tolerance = possible replay
    /// attack.
    #[error("signature timestamp outside tolerance window (delta={delta_secs}s)")]
    TimestampOutOfTolerance {
        /// Absolute difference between signed timestamp and now.
        delta_secs: i64,
    },

    /// Invalid input (empty secret, zero max_attempts, etc.).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Inner core error.
    #[error("core error")]
    Core(#[from] op_core::Error),
}
