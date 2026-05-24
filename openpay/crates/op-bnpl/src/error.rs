//! Sealed error enum.
//!
//! One variant per failure class. `Clone` so callers can fan a single
//! error out to multiple sinks (retry bookkeeping, telemetry, ledger)
//! without cloning the wrapped causes by hand.

use thiserror::Error;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes for BNPL acceptance.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// HTTP transport failure (DNS, TLS, connection, timeout).
    #[error("transport error: {0}")]
    Transport(String),

    /// Provider returned a non-2xx HTTP status with a parsed body.
    #[error("BNPL provider rejected: status={status} code={code}: {message}")]
    ProviderRejected {
        /// HTTP status code.
        status: u16,
        /// Provider-specific error code (or `"unknown"` if absent).
        code: String,
        /// Human-readable message from the provider.
        message: String,
    },

    /// Response was structurally valid JSON but a required field was
    /// missing.
    #[error("response missing field: {0}")]
    MissingField(&'static str),

    /// JSON parse failure on a provider response.
    #[error("response parse failed: {0}")]
    Parse(String),

    /// The caller passed an instalment plan or amount that the provider
    /// will reject before the network round-trip (e.g. zero amount,
    /// negative line item, mismatched currency).
    #[error("invalid intent: {0}")]
    InvalidIntent(String),

    /// Webhook signature verification failed.
    #[error("invalid webhook signature")]
    InvalidSignature,

    /// Webhook header was missing or malformed.
    #[error("malformed webhook header: {0}")]
    MalformedSignatureHeader(String),

    /// Idempotency key collided with a different request body.
    #[error("idempotency key reused with mismatched body")]
    IdempotencyMismatch,

    /// Consumer was deemed ineligible (geo, age, credit, amount band).
    #[error("consumer ineligible: {0}")]
    Ineligible(String),

    /// Underlying op-core domain error (currency mismatch, overflow).
    #[error(transparent)]
    Core(#[from] op_core::Error),
}
