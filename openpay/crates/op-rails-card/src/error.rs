//! Sealed error type. One variant per failure class.

use thiserror::Error;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes for card-rail operations.
///
/// `Clone` mirrors `op_core::Error` so callers can fan an error out to
/// multiple sinks (retry bookkeeping, telemetry, test doubles) without
/// reconstructing it.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// HTTP transport failure (DNS, TLS, connection, timeout).
    #[error("transport error: {0}")]
    Transport(String),

    /// PSP returned a non-2xx HTTP status with a machine-readable body.
    #[error("PSP returned status {status}: {code}: {message}")]
    PspRejected {
        /// HTTP status code.
        status: u16,
        /// PSP-specific error code.
        code: String,
        /// Human-readable message.
        message: String,
    },

    /// JSON response was structurally valid but missing a required field.
    #[error("response missing field: {0}")]
    MissingField(&'static str),

    /// JSON parse failure on a PSP response.
    #[error("response parse failed: {0}")]
    Parse(String),

    /// The PSP returned an authorization status we don't recognize.
    /// New PSPs are added regularly; an unknown status is treated as a
    /// soft failure, not a panic.
    #[error("unknown PSP status: {0}")]
    UnknownStatus(String),

    /// The caller passed a `PaymentMethod` variant the active driver
    /// doesn't support (e.g. A2A given to a card driver).
    #[error("unsupported payment method for card rail")]
    UnsupportedMethod,

    /// Currency / amount validation failed in `op-core`.
    #[error(transparent)]
    Core(#[from] op_core::Error),

    /// The driver refused the request before sending it (e.g. amount
    /// outside the PSP's allowed range, missing capability).
    #[error("driver-side validation: {0}")]
    DriverValidation(String),
}
