//! Error type for `op-tax`. One sealed enum, exhaustive `match` at the callsite.

use thiserror::Error;

/// Result alias for `op-tax`.
pub type Result<T> = core::result::Result<T, Error>;

/// All possible failure modes inside `op-tax`.
///
/// The variant set is intentionally broad: tax APIs can fail in many
/// ways (missing rate, bad jurisdiction, vendor outage, transport,
/// parse). We collapse vendor-specific error codes into the four
/// transport / decode / vendor / domain buckets below.
#[derive(Debug, Error)]
pub enum Error {
    /// The requested jurisdiction has no rate on file for the given
    /// product category and date.
    #[error("no rate on file: jurisdiction={jurisdiction:?}, category={category:?}")]
    NoRate {
        /// Stringified jurisdiction.
        jurisdiction: String,
        /// Stringified product category.
        category: String,
    },

    /// Math overflow on a tax calculation. Should be unreachable
    /// at real-world transaction sizes — Decimal carries ~28 digits.
    #[error("arithmetic overflow in tax math")]
    Overflow,

    /// `op-core` Money rejected the operation (currency mismatch,
    /// overflow on minor units, etc.).
    #[error("money error: {0}")]
    Money(#[from] op_core::Error),

    /// HTTP / network transport failure talking to a commercial
    /// backend.
    #[error("transport error: {0}")]
    Transport(String),

    /// The commercial backend returned a non-success status. The
    /// inner string is the vendor's error payload, untouched.
    #[error("vendor error: status={status}, body={body}")]
    Vendor {
        /// HTTP status code returned by the vendor.
        status: u16,
        /// Raw response body from the vendor (already redacted of
        /// any credentials by the adapter).
        body: String,
    },

    /// JSON encoding or decoding failed.
    #[error("encode/decode error: {0}")]
    Codec(String),

    /// On-disk snapshot couldn't be loaded.
    #[error("snapshot load error: {0}")]
    Snapshot(String),

    /// The exemption certificate is expired or out of scope.
    #[error("invalid exemption certificate: {0}")]
    InvalidExemption(String),

    /// Calculator-specific configuration error (e.g. missing API key).
    #[error("configuration error: {0}")]
    Config(String),
}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Self::Transport(e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Codec(e.to_string())
    }
}
