//! Sealed error type for `op-crypto-extra`.

use thiserror::Error;

/// Crate `Result` alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Failure modes for extended crypto-rail primitives.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// Hex / bech32 / base58 / base64 decode failure.
    #[error("decode: {0}")]
    Decode(String),

    /// Encoded value's length or shape doesn't match the spec.
    #[error("invalid layout: {0}")]
    InvalidLayout(String),

    /// A required field is missing or the wrong type.
    #[error("missing required field: {0}")]
    MissingField(&'static str),

    /// Field value violates the spec's value constraints (e.g.
    /// negative amount in CCTP burn, gas limit zero in 4337).
    #[error("constraint violated: {field}: {reason}")]
    Constraint {
        /// Field name.
        field: &'static str,
        /// Why the constraint is violated.
        reason: String,
    },

    /// Unknown / unsupported variant (chain, network, lightning
    /// network prefix, etc.).
    #[error("unsupported variant `{0}`")]
    Unsupported(String),

    /// Checksum / signature / proof failed verification at the
    /// structural level. (Cryptographic verification is operator's
    /// job; this surfaces shape mismatches like wrong-length
    /// signatures or BOLT-11 checksum failure.)
    #[error("integrity check failed: {0}")]
    Integrity(String),
}
