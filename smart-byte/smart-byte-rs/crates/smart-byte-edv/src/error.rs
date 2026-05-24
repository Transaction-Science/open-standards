//! Typed errors for Encrypted Data Vaults.

use thiserror::Error;

/// All EDV errors collapse to a single typed enum so callers can match on
/// the kind without depending on intermediate crates.
#[derive(Debug, Error)]
pub enum EdvError {
    /// JSON encode/decode failed.
    #[error("json error: {0}")]
    Json(String),
    /// Base64 encode/decode failed.
    #[error("base64 error: {0}")]
    Base64(String),
    /// Cryptographic primitive failed (AEAD, key agreement, HMAC).
    #[error("crypto error: {0}")]
    Crypto(String),
    /// JWE structural validation failed.
    #[error("jose error: {0}")]
    Jose(String),
    /// HKDF derivation failed.
    #[error("hkdf error: {0}")]
    Hkdf(String),
    /// No matching recipient key found in the JWE envelope.
    #[error("no matching recipient key")]
    NoRecipientKey,
    /// Document not found in the vault.
    #[error("document not found: {0}")]
    NotFound(String),
    /// Stream chunk index out of order or missing.
    #[error("stream error: {0}")]
    Stream(String),
    /// Capability chain validation failed.
    #[error("capability error: {0}")]
    Capability(String),
    /// Caller does not hold a capability granting the requested action.
    #[error("unauthorized: action {0} on {1}")]
    Unauthorized(String, String),
    /// Encrypted index lookup failed.
    #[error("index error: {0}")]
    Index(String),
    /// Unsupported algorithm or encoding.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// Internal invariant violated.
    #[error("internal: {0}")]
    Internal(String),
}

impl From<serde_json::Error> for EdvError {
    fn from(e: serde_json::Error) -> Self {
        EdvError::Json(e.to_string())
    }
}

impl From<base64::DecodeError> for EdvError {
    fn from(e: base64::DecodeError) -> Self {
        EdvError::Base64(e.to_string())
    }
}
