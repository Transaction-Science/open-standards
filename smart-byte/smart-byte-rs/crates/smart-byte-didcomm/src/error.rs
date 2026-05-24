//! Typed errors for DIDComm v2 messaging.

use thiserror::Error;

/// All DIDComm errors collapse to a single typed enum so callers can
/// match on the kind without depending on intermediate crates.
#[derive(Debug, Error)]
pub enum DidcommError {
    /// JSON encode/decode failed.
    #[error("json error: {0}")]
    Json(String),
    /// Base64 encode/decode failed.
    #[error("base64 error: {0}")]
    Base64(String),
    /// Cryptographic primitive failed (signing, AEAD, key agreement).
    #[error("crypto error: {0}")]
    Crypto(String),
    /// JWS/JWE structural validation failed.
    #[error("jose error: {0}")]
    Jose(String),
    /// Signature verification failed.
    #[error("signature verification failed: {0}")]
    Signature(String),
    /// Message could not be unpacked: wrong recipient or no matching key.
    #[error("no matching recipient key")]
    NoRecipientKey,
    /// Message has expired (per `expires_time`).
    #[error("message expired")]
    Expired,
    /// Message has already been seen (replay protection).
    #[error("replay detected: id {0}")]
    Replay(String),
    /// Application-protocol message did not match the expected schema.
    #[error("protocol error: {0}")]
    Protocol(String),
    /// DID resolution required for routing/key lookup failed.
    #[error("did resolution failed: {0}")]
    DidResolution(String),
    /// Unsupported algorithm or curve.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// Internal invariant violated.
    #[error("internal: {0}")]
    Internal(String),
}

impl From<serde_json::Error> for DidcommError {
    fn from(e: serde_json::Error) -> Self {
        DidcommError::Json(e.to_string())
    }
}

impl From<base64::DecodeError> for DidcommError {
    fn from(e: base64::DecodeError) -> Self {
        DidcommError::Base64(e.to_string())
    }
}
