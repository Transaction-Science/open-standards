//! Typed errors for the CRDT engine.

use thiserror::Error;

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, CrdtError>;

/// All errors produced by the CRDT engine.
#[derive(Debug, Error)]
pub enum CrdtError {
    /// CBOR encode/decode failure.
    #[error("cbor encode/decode failed: {0}")]
    Cbor(String),

    /// JSON encode/decode failure (for the JSON variant of value payloads).
    #[error("json encode/decode failed: {0}")]
    Json(String),

    /// A path traversal failed (segment missing or wrong node kind).
    #[error("invalid path: {0}")]
    InvalidPath(String),

    /// Operation targets a node whose kind does not match.
    #[error("type mismatch at path {path}: expected {expected}, got {actual}")]
    TypeMismatch {
        path: String,
        expected: &'static str,
        actual: &'static str,
    },

    /// Automerge interop failure.
    #[error("automerge interop failed: {0}")]
    Automerge(String),

    /// Yjs interop failure.
    #[error("yjs interop failed: {0}")]
    Yjs(String),

    /// Wall-clock readings went backwards in a way HLC could not recover from.
    #[error("hlc skew exceeded bound: {0} ms")]
    HlcSkew(i64),

    /// Op log integrity failure (id mismatch, etc).
    #[error("op integrity failure: {0}")]
    OpIntegrity(String),
}

impl From<serde_cbor::Error> for CrdtError {
    fn from(e: serde_cbor::Error) -> Self {
        CrdtError::Cbor(e.to_string())
    }
}

impl From<serde_json::Error> for CrdtError {
    fn from(e: serde_json::Error) -> Self {
        CrdtError::Json(e.to_string())
    }
}
