//! Typed errors for the ActivityPub adapter.

use thiserror::Error;

/// All ActivityPub-side errors collapse to a single typed enum so callers
/// can match on the kind without depending on intermediate crates.
#[derive(Debug, Error)]
pub enum ActivityPubError {
    /// JSON encode / decode failed.
    #[error("json error: {0}")]
    Json(String),
    /// Malformed actor URI / IRI.
    #[error("invalid iri: {0}")]
    InvalidIri(String),
    /// Webfinger resource was malformed or did not match the requested acct.
    #[error("webfinger error: {0}")]
    Webfinger(String),
    /// HTTP Signature could not be produced or verified.
    #[error("http signature error: {0}")]
    HttpSig(String),
    /// A required `@context` term was missing from an incoming document.
    #[error("invalid context: {0}")]
    InvalidContext(String),
    /// An Activity referenced an unknown or unsupported `type`.
    #[error("unknown activity type: {0}")]
    UnknownActivity(String),
    /// An Activity violated a side-effect precondition (e.g. Undo with
    /// a mismatched actor).
    #[error("activity rejected: {0}")]
    Rejected(String),
    /// An object was not found in local storage.
    #[error("not found: {0}")]
    NotFound(String),
    /// A cryptographic operation (key parse / sign / verify) failed.
    #[error("crypto error: {0}")]
    Crypto(String),
    /// Vocabulary validation failed (missing required field, wrong type).
    #[error("vocabulary error: {0}")]
    Vocabulary(String),
    /// Inbox / outbox delivery failed.
    #[error("delivery error: {0}")]
    Delivery(String),
}

impl From<serde_json::Error> for ActivityPubError {
    fn from(e: serde_json::Error) -> Self {
        ActivityPubError::Json(e.to_string())
    }
}

impl From<url::ParseError> for ActivityPubError {
    fn from(e: url::ParseError) -> Self {
        ActivityPubError::InvalidIri(e.to_string())
    }
}

/// Convenience result alias.
pub type Result<T> = core::result::Result<T, ActivityPubError>;
