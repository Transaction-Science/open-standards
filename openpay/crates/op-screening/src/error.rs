//! Typed errors for `op-screening`. One sealed enum, exhaustive matches.

use thiserror::Error;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failures the crate can return.
#[derive(Debug, Error)]
pub enum Error {
    /// HTTP transport failure while pulling a sanctions list.
    #[error("http error: {0}")]
    Http(String),

    /// The downloaded payload failed to parse.
    #[error("parse error: {0}")]
    Parse(String),

    /// On-disk snapshot read/write error.
    #[error("storage io error: {0}")]
    Io(String),

    /// CBOR (de)serialisation error.
    #[error("cbor error: {0}")]
    Cbor(String),

    /// Audit-log signature verification failed.
    #[error("audit signature invalid")]
    AuditSignatureInvalid,

    /// Audit-log hash chain has been tampered with.
    #[error("audit chain broken at index {0}")]
    AuditChainBroken(usize),

    /// Configuration error in a [`crate::screener::ScreenerConfig`].
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// The requested entity was not present in the index.
    #[error("entity not found: {0}")]
    EntityNotFound(String),
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<reqwest::Error> for Error {
    fn from(value: reqwest::Error) -> Self {
        Self::Http(value.to_string())
    }
}

impl From<quick_xml::Error> for Error {
    fn from(value: quick_xml::Error) -> Self {
        Self::Parse(value.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::Parse(value.to_string())
    }
}
