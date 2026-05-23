//! Error types for EOC.

use thiserror::Error;

/// EOC error.
#[derive(Debug, Error)]
pub enum Error {
    /// Serialization failed.
    #[error("serialization error: {0}")]
    Serde(String),

    /// A stage backend returned an error.
    #[error("stage backend error: {0}")]
    Backend(String),

    /// I/O failure.
    #[error("io error: {0}")]
    Io(String),

    /// Hardware energy counter unavailable.
    #[error("no joule counter available")]
    NoCounter,
}

/// Convenience `Result` alias.
pub type Result<T> = std::result::Result<T, Error>;

impl From<serde_cbor::Error> for Error {
    fn from(err: serde_cbor::Error) -> Self {
        Error::Serde(err.to_string())
    }
}
