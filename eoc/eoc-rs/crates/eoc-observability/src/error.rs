//! Error types for `eoc-observability`.

use thiserror::Error;

/// Observability error.
#[derive(Debug, Error)]
pub enum ObsError {
    /// Span / context propagation parse failure.
    #[error("parse error: {0}")]
    Parse(String),

    /// Exporter failure.
    #[error("exporter error: {0}")]
    Exporter(String),

    /// Sampler misconfiguration.
    #[error("sampler error: {0}")]
    Sampler(String),

    /// Metric type mismatch (e.g. record on a counter that wants delta only).
    #[error("metric error: {0}")]
    Metric(String),

    /// Serialization failure.
    #[error("serialization error: {0}")]
    Serde(String),
}

impl From<serde_json::Error> for ObsError {
    fn from(e: serde_json::Error) -> Self {
        ObsError::Serde(e.to_string())
    }
}

/// Crate result alias.
pub type ObsResult<T> = std::result::Result<T, ObsError>;
