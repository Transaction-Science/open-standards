//! Typed errors for the multi-modal surface.

use thiserror::Error;

/// Errors raised by multi-modal ingestion, preprocessing, transport, or
/// routing.
#[derive(Debug, Error)]
pub enum MultimodalError {
    /// Underlying transport (HTTP) failed.
    #[error("network error: {0}")]
    Network(String),

    /// Vendor returned a non-2xx status code or rejected the payload.
    #[error("vendor error (status={status}): {body}")]
    Vendor {
        /// HTTP status code.
        status: u16,
        /// Truncated response body.
        body: String,
    },

    /// HTTP 429 — the vendor rate-limited the request.
    #[error("rate limited (retry-after: {retry_after_secs:?}s)")]
    RateLimited {
        /// Seconds the vendor asked us to wait, when provided.
        retry_after_secs: Option<u64>,
    },

    /// HTTP 401/403 — the API key was rejected.
    #[error("invalid api key")]
    InvalidApiKey,

    /// The named model is not known to the vendor / local registry.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// I/O failure while reading a file-backed reference.
    #[error("io error: {0}")]
    Io(String),

    /// Image / audio decoding failed.
    #[error("decode error: {0}")]
    Decode(String),

    /// A required modality is missing from the [`crate::MultimodalQuery`].
    #[error("missing modality: {0:?}")]
    MissingModality(crate::Modality),

    /// The configured backend cannot satisfy the requested modality.
    #[error("unsupported modality: {0:?}")]
    Unsupported(crate::Modality),

    /// A feature-gated backend was invoked but the feature is disabled.
    #[error("feature not enabled: {0}")]
    FeatureDisabled(&'static str),

    /// Response payload could not be parsed.
    #[error("parse error: {0}")]
    Parse(String),
}

/// Convenience alias.
pub type MultimodalResult<T> = std::result::Result<T, MultimodalError>;

impl From<reqwest::Error> for MultimodalError {
    fn from(err: reqwest::Error) -> Self {
        MultimodalError::Network(err.to_string())
    }
}

impl From<serde_json::Error> for MultimodalError {
    fn from(err: serde_json::Error) -> Self {
        MultimodalError::Parse(err.to_string())
    }
}

impl From<std::io::Error> for MultimodalError {
    fn from(err: std::io::Error) -> Self {
        MultimodalError::Io(err.to_string())
    }
}

impl From<image::ImageError> for MultimodalError {
    fn from(err: image::ImageError) -> Self {
        MultimodalError::Decode(err.to_string())
    }
}

impl From<base64::DecodeError> for MultimodalError {
    fn from(err: base64::DecodeError) -> Self {
        MultimodalError::Decode(err.to_string())
    }
}
