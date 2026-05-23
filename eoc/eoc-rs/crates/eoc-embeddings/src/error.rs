//! Typed errors returned by embedding backends.

use thiserror::Error;

/// Embedding-backend error.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// HTTP 429 — the vendor rate-limited the request.
    #[error("rate limited (retry-after: {retry_after_secs:?}s)")]
    RateLimited {
        /// Seconds the vendor asked us to wait (if provided).
        retry_after_secs: Option<u64>,
    },

    /// HTTP 401/403 — the API key was rejected.
    #[error("invalid api key")]
    InvalidApiKey,

    /// The requested model is not known to the vendor.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// The input batch was too large for the vendor.
    #[error("batch too large: {0}")]
    BatchTooLarge(String),

    /// The request timed out client-side.
    #[error("request timed out")]
    Timeout,

    /// Generic network failure (DNS, TLS, connection reset).
    #[error("network error: {0}")]
    NetworkError(String),

    /// Response payload could not be parsed.
    #[error("decode error: {0}")]
    Decode(String),

    /// Local-backend (ONNX / tokenizer) failure.
    #[error("local backend error: {0}")]
    Local(String),

    /// Catch-all for unexpected non-2xx responses.
    #[error("vendor error (status={status}): {body}")]
    Unexpected {
        /// HTTP status code.
        status: u16,
        /// Truncated response body for diagnostics (no secrets).
        body: String,
    },
}

/// Convenience alias.
pub type EmbeddingResult<T> = std::result::Result<T, EmbeddingError>;

impl From<reqwest::Error> for EmbeddingError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            return EmbeddingError::Timeout;
        }
        if err.is_decode() {
            return EmbeddingError::Decode(err.to_string());
        }
        EmbeddingError::NetworkError(err.to_string())
    }
}

impl From<serde_json::Error> for EmbeddingError {
    fn from(err: serde_json::Error) -> Self {
        EmbeddingError::Decode(err.to_string())
    }
}
