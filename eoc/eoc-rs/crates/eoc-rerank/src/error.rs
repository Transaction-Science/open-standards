//! Typed errors for the re-ranking and hybrid retrieval layer.

use thiserror::Error;

/// Re-ranker / retriever error.
#[derive(Debug, Error)]
pub enum RerankError {
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

    /// The candidate set was too large for the vendor.
    #[error("batch too large: {0}")]
    BatchTooLarge(String),

    /// The request timed out client-side.
    #[error("request timed out")]
    Timeout,

    /// Generic network failure.
    #[error("network error: {0}")]
    NetworkError(String),

    /// Response payload could not be parsed.
    #[error("decode error: {0}")]
    Decode(String),

    /// Local-backend (ONNX / tokenizer) failure.
    #[error("local backend error: {0}")]
    Local(String),

    /// Index is misconfigured (e.g. dimensions mismatch).
    #[error("index error: {0}")]
    Index(String),

    /// An upstream embedder returned an error.
    #[error("embedding error: {0}")]
    Embedding(#[from] eoc_embeddings::error::EmbeddingError),

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
pub type RerankResult<T> = std::result::Result<T, RerankError>;

impl From<reqwest::Error> for RerankError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            return RerankError::Timeout;
        }
        if err.is_decode() {
            return RerankError::Decode(err.to_string());
        }
        RerankError::NetworkError(err.to_string())
    }
}

impl From<serde_json::Error> for RerankError {
    fn from(err: serde_json::Error) -> Self {
        RerankError::Decode(err.to_string())
    }
}
