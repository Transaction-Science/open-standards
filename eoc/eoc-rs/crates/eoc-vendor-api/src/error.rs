//! Typed errors returned by vendor backends.

use thiserror::Error;

/// Vendor-API error.
#[derive(Debug, Error)]
pub enum VendorError {
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

    /// The vendor refused to answer due to its content policy.
    #[error("content filtered: {0}")]
    ContentFiltered(String),

    /// The request timed out client-side.
    #[error("request timed out")]
    Timeout,

    /// Generic network failure (DNS, TLS, connection reset).
    #[error("network error: {0}")]
    NetworkError(String),

    /// Response payload could not be parsed.
    #[error("decode error: {0}")]
    Decode(String),

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
pub type VendorResult<T> = std::result::Result<T, VendorError>;

impl From<reqwest::Error> for VendorError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            return VendorError::Timeout;
        }
        if err.is_decode() {
            return VendorError::Decode(err.to_string());
        }
        VendorError::NetworkError(err.to_string())
    }
}

impl From<serde_json::Error> for VendorError {
    fn from(err: serde_json::Error) -> Self {
        VendorError::Decode(err.to_string())
    }
}
