//! Typed errors for the safety primitives.

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, SafetyError>;

/// All failures surfaced by safety detectors / guards.
#[derive(Debug, thiserror::Error)]
pub enum SafetyError {
    /// A bundled regex failed to compile (programmer error).
    #[error("regex compile failed: {0}")]
    Regex(#[from] regex::Error),

    /// Input rejected by a guard / detector.
    #[error("input rejected: {reason}")]
    Rejected {
        /// Human-readable rejection reason.
        reason: String,
    },

    /// Output did not validate against the supplied JSON schema.
    #[error("structured output invalid: {0}")]
    Structure(String),

    /// Rate limit exceeded for the supplied principal.
    #[error("rate limit exceeded for principal {principal}")]
    RateLimit {
        /// Principal identifier (user, IP, API key, ...).
        principal: String,
    },

    /// JSON parsing failure.
    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// Catch-all for detector-specific failures.
    #[error("{0}")]
    Other(String),
}
