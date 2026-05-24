//! Typed errors for local backends.

use thiserror::Error;

/// Errors produced by the local-inference backends and supporting code.
#[derive(Debug, Error)]
pub enum LocalError {
    /// The on-disk model file was not where it was supposed to be.
    #[error("model file not found: {0}")]
    ModelNotFound(String),

    /// The model file exists but its format is malformed or unsupported.
    #[error("invalid model format: {0}")]
    InvalidModelFormat(String),

    /// A native backend (llama.cpp, MLX, ONNX, TVM) returned an error.
    #[error("backend error ({backend}): {message}")]
    Backend {
        /// Backend identifier (`"llamacpp"`, `"mlx"`, `"mlc"`, `"onnx"`).
        backend: &'static str,
        /// Backend-specific error message.
        message: String,
    },

    /// A tokenizer encode/decode operation failed.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// A sampler was asked to sample from an empty / NaN logit vector.
    #[error("sampling error: {0}")]
    Sampling(String),

    /// The configured joule budget would be exceeded by running this
    /// query through this backend.
    #[error("budget exceeded: would cost ~{would_cost_microjoules} µJ, budget {budget_microjoules} µJ")]
    BudgetExceeded {
        /// Predicted cost of running the query.
        would_cost_microjoules: u64,
        /// Configured budget ceiling.
        budget_microjoules: u64,
    },

    /// Model registry I/O.
    #[error("registry error: {0}")]
    Registry(String),

    /// Generic I/O.
    #[error("io error: {0}")]
    Io(String),
}

/// Convenience alias.
pub type LocalResult<T> = std::result::Result<T, LocalError>;

impl From<std::io::Error> for LocalError {
    fn from(err: std::io::Error) -> Self {
        LocalError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for LocalError {
    fn from(err: serde_json::Error) -> Self {
        LocalError::Registry(err.to_string())
    }
}

impl From<LocalError> for eoc_core::Error {
    fn from(err: LocalError) -> Self {
        eoc_core::Error::Backend(err.to_string())
    }
}
