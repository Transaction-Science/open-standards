//! Typed errors for the eval harnesses.

use std::path::PathBuf;

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, EvalError>;

/// All failures surfaced by an eval harness.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// IO problem reading a local dataset file.
    #[error("io error reading {path:?}: {source}")]
    Io {
        /// Path that triggered the error.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },

    /// JSON parsing failure (corrupt dataset, schema mismatch, etc.).
    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// A dataset row was missing a required field.
    #[error("dataset row missing field: {field}")]
    MissingField {
        /// Name of the missing field.
        field: &'static str,
    },

    /// The runtime tried to use a feature that was compiled out.
    #[error("feature {feature:?} is required for this operation")]
    FeatureDisabled {
        /// Cargo feature name (e.g. `"download"`, `"python"`).
        feature: &'static str,
    },

    /// HuggingFace Hub fetch failure (only reachable with `download`).
    #[error("dataset fetch failed: {0}")]
    Fetch(String),

    /// Sandbox grader (HumanEval) reported a failure.
    #[error("sandbox grader failed: {0}")]
    Sandbox(String),

    /// Catch-all for harness-specific failures.
    #[error("{0}")]
    Other(String),
}
