//! Typed errors for the learned-router crate.

use thiserror::Error;

/// Crate-level error type.
#[derive(Debug, Error)]
pub enum Error {
    /// A router was queried before being trained / fit.
    #[error("router has not been trained yet")]
    NotTrained,

    /// An embedding vector had the wrong length.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        got: usize,
    },

    /// State import (deserialization) failed.
    #[error("state import failed: {0}")]
    StateImportFailed(String),

    /// Generic JSON failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Crate-level result alias.
pub type Result<T> = std::result::Result<T, Error>;
