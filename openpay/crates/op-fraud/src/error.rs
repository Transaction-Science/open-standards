//! Sealed error type for op-fraud.

use thiserror::Error;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes for fraud scoring.
#[derive(Debug, Error)]
pub enum Error {
    /// Feature extraction got input it couldn't represent (e.g.
    /// non-finite f32, time conversion overflow).
    #[error("feature extraction failed: {0}")]
    Features(String),

    /// Model file couldn't be read or parsed.
    #[error("model load failed: {0}")]
    ModelLoad(String),

    /// Model produced output we don't understand (wrong rank, NaN).
    #[error("model output invalid: {0}")]
    ModelOutput(String),

    /// Score outside `[0.0, 1.0]` after model evaluation. Indicates a
    /// model that hasn't been trained with sigmoid output and a calibration
    /// step before export. Operators must fix their export pipeline.
    #[error("score {0} out of [0.0, 1.0]")]
    ScoreOutOfRange(f32),

    /// Scorer construction or invocation failed for reasons specific to
    /// the backend (e.g. ort initialization).
    #[error("scorer backend: {0}")]
    Backend(String),

    /// Core-layer error.
    #[error(transparent)]
    Core(#[from] op_core::Error),
}
