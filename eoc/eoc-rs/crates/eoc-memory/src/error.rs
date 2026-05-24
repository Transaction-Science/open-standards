//! Errors raised by `eoc-memory`.

use thiserror::Error;

/// Memory-subsystem error.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// The requested episode / item id was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// The configuration is invalid (e.g. zero capacity, negative
    /// decay constant).
    #[error("invalid config: {0}")]
    Config(String),

    /// The supplied embedding has the wrong dimensionality for this
    /// memory store.
    #[error("embedding dim mismatch: expected {expected}, got {got}")]
    Dim {
        /// The configured embedding dimensionality.
        expected: usize,
        /// The dimensionality of the incoming vector.
        got: usize,
    },

    /// A clock supplied a timestamp that runs backwards relative to
    /// the previously-observed event.
    #[error("non-monotonic timestamp: {got_ms} < {last_ms}")]
    NonMonotonic {
        /// The newly-supplied timestamp in ms.
        got_ms: u64,
        /// The most recent timestamp on record in ms.
        last_ms: u64,
    },

    /// An attention budget overflowed (working memory full).
    #[error("attention budget exhausted: {used}/{cap} tokens")]
    BudgetExhausted {
        /// Tokens currently held in the budget.
        used: u32,
        /// Configured token cap.
        cap: u32,
    },

    /// Backing-store IO or serialization failure (string for
    /// determinism).
    #[error("store: {0}")]
    Store(String),
}

/// Convenience result alias.
pub type MemoryResult<T> = std::result::Result<T, MemoryError>;
