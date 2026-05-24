//! Typed errors for the speculative-decoding pipeline.

use thiserror::Error;

/// Errors raised by draft / target / orchestrator paths.
#[derive(Debug, Error)]
pub enum SpecDecodeError {
    /// A draft model returned an empty proposal even though a non-zero
    /// `k` was requested. Almost always a backend bug; the orchestrator
    /// gives up rather than silently degrading to non-speculative
    /// generation.
    #[error("draft proposal was empty (k = {requested})")]
    EmptyDraft {
        /// The `k` the orchestrator asked for.
        requested: usize,
    },

    /// A target backend returned no logits / no tokens, so verification
    /// can't proceed.
    #[error("target verification produced no output")]
    EmptyVerification,

    /// The draft model's logit width didn't match the target's
    /// vocabulary; speculative decoding requires both to share a
    /// vocabulary.
    #[error("vocabulary mismatch: draft={draft}, target={target}")]
    VocabMismatch {
        /// Draft vocabulary size.
        draft: usize,
        /// Target vocabulary size.
        target: usize,
    },

    /// A sampler rejected its input (empty / NaN logits, etc.).
    #[error("sampler error: {0}")]
    Sampling(String),

    /// A backend invocation failed.
    #[error("backend error: {0}")]
    Backend(String),

    /// Configuration is internally inconsistent (e.g. `k == 0`,
    /// `max_new_tokens == 0`).
    #[error("invalid configuration: {0}")]
    Config(String),

    /// Stub algorithm that requires a real backend implementation was
    /// invoked. Carries the algorithm name for diagnostics.
    #[error("algorithm '{0}' is a documented stub — see module docs")]
    StubAlgorithm(&'static str),
}

/// Result alias for the crate.
pub type SpecDecodeResult<T> = std::result::Result<T, SpecDecodeError>;
