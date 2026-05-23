//! Typed errors for the post-quantum signature layer.

use thiserror::Error;

/// Errors that can occur during PQ key generation, signing, or verification.
#[derive(Debug, Error)]
pub enum PqError {
    /// The signature did not verify against the message and public key.
    #[error("post-quantum signature verification failed")]
    BadSignature,

    /// A key blob did not decode to a valid public or secret key for
    /// the requested algorithm (wrong length, wrong encoding, or
    /// otherwise malformed).
    #[error("malformed key for algorithm {0:?}: {1}")]
    MalformedKey(crate::algorithm::SignatureAlgorithm, String),

    /// A signature blob did not decode to a valid signature for the
    /// requested algorithm (wrong length or otherwise malformed).
    #[error("malformed signature for algorithm {0:?}: {1}")]
    MalformedSignature(crate::algorithm::SignatureAlgorithm, String),

    /// The caller mixed components from different algorithms (for
    /// example a Falcon public key with an ML-DSA signature).
    #[error("algorithm mismatch: expected {expected:?}, got {actual:?}")]
    AlgorithmMismatch {
        /// The algorithm the consumer claimed.
        expected: crate::algorithm::SignatureAlgorithm,
        /// The algorithm the operand actually belongs to.
        actual: crate::algorithm::SignatureAlgorithm,
    },

    /// The selected algorithm is recognized but not built into this
    /// crate at compile time. Most commonly returned for FN-DSA
    /// variants when the `falcon` feature is disabled.
    #[error("post-quantum algorithm {0:?} is not supported by this build")]
    UnsupportedAlgorithm(crate::algorithm::SignatureAlgorithm),
}

/// Convenience result alias used throughout the crate.
pub type Result<T> = core::result::Result<T, PqError>;
