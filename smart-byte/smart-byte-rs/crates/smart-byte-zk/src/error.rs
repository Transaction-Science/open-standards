//! Typed errors for the ZK primitives.

use thiserror::Error;

/// All ZK errors collapse to a single typed enum so callers can match
/// on the *kind* without depending on a specific backend crate.
#[derive(Debug, Error)]
pub enum ZkError {
    /// A range-proof bit-length is not a power of two in
    /// `{8, 16, 32, 64}`, as required by the Bulletproofs range-proof
    /// API.
    #[error("unsupported range-proof bit length: {0} (must be one of 8, 16, 32, 64)")]
    UnsupportedBitLength(u32),

    /// The witness value does not fit within `bit_length` bits.
    #[error("witness {value} does not fit in {bit_length} bits")]
    WitnessOutOfRange {
        /// Value supplied by the prover.
        value: u64,
        /// Declared bit length.
        bit_length: u32,
    },

    /// The set-membership prover was asked to prove membership of a
    /// value that is not actually in the supplied set.
    #[error("set-membership: value not in supplied set")]
    NotInSet,

    /// A Bulletproofs proof failed to verify.
    #[error("bulletproofs verification failed: {0}")]
    BulletproofVerification(String),

    /// A serialisation step on a wire-level structure failed.
    #[error("encoding error: {0}")]
    Encoding(String),

    /// A scheme stub was used outside of its documented surface
    /// (typically: malformed dummy proof bytes).
    #[error("stub scheme error: {0}")]
    Stub(&'static str),
}
