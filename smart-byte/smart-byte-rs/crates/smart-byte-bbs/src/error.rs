//! Typed errors for BBS+ keygen, sign, verify, prove, and cargo
//! bridging.

use thiserror::Error;

/// All BBS+ errors collapse to a single typed enum so callers can
/// match on the kind without depending on the underlying curve crate.
#[derive(Debug, Error)]
pub enum BbsError {
    /// Generator count did not match the message count.
    #[error("generator count mismatch: have {have}, need {need}")]
    GeneratorCount {
        /// How many generators were provided.
        have: usize,
        /// How many generators were required.
        need: usize,
    },

    /// A bytewise decode produced an off-curve or non-canonical value.
    #[error("invalid encoding: {0}")]
    InvalidEncoding(String),

    /// A disclosed-index list referenced an index out of bounds for the
    /// message vector.
    #[error("disclosed index {index} out of range (have {len} messages)")]
    DisclosedIndexOutOfRange {
        /// The offending index.
        index: usize,
        /// The total message count.
        len: usize,
    },

    /// A disclosed-index list contained a duplicate index.
    #[error("disclosed index {0} repeated")]
    DisclosedIndexDuplicate(usize),

    /// Signature verification (sign-side) failed.
    #[error("signature verification failed")]
    SignatureVerification,

    /// Proof verification failed (challenge mismatch, pairing inequality,
    /// or undisclosed-response count mismatch).
    #[error("proof verification failed: {0}")]
    ProofVerification(String),

    /// Smart Byte envelope bridge failed.
    #[error("envelope bridge error: {0}")]
    Bridge(String),

    /// CBOR encode/decode failed.
    #[error("cbor error: {0}")]
    Cbor(String),

    /// JSON encode/decode failed.
    #[error("json error: {0}")]
    Json(String),

    /// Multibase encode/decode failed.
    #[error("multibase error: {0}")]
    Multibase(String),
}

impl From<serde_cbor::Error> for BbsError {
    fn from(e: serde_cbor::Error) -> Self {
        BbsError::Cbor(e.to_string())
    }
}

impl From<serde_json::Error> for BbsError {
    fn from(e: serde_json::Error) -> Self {
        BbsError::Json(e.to_string())
    }
}
