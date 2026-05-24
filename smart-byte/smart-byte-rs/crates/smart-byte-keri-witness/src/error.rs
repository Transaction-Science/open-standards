//! Typed error surface for the KERI + witness layer.

use smart_byte_core::Said;
use thiserror::Error;

/// Result alias used across this crate.
pub type Result<T> = std::result::Result<T, KeriError>;

/// All failures surfaced by the KERI key-event log, witness, watcher,
/// verifier, storage, and OOBI subsystems.
#[derive(Debug, Error)]
pub enum KeriError {
    /// The event's recomputed SAID did not match its asserted `d`.
    #[error("said mismatch: event asserts {asserted} but content hashes to {computed}")]
    SaidMismatch {
        /// SAID claimed by the event's `d` field.
        asserted: Said,
        /// SAID re-derived from the event's body with the placeholder substituted.
        computed: Said,
    },

    /// Sequence number does not follow the prior event.
    #[error("sequence error: expected {expected}, got {got}")]
    SequenceError {
        /// Expected sequence number.
        expected: u64,
        /// Sequence number actually present in the event.
        got: u64,
    },

    /// The event's `p` (prior-event SAID) does not match the previous event's `d`.
    #[error("prior-event mismatch: expected {expected}, got {got}")]
    PriorMismatch {
        /// Expected prior SAID.
        expected: Said,
        /// Prior SAID actually carried in the event.
        got: Said,
    },

    /// The event's `i` (controller AID) does not match the inception event's AID.
    #[error("controller AID mismatch: expected {expected}, got {got}")]
    ControllerAidMismatch {
        /// Expected AID derived from the inception event.
        expected: String,
        /// AID actually carried in the event.
        got: String,
    },

    /// The rotation event's revealed keys did not hash to the previously committed digests.
    #[error("pre-rotation commitment broken at position {index}")]
    PreRotationMismatch {
        /// Position in `k` / `n` where the per-key SAID failed to match.
        index: usize,
    },

    /// Threshold of valid signatures was not met.
    #[error("signature threshold not met: have {have}, need {need}")]
    ThresholdNotMet {
        /// Number of valid signatures collected.
        have: u32,
        /// Threshold required.
        need: u32,
    },

    /// Witness threshold of receipts was not met.
    #[error("witness threshold not met: have {have}, need {need}")]
    WitnessThresholdNotMet {
        /// Number of valid receipts collected.
        have: u32,
        /// Receipt threshold required.
        need: u32,
    },

    /// A witness has already issued a receipt for a different event at this sequence.
    #[error("duplicity: witness already signed a different event at sequence {sequence}")]
    DuplicityRefused {
        /// Sequence number at which the witness already committed.
        sequence: u64,
    },

    /// The event references an unknown algorithm byte or malformed key encoding.
    #[error("malformed key: {0}")]
    MalformedKey(String),

    /// Signature failed cryptographic verification.
    #[error("bad signature")]
    BadSignature,

    /// A recovery event was attempted without the required pre-conditions.
    #[error("recovery rejected: {0}")]
    RecoveryRejected(String),

    /// A delegation event references an unknown delegating parent.
    #[error("delegation parent {0} not found")]
    DelegationParentMissing(String),

    /// `rec` event encountered while v1 strict mode is engaged.
    #[error("recovery (rec) events disabled by strict-spec policy")]
    StrictSpecRejectsRec,

    /// Generic structural malformation surfaced by the verifier or storage layer.
    #[error("malformed event: {0}")]
    Malformed(String),

    /// CBOR encoding/decoding failure.
    #[error("cbor error: {0}")]
    Cbor(String),

    /// Filesystem I/O failure from `FileStorage`.
    #[error("io error: {0}")]
    Io(String),

    /// Storage layer reported an inconsistency the upper layers must surface.
    #[error("storage error: {0}")]
    Storage(String),
}

impl From<serde_cbor::Error> for KeriError {
    fn from(e: serde_cbor::Error) -> Self {
        Self::Cbor(e.to_string())
    }
}

impl From<std::io::Error> for KeriError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
