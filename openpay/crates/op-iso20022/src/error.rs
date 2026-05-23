//! Errors. One sealed enum per the project-wide convention.

use thiserror::Error;

/// Result alias for `op-iso20022`.
pub type Result<T> = core::result::Result<T, Error>;

/// Failure modes for ISO 20022 message handling.
#[derive(Debug, Error)]
pub enum Error {
    /// Underlying XML parser failed.
    #[error("xml decode failed: {0}")]
    XmlDecode(String),

    /// Underlying XML serializer failed.
    #[error("xml encode failed: {0}")]
    XmlEncode(String),

    /// The XML parsed but didn't match the expected message shape.
    #[error("schema mismatch: expected {expected}, got {got}")]
    SchemaMismatch {
        /// Message kind the caller asked for.
        expected: &'static str,
        /// Message kind actually present in the XML.
        got: String,
    },

    /// A field required by ISO 20022 or by the active rail profile is missing.
    #[error("missing required field: {0}")]
    MissingField(&'static str),

    /// A field's value is outside the allowed code list / format / range.
    #[error("invalid value for {field}: {reason}")]
    InvalidField {
        /// Path to the field, e.g. `"GrpHdr.CtrlSum"`.
        field: &'static str,
        /// Human-readable reason.
        reason: String,
    },

    /// Rail profile rejected the message (e.g. `FedNow` requires UETR; missing).
    #[error("rail profile {profile} validation failed: {reason}")]
    ProfileViolation {
        /// Which profile.
        profile: &'static str,
        /// Why.
        reason: String,
    },

    /// Round-trip mismatch — the canonical XML differs from input after a
    /// parse-then-reserialize cycle. Used by conformance tests.
    #[error("round-trip mismatch")]
    RoundTripMismatch,

    /// Forwarded `op-core` error (e.g. currency / overflow during mapping).
    #[error(transparent)]
    Core(#[from] op_core::Error),
}
