//! Typed error surface for the ACDC subsystem.

use smart_byte_core::Said;
use thiserror::Error;

/// Result alias used across this crate.
pub type Result<T> = std::result::Result<T, AcdcError>;

/// All failures surfaced by the ACDC credential, schema, TEL, IPEX,
/// edge-graph, registry, and selective-disclosure layers.
#[derive(Debug, Error)]
pub enum AcdcError {
    /// The recomputed SAID did not match the asserted `d` field.
    #[error("said mismatch: asserted {asserted} but body hashes to {computed}")]
    SaidMismatch {
        /// SAID claimed by the credential or schema's `d` field.
        asserted: Said,
        /// SAID re-derived from canonical content.
        computed: Said,
    },

    /// A required field was missing.
    #[error("missing field: {0}")]
    MissingField(&'static str),

    /// A field had the wrong shape or type.
    #[error("malformed field {field}: {detail}")]
    MalformedField {
        /// Field name.
        field: &'static str,
        /// Human-readable detail.
        detail: String,
    },

    /// Credential attributes failed schema validation.
    #[error("schema violation: {0}")]
    SchemaViolation(String),

    /// Selective-disclosure derivation rejected the request.
    #[error("selective disclosure: {0}")]
    SelectiveDisclosure(String),

    /// TEL event was invalid or out of order.
    #[error("tel error: {0}")]
    Tel(String),

    /// IPEX message could not be processed.
    #[error("ipex error: {0}")]
    Ipex(String),

    /// Edge section referenced a missing ACDC.
    #[error("edge target {0} not in graph")]
    EdgeMissing(Said),

    /// Cycle detected during edge graph traversal.
    #[error("edge cycle detected involving {0}")]
    EdgeCycle(Said),

    /// Registry rejected a write (e.g. duplicate, revoked, unknown).
    #[error("registry error: {0}")]
    Registry(String),

    /// Credential has been revoked.
    #[error("credential {0} revoked")]
    Revoked(Said),

    /// JSON encoding/decoding failure.
    #[error("json error: {0}")]
    Json(String),

    /// Canonical JSON (JCS) encoding failure.
    #[error("jcs error: {0}")]
    Jcs(String),
}

impl From<serde_json::Error> for AcdcError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e.to_string())
    }
}
