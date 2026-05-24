//! Typed errors for `op-3ds2`.
//!
//! One sealed enum. Every public fallible API returns `Result<T>` from
//! this module so callers can `match` exhaustively. Adding a variant is
//! a SemVer-major change.

use thiserror::Error;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All possible failure modes inside `op-3ds2`.
#[derive(Debug, Error)]
pub enum Error {
    /// JSON encode / decode of a 3DS message failed.
    #[error("json codec error: {0}")]
    Json(#[from] serde_json::Error),

    /// XML encode / decode (legacy 2.1.0 transition payloads).
    #[error("xml codec error: {0}")]
    Xml(String),

    /// A required field for the selected protocol version was missing.
    #[error("required field missing for {version:?}: {field}")]
    MissingField {
        /// Spec version we were emitting for.
        version: crate::version::ProtocolVersion,
        /// Field name as published in the EMVCo message catalogue.
        field: &'static str,
    },

    /// A field that the selected version forbids was present.
    #[error("field {field} is not permitted in {version:?}")]
    ForbiddenField {
        /// Spec version we were validating against.
        version: crate::version::ProtocolVersion,
        /// Field name.
        field: &'static str,
    },

    /// PAN failed the LUHN-10 check on the way into a version-check.
    #[error("invalid PAN: failed LUHN check")]
    InvalidPan,

    /// PAN's BIN did not resolve to any scheme we route to.
    #[error("no directory server route for PAN BIN {bin}")]
    NoDsRoute {
        /// First six (or eight) digits, safe to log.
        bin: String,
    },

    /// The DS returned a transport-level error or unexpected status.
    #[error("directory server transport error: {0}")]
    DsTransport(String),

    /// The ACS returned a transport-level error or unexpected status.
    #[error("acs transport error: {0}")]
    AcsTransport(String),

    /// The ACS returned an [`ErrorMessage`](crate::message::ErrorMessage).
    #[error("3ds error message: code={code} component={component}")]
    ProtocolError {
        /// EMVCo error code (5 digits).
        code: String,
        /// The component that originated the error.
        component: String,
    },

    /// The challenge window exceeded the timeout (default 5 minutes).
    #[error("challenge timeout exceeded")]
    ChallengeTimeout,

    /// Decoupled-authentication polling exceeded its budget.
    #[error("decoupled authentication timed out after {polls} polls")]
    DecoupledTimeout {
        /// How many polls were issued before giving up.
        polls: u32,
    },

    /// Browser fingerprint payload was malformed.
    #[error("invalid fingerprint payload: {0}")]
    InvalidFingerprint(&'static str),

    /// A base64 / cryptogram decode failed.
    #[error("invalid cryptogram encoding")]
    InvalidCryptogram,

    /// Version negotiation produced no overlap with the DS's supported
    /// versions. Caller should fall back to a non-3DS authorization.
    #[error("no common 3DS protocol version between requestor and DS")]
    NoCommonVersion,

    /// Exemption was requested but the runtime context did not satisfy
    /// the regulatory prerequisites (e.g. TRA without a qualifying
    /// fraud-rate bracket).
    #[error("exemption {0} not eligible under PSD2/PSD3 RTS")]
    ExemptionIneligible(&'static str),

    /// An internal invariant was violated. Indicates a bug.
    #[error("invariant violation: {0}")]
    Invariant(&'static str),
}
