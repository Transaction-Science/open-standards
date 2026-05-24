//! Typed errors for the mDL / mDOC ingestion path.

use thiserror::Error;

/// All errors produced by the `smart-byte-mdl` crate collapse to a
/// single typed enum so callers can match on the kind without depending
/// on intermediate crates.
#[derive(Debug, Error)]
pub enum MdlError {
    /// CBOR encode or decode failed.
    #[error("cbor error: {0}")]
    Cbor(String),

    /// COSE_Sign1 construction or parsing failed.
    #[error("cose error: {0}")]
    Cose(String),

    /// Cryptographic signature verification failed.
    #[error("signature verification failed: {0}")]
    Signature(String),

    /// Required field missing from a credential or transcript.
    #[error("missing field: {0}")]
    Missing(String),

    /// A field's type or shape did not match the ISO 18013-5 schema.
    #[error("type error: {0}")]
    Type(String),

    /// MSO digest did not match an issuer-signed item's hash.
    #[error("digest mismatch in namespace {namespace} for element {element}")]
    DigestMismatch {
        /// CBOR namespace identifier (e.g. `org.iso.18013.5.1`).
        namespace: String,
        /// Element identifier (e.g. `family_name`).
        element: String,
    },

    /// The MSO's validity window does not include the current time.
    #[error("validity info out of range: now={now} signed={signed} valid_from={valid_from} valid_until={valid_until}")]
    ValidityOutOfRange {
        /// Current time presented by the verifier's clock.
        now: String,
        /// `signed` timestamp from the MSO.
        signed: String,
        /// `validFrom` timestamp from the MSO.
        valid_from: String,
        /// `validUntil` timestamp from the MSO.
        valid_until: String,
    },

    /// No matching trust anchor accepted the issuer COSE_Sign1.
    #[error("no trust anchor accepted issuer auth")]
    NoTrustAnchor,

    /// Smart Byte envelope bridge failed.
    #[error("envelope bridge: {0}")]
    Bridge(String),

    /// Caller supplied an unsupported COSE algorithm.
    #[error("unsupported algorithm: {0}")]
    UnsupportedAlg(String),

    /// Base64 decode failed.
    #[error("base64 decode: {0}")]
    Base64(String),
}

impl From<serde_cbor::Error> for MdlError {
    fn from(e: serde_cbor::Error) -> Self {
        MdlError::Cbor(e.to_string())
    }
}

impl From<base64::DecodeError> for MdlError {
    fn from(e: base64::DecodeError) -> Self {
        MdlError::Base64(e.to_string())
    }
}

impl From<coset::CoseError> for MdlError {
    fn from(e: coset::CoseError) -> Self {
        MdlError::Cose(format!("{e:?}"))
    }
}
