//! Typed errors for VC, DID, JWT, SD-JWT, Data Integrity, and Status List.

use thiserror::Error;

/// All VC-side errors collapse to a single typed enum so callers can
/// match on the kind without depending on intermediate crates.
#[derive(Debug, Error)]
pub enum VcError {
    /// JSON encode/decode failed.
    #[error("json error: {0}")]
    Json(String),
    /// Canonical-JSON (JCS) encoding failed.
    #[error("jcs canonicalisation error: {0}")]
    Jcs(String),
    /// Base64 decoding failed.
    #[error("base64 decode error: {0}")]
    Base64(String),
    /// Multibase encoding/decoding failed.
    #[error("multibase error: {0}")]
    Multibase(String),
    /// IRI parsing failed.
    #[error("invalid IRI: {0}")]
    Iri(String),
    /// DID syntax invalid.
    #[error("invalid DID: {0}")]
    Did(String),
    /// Cryptographic verification failed.
    #[error("signature verification failed: {0}")]
    Signature(String),
    /// Cryptosuite is not implemented in this build.
    #[error("unsupported cryptosuite: {0}")]
    UnsupportedCryptosuite(String),
    /// JWT structural error.
    #[error("jwt error: {0}")]
    Jwt(String),
    /// Selective-disclosure error.
    #[error("sd-jwt error: {0}")]
    SdJwt(String),
    /// Status list error.
    #[error("status list error: {0}")]
    StatusList(String),
    /// The decoded payload was not a valid credential.
    #[error("invalid credential: {0}")]
    Credential(String),
    /// I/O failure when reading a bitstring or fixture.
    #[error("io error: {0}")]
    Io(String),
    /// Smart Byte envelope bridge error.
    #[error("envelope bridge error: {0}")]
    Bridge(String),
}

impl From<serde_json::Error> for VcError {
    fn from(e: serde_json::Error) -> Self {
        VcError::Json(e.to_string())
    }
}

impl From<base64::DecodeError> for VcError {
    fn from(e: base64::DecodeError) -> Self {
        VcError::Base64(e.to_string())
    }
}

impl From<multibase::Error> for VcError {
    fn from(e: multibase::Error) -> Self {
        VcError::Multibase(e.to_string())
    }
}

impl From<std::io::Error> for VcError {
    fn from(e: std::io::Error) -> Self {
        VcError::Io(e.to_string())
    }
}
