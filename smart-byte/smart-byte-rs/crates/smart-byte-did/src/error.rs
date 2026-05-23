//! Typed errors for DID parsing and resolution.

use thiserror::Error;

/// All DID-side errors collapse to a single typed enum so callers can
/// match on the kind without depending on intermediate crates.
#[derive(Debug, Error)]
pub enum DidError {
    /// The DID method requested has no resolver registered in this build.
    #[error("DID method not supported: {0}")]
    MethodNotSupported(String),
    /// DID syntax did not match `did:<method>:<id>`.
    #[error("invalid DID identifier: {0}")]
    InvalidIdentifier(String),
    /// DID URL syntax (path/query/fragment) is malformed.
    #[error("invalid DID URL: {0}")]
    InvalidDidUrl(String),
    /// Resolution succeeded structurally but the document was not found.
    #[error("DID document not found: {0}")]
    NotFound(String),
    /// The decoded payload was not a valid DID document.
    #[error("invalid DID document: {0}")]
    InvalidDocument(String),
    /// Network I/O failure during did:web (or other network-bound) resolution.
    #[error("network error: {0}")]
    NetworkError(String),
    /// Multibase encoding/decoding failed.
    #[error("multibase error: {0}")]
    Multibase(String),
    /// Base64 encoding/decoding failed.
    #[error("base64 error: {0}")]
    Base64(String),
    /// JSON encode/decode failed.
    #[error("json error: {0}")]
    Json(String),
    /// Multicodec prefix did not match a supported key type.
    #[error("unsupported key codec: 0x{0:x}")]
    UnsupportedKeyCodec(u64),
    /// Cryptographic key material was structurally invalid.
    #[error("invalid key material: {0}")]
    InvalidKey(String),
    /// A signature-method/key-type combination was inconsistent.
    #[error("signature method mismatch: {0}")]
    SignatureMethodMismatch(String),
    /// The DID method is recognised but this build only stubs it.
    #[error("DID method is stubbed in this build: {0}")]
    Stubbed(String),
}

impl From<multibase::Error> for DidError {
    fn from(e: multibase::Error) -> Self {
        DidError::Multibase(e.to_string())
    }
}

impl From<serde_json::Error> for DidError {
    fn from(e: serde_json::Error) -> Self {
        DidError::Json(e.to_string())
    }
}

impl From<base64::DecodeError> for DidError {
    fn from(e: base64::DecodeError) -> Self {
        DidError::Base64(e.to_string())
    }
}
