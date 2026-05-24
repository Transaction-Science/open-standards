//! Typed errors for the AT Protocol adapter.

use thiserror::Error;

/// All AT-Protocol-side errors collapse to a single typed enum so callers
/// can match on the kind without depending on intermediate crates.
#[derive(Debug, Error)]
pub enum AtprotoError {
    /// AT URI syntax did not match `at://<authority>[/<path>]`.
    #[error("invalid AT URI: {0}")]
    InvalidAtUri(String),
    /// CAR file structure was malformed or truncated.
    #[error("invalid CAR file: {0}")]
    InvalidCar(String),
    /// IPLD / CID decoding failed.
    #[error("invalid CID: {0}")]
    InvalidCid(String),
    /// DAG-CBOR encoding or decoding failed.
    #[error("dag-cbor error: {0}")]
    DagCbor(String),
    /// JSON encode / decode failed.
    #[error("json error: {0}")]
    Json(String),
    /// Network I/O failed during PDS / firehose interaction.
    #[error("network error: {0}")]
    Network(String),
    /// XRPC server returned a structured error.
    #[error("xrpc error {code}: {message}")]
    Xrpc {
        /// Machine-readable error code (e.g. `"InvalidToken"`).
        code: String,
        /// Human-readable description.
        message: String,
    },
    /// Cryptographic key material or signature was invalid.
    #[error("crypto error: {0}")]
    Crypto(String),
    /// Repository invariant violated (missing field, bad pointer, etc.).
    #[error("repo error: {0}")]
    Repo(String),
    /// Lexicon schema validation failed.
    #[error("lexicon error: {0}")]
    Lexicon(String),
    /// MST invariant violated.
    #[error("mst error: {0}")]
    Mst(String),
    /// DID resolution failed.
    #[error("did error: {0}")]
    Did(String),
    /// An entry was looked up that does not exist.
    #[error("not found: {0}")]
    NotFound(String),
}

impl From<serde_json::Error> for AtprotoError {
    fn from(e: serde_json::Error) -> Self {
        AtprotoError::Json(e.to_string())
    }
}

impl From<serde_cbor::Error> for AtprotoError {
    fn from(e: serde_cbor::Error) -> Self {
        AtprotoError::DagCbor(e.to_string())
    }
}

impl From<smart_byte_did::DidError> for AtprotoError {
    fn from(e: smart_byte_did::DidError) -> Self {
        AtprotoError::Did(e.to_string())
    }
}
