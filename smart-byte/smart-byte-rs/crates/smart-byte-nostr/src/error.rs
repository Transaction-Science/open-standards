//! Typed errors for the Nostr adapter.

use thiserror::Error;

/// All Nostr-side errors collapse to a single typed enum so callers can
/// match on the kind without depending on intermediate crates.
#[derive(Debug, Error)]
pub enum NostrError {
    /// Event JSON was structurally invalid or had a missing field.
    #[error("invalid event: {0}")]
    InvalidEvent(String),
    /// Event hash (id) did not match the canonical serialization.
    #[error("event id mismatch")]
    IdMismatch,
    /// Schnorr signature verification failed.
    #[error("bad signature")]
    BadSignature,
    /// Cryptographic key material was malformed.
    #[error("invalid key: {0}")]
    InvalidKey(String),
    /// Hex decode failed.
    #[error("hex error: {0}")]
    Hex(String),
    /// Bech32 (NIP-19) decode or encode failed.
    #[error("bech32 error: {0}")]
    Bech32(String),
    /// NIP-19 TLV payload was malformed.
    #[error("tlv error: {0}")]
    Tlv(String),
    /// Encryption / decryption failed (NIP-04, NIP-44, NIP-17).
    #[error("crypto error: {0}")]
    Crypto(String),
    /// MAC verification failed in NIP-44.
    #[error("mac mismatch")]
    MacMismatch,
    /// Unsupported NIP-44 version byte.
    #[error("unsupported nip-44 version: {0}")]
    UnsupportedVersion(u8),
    /// JSON encode / decode failed.
    #[error("json error: {0}")]
    Json(String),
    /// NIP-26 delegation token failed verification.
    #[error("invalid delegation: {0}")]
    InvalidDelegation(String),
    /// NIP-13 proof-of-work was below the requested target.
    #[error("insufficient pow: have {have}, want {want}")]
    InsufficientPow {
        /// Achieved leading-zero bit count.
        have: u32,
        /// Requested target leading-zero bit count.
        want: u32,
    },
    /// Relay-level NOTICE or rejected OK message.
    #[error("relay error: {0}")]
    Relay(String),
    /// NIP-05 verification failed.
    #[error("nip-05 verification failed: {0}")]
    Nip05(String),
    /// Decoder consumed all input but expected more.
    #[error("unexpected eof")]
    UnexpectedEof,
}

impl From<serde_json::Error> for NostrError {
    fn from(e: serde_json::Error) -> Self {
        NostrError::Json(e.to_string())
    }
}

impl From<secp256k1::Error> for NostrError {
    fn from(e: secp256k1::Error) -> Self {
        NostrError::Crypto(e.to_string())
    }
}
