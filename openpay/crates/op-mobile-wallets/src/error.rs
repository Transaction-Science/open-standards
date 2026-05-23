//! Typed errors for mobile-wallet decryption.
//!
//! One sealed enum per crate convention. Every error variant maps
//! deterministically to a single decryption failure mode so callers
//! can drive precise telemetry without string-matching.

use thiserror::Error;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes for mobile-wallet payment-token decryption.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// The wrapping signature on the token did not verify against the
    /// supplied root signing keys. Apple Pay: the leaf certificate
    /// could not be chained to the Apple Root CA. Google Pay: the
    /// `intermediateSigningKey.signatures` did not verify against any
    /// configured Google root key, or the `signedMessage` signature
    /// did not verify against the intermediate key. Samsung Pay: the
    /// signing-leaf signature failed.
    #[error("bad signature on wallet token")]
    BadSignature,

    /// The certificate chain failed validation: malformed encoding,
    /// wrong issuer, or — most commonly in tests — the configured
    /// signing-key material is expired.
    #[error("bad certificate / cert chain")]
    BadCertificate,

    /// The token / cryptogram's authoring timestamp is outside the
    /// allowable window relative to `now`. Returned for replay-window
    /// failures: Google Pay's `messageExpiration` has passed, Apple
    /// Pay's `transaction_identifier` carries a too-old timestamp, or
    /// the cryptogram itself is past its issuance-window expiry.
    #[error("cryptogram / message expired")]
    BadCryptogramExpired,

    /// The token's declared protocol version is not supported by this
    /// build. Google Pay ECv1 returns this (deprecated and stubbed in
    /// this crate); any non-ECv2 Google Pay protocol returns this.
    #[error("unsupported wallet protocol version: {0}")]
    UnsupportedProtocolVersion(String),

    /// The token's bound merchant identifier does not match the
    /// `merchant_id` (or `merchant_id_hash` for Apple Pay) configured
    /// on the decryptor. This is a hard fail: a token issued for
    /// merchant A must not decrypt at merchant B.
    #[error("merchant id mismatch")]
    MerchantIdMismatch,

    /// The encrypted payload is malformed: wrong base64, wrong
    /// JSON shape, missing field, or an internal length/structure
    /// invariant was violated.
    #[error("malformed payment payload: {0}")]
    MalformedPayload(&'static str),

    /// The AEAD step failed to authenticate. AES-GCM tag mismatch
    /// (Apple Pay, Samsung Pay) or HMAC-SHA256 mismatch on the
    /// CTR-mode payload (Google Pay ECv2). Either the ciphertext was
    /// tampered with, or the wrong recipient private key was used.
    #[error("AEAD authentication failed (tag mismatch)")]
    AeadAuthFailed,

    /// The ECDH key-agreement step failed: a point was not on the
    /// curve, or the operator's recipient private key didn't parse.
    #[error("ECDH key-agreement failed")]
    KeyAgreementFailed,

    /// The operator-supplied [`crate::VaultTokenizer`] returned an
    /// error. This is a non-recoverable failure of the merchant's
    /// vault integration; the caller's vault is unavailable or
    /// rejected the credential.
    #[error("vault tokenizer failed: {0}")]
    VaultTokenizer(String),

    /// Internal cryptographic invariant violated — should not occur
    /// in practice and indicates a bug in this crate or one of its
    /// dependencies.
    #[error("internal crypto invariant: {0}")]
    Internal(&'static str),
}
