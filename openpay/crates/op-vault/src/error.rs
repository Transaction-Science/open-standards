//! Sealed error type for op-vault.
//!
//! Errors are deliberately coarse-grained. Detailed crypto failure
//! reasons are NOT exposed because the surface is part of the security
//! boundary — a verbose error message in the wrong place leaks oracle
//! information. Distinguish "token unknown" from "decryption failed"
//! only when the operator opts into the `debug-errors` build (which
//! does not exist yet on purpose).

use thiserror::Error;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes for vault operations.
#[derive(Debug, Error)]
pub enum Error {
    /// The token id is malformed (wrong length, bad characters, looks
    /// like a PAN). Distinct from `NotFound` so the caller can know
    /// it never had a chance.
    #[error("invalid token format")]
    InvalidToken,

    /// The token isn't in the vault, OR the vault refuses to confirm
    /// (these collapse intentionally — leaking which one is an oracle).
    #[error("token not found")]
    NotFound,

    /// Authenticated decryption failed. Could be tampering, wrong key,
    /// or corruption. Same opacity rationale.
    #[error("authentication failed")]
    AuthFailed,

    /// The token has expired per the operator's `TokenizationPolicy`.
    #[error("token expired")]
    Expired,

    /// The token is single-use and was already consumed.
    #[error("token already consumed")]
    AlreadyConsumed,

    /// Resource exhaustion (vault full, rate-limited, etc.).
    #[error("vault capacity or rate limit")]
    Capacity,

    /// Backend-specific failure. The string is operator-facing only —
    /// never propagate it to API responses.
    #[error("vault backend: {0}")]
    Backend(String),

    /// PAN validation failed during input (Luhn, length, exp date).
    #[error("invalid card data: {0}")]
    InvalidCard(String),
}
