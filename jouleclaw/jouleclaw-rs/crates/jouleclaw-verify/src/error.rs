//! Errors produced by verifier *constructors* (not by their checks —
//! a failed check is a [`crate::VerifyResult::Fail`], not an error).

use thiserror::Error;

/// Errors raised when building a verifier from user input.
///
/// A failing check is **not** an error — that is a
/// [`crate::VerifyResult::Fail`]. Errors here represent malformed
/// configuration: an invalid regex pattern, a malformed expected
/// hash, etc.
#[derive(Debug, Error)]
pub enum VerifyError {
    /// The supplied regular expression failed to compile.
    #[error("invalid regex: {0}")]
    InvalidRegex(#[from] regex::Error),

    /// The expected BLAKE3 hash hex was malformed (wrong length or
    /// non-hex characters).
    #[error("invalid expected hash: {0}")]
    InvalidHash(String),
}
