//! Sealed error type.

use thiserror::Error;

/// Crate result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes for BIN parsing, lookup, and classification.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// BIN must be 6, 7, or 8 ASCII digits.
    #[error("invalid BIN length: got {got} digits, want 6..=8")]
    InvalidBinLength {
        /// Length of the rejected input.
        got: usize,
    },

    /// BIN contained a non-ASCII-digit character.
    #[error("invalid BIN character: {0:?}")]
    InvalidBinCharacter(char),

    /// No range in the tree contains the queried BIN.
    #[error("no range for BIN: {0}")]
    NoRange(String),

    /// A range was constructed with `low >= high` (empty / inverted).
    #[error("invalid range: low {low} must be < high {high}")]
    InvalidRange {
        /// Inclusive lower bound (8-digit form).
        low: u32,
        /// Exclusive upper bound (8-digit form).
        high: u32,
    },

    /// Luhn check failed.
    #[error("luhn check failed")]
    LuhnFailed,

    /// Country code was not a 2-character ASCII-uppercase string.
    #[error("invalid ISO 3166-1 alpha-2 country code: {0:?}")]
    InvalidCountryCode(String),
}
