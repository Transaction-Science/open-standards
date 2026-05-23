//! Error type for `op-core`. One sealed enum, exhaustive `match` at the callsite.

use thiserror::Error;

/// Result alias for `op-core`.
pub type Result<T> = core::result::Result<T, Error>;

/// All possible failure modes inside `op-core`.
///
/// This is intentionally a sealed enum (no `#[non_exhaustive]` until we have
/// a release line). Callers should `match` exhaustively; adding a variant is
/// a SemVer-major change.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// Two `Money` values had different currencies.
    #[error("currency mismatch")]
    CurrencyMismatch,

    /// Integer overflow on a money amount or counter.
    #[error("arithmetic overflow")]
    Overflow,

    /// Invalid ISO 4217 currency code.
    #[error("invalid currency code")]
    InvalidCurrency,

    /// State machine: attempted a transition that is not legal from
    /// the current state.
    #[error("illegal state transition: {from} -> {to}")]
    IllegalTransition {
        /// State the payment was in.
        from: &'static str,
        /// State the caller tried to move it to.
        to: &'static str,
    },

    /// The payment method is not supported by the selected rail.
    #[error("payment method not supported on this rail")]
    UnsupportedMethod,

    /// A vault reference resolved to nothing.
    #[error("vault token not found")]
    VaultNotFound,
}
