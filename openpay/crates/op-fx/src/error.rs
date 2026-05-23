//! Sealed error type.

use thiserror::Error;

/// Crate result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// No quote is available for the requested currency pair.
    #[error("no quote: {from_currency}/{to_currency}")]
    NoQuote {
        /// Source currency code.
        from_currency: String,
        /// Target currency code.
        to_currency: String,
    },

    /// The quote's `valid_until_unix_secs` is in the past relative
    /// to `now`.
    #[error("quote expired: {from_currency}/{to_currency} valid_until={valid_until} now={now}")]
    QuoteExpired {
        /// Source currency code.
        from_currency: String,
        /// Target currency code.
        to_currency: String,
        /// Quote's expiry timestamp.
        valid_until: u64,
        /// Current timestamp.
        now: u64,
    },

    /// Caller tried to convert between identical source/target —
    /// no FX needed.
    #[error("same-currency conversion: {0}")]
    SameCurrency(String),

    /// Quote's `source_currency` doesn't match the `Money` being
    /// converted.
    #[error("currency mismatch: money is {money}, quote source is {quote_source}")]
    CurrencyMismatch {
        /// Money's currency code.
        money: String,
        /// Quote's source currency code.
        quote_source: String,
    },

    /// `rate_ppm` was 0 — would always produce zero.
    #[error("invalid rate: ppm cannot be zero")]
    InvalidRate,

    /// Conversion overflowed `i64::MAX`. Only reachable on
    /// pathological inputs (e.g. `i64::MAX` USD-equivalent).
    #[error("conversion overflow")]
    Overflow,

    /// Forwarded `op-core` error.
    #[error(transparent)]
    Core(#[from] op_core::Error),
}
