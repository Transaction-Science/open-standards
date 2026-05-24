//! Error types for `op-statements`.

use thiserror::Error;

/// Result alias for this crate.
pub type Result<T> = core::result::Result<T, Error>;

/// Sealed error enum for `op-statements`.
#[derive(Debug, Error)]
pub enum Error {
    /// A statement line referenced a currency that does not appear in
    /// the statement's primary currency aggregate.
    #[error("currency mismatch: line is {line}, statement is {statement}")]
    CurrencyMismatch {
        /// The line's currency code.
        line: String,
        /// The statement's primary currency code.
        statement: String,
    },

    /// Two statement lines collided on the same `id`. Statement line
    /// ids must be unique within a [`Statement`](crate::Statement).
    #[error("duplicate statement line id: {0}")]
    DuplicateLineId(String),

    /// Integer arithmetic overflowed while aggregating amounts. The
    /// statement is unrepresentable; the caller should split into
    /// smaller periods.
    #[error("arithmetic overflow during aggregation")]
    Overflow,

    /// The statement's [`Period`](crate::Period) had its end before its
    /// start.
    #[error("period end {end} precedes start {start}")]
    InvalidPeriod {
        /// Start (unix epoch seconds).
        start: u64,
        /// End (unix epoch seconds).
        end: u64,
    },

    /// A cadence anchor produced a degenerate period (zero or negative
    /// length).
    #[error("cadence produced zero-length period at anchor {0}")]
    DegenerateCadence(u64),

    /// The reconciliation engine encountered a structural problem
    /// (e.g. a discrepancy with no candidate).
    #[error("reconciliation: {0}")]
    Reconciliation(String),

    /// Underlying op-core error (currency / money / overflow).
    #[error(transparent)]
    Core(#[from] op_core::Error),

    /// JSON encode/decode failure when rendering or round-tripping.
    #[error("json: {0}")]
    Json(String),

    /// XML emission failure when building an ISO 20022 message.
    #[error("xml: {0}")]
    Xml(String),
}
