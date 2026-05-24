//! Error type for `op-revrec`. One sealed enum, exhaustive `match` at the callsite.

use thiserror::Error;

/// Result alias for `op-revrec`.
pub type Result<T> = core::result::Result<T, Error>;

/// All possible failure modes inside `op-revrec`.
///
/// The variant set is intentionally narrow: revenue-recognition math
/// either succeeds or it produces a precise, auditable failure that the
/// caller has to handle. There are no "unknown" or "soft" errors — every
/// failure has a one-to-one mapping to an ASC 606 / IFRS 15 concept.
#[derive(Debug, Error)]
pub enum Error {
    /// A performance obligation referenced by id does not exist on the
    /// contract.
    #[error("unknown performance obligation: {0}")]
    UnknownObligation(String),

    /// The sum of standalone selling prices on a contract is zero —
    /// allocation by relative SSP is undefined.
    #[error("transaction price allocation requires non-zero total SSP")]
    ZeroStandaloneSellingPrice,

    /// Currency of a performance obligation does not match the contract
    /// transaction-price currency.
    #[error("currency mismatch between obligation and contract transaction price")]
    CurrencyMismatch,

    /// Money arithmetic in `op-core` overflowed or otherwise rejected the
    /// operation.
    #[error("money error: {0}")]
    Money(#[from] op_core::Error),

    /// Variable-consideration estimate exceeds the constrained ceiling
    /// — recognition would book revenue that is not "highly probable"
    /// of not reversing (ASC 606-10-32-11).
    #[error("variable consideration estimate {estimate} exceeds constraint ceiling {ceiling}")]
    VariableConstraintBreached {
        /// The unconstrained estimate, in minor units of the contract currency.
        estimate: i64,
        /// The constraint ceiling, in minor units of the contract currency.
        ceiling: i64,
    },

    /// A refund or reversal exceeds the amount previously recognized.
    #[error("refund {refund} exceeds recognized revenue {recognized}")]
    RefundExceedsRecognized {
        /// Refund amount in minor units.
        refund: i64,
        /// Previously recognized amount in minor units.
        recognized: i64,
    },

    /// A recognition schedule was asked to produce a period with
    /// `end < start`, or zero-length where positive length is required.
    #[error("invalid schedule period: start={start}, end={end}")]
    InvalidPeriod {
        /// Period start (ISO-8601 date).
        start: String,
        /// Period end (ISO-8601 date).
        end: String,
    },

    /// Contract-modification handling rejected the change.
    #[error("contract modification rejected: {0}")]
    ModificationRejected(String),

    /// FX translation failed because no rate was supplied for the
    /// recognition date.
    #[error("no FX rate for {from}/{to} on {date}")]
    MissingFxRate {
        /// Source currency code.
        from: String,
        /// Target currency code.
        to: String,
        /// Recognition date (ISO-8601).
        date: String,
    },

    /// Generic invariant violation. Carries a description for the audit
    /// trail; reachable only from defensive checks.
    #[error("invariant violated: {0}")]
    Invariant(String),
}
