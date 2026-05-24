//! Sealed error type for `op-payouts`.

use thiserror::Error;

/// Result alias for payout operations.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes a payout driver can return.
#[derive(Debug, Error)]
pub enum Error {
    /// Beneficiary account is structurally invalid (bad IBAN checksum,
    /// non-numeric ABA, malformed crypto address, etc.).
    #[error("invalid beneficiary account for {rail}: {detail}")]
    InvalidBeneficiary {
        /// Rail that rejected the account.
        rail: &'static str,
        /// Why the account is invalid.
        detail: String,
    },

    /// The caller passed a [`PayoutMethod`](crate::PayoutMethod) variant
    /// the active driver does not support (e.g. an IBAN to Visa Direct).
    #[error("unsupported payout method for rail {rail}")]
    UnsupportedMethod {
        /// Rail name.
        rail: &'static str,
    },

    /// Amount or currency outside this rail's allowed envelope (e.g.
    /// non-EUR to SEPA, non-USD to FedNow, >$1M to RTP).
    #[error("currency or amount limit violation on {rail}: {detail}")]
    LimitViolation {
        /// Rail name.
        rail: &'static str,
        /// Specific limit / mismatch text.
        detail: String,
    },

    /// Beneficiary failed KYC screening (delegate to `op-screening`).
    #[error("beneficiary KYC rejected: {0}")]
    KycRejected(String),

    /// Funding source is insufficient or unavailable.
    #[error("funding error: {0}")]
    Funding(String),

    /// Network / transport failure when (a hypothetical) live transport
    /// is wired in. Drivers in this crate are offline-pure; this variant
    /// exists for downstream operators to surface their own failures.
    #[error("transport error: {0}")]
    Transport(String),

    /// Rail returned a final reject. `code` is rail-specific.
    #[error("rail rejected on {rail}: code={code}, message={message}")]
    RailRejected {
        /// Rail name.
        rail: &'static str,
        /// Rail-specific reject code.
        code: String,
        /// Human-readable message.
        message: String,
    },

    /// Driver-side validation rejected before submission.
    #[error("driver validation: {0}")]
    DriverValidation(String),

    /// Core layer error (Money, Currency).
    #[error(transparent)]
    Core(#[from] op_core::Error),
}
