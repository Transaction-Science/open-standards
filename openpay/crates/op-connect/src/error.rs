//! Error types for `op-connect`.
//!
//! One sealed enum per crate convention. Callers should `match` exhaustively;
//! adding a variant is a SemVer-major change.

use thiserror::Error;

/// Result alias for `op-connect`.
pub type Result<T> = core::result::Result<T, Error>;

/// All possible failure modes inside `op-connect`.
#[derive(Debug, Error)]
pub enum Error {
    /// A connected account was referenced but does not exist in the registry.
    #[error("connected account not found: {0}")]
    AccountNotFound(String),

    /// The account is not in the right onboarding state for the requested operation.
    #[error("account state invalid for operation: {reason}")]
    AccountStateInvalid {
        /// Human-readable reason.
        reason: String,
    },

    /// An onboarding step was submitted out of order or twice.
    #[error("onboarding step invalid: {reason}")]
    OnboardingStepInvalid {
        /// Human-readable reason.
        reason: String,
    },

    /// A required onboarding requirement is missing.
    #[error("requirement missing: {0}")]
    RequirementMissing(String),

    /// Beneficial-owner identification incomplete.
    ///
    /// Per FinCEN Customer Due Diligence Final Rule (31 CFR § 1010.230)
    /// and EU AMLD5 (Directive (EU) 2018/843, Art. 3(6)), institutions
    /// must identify each natural person who owns 25% or more of an
    /// entity. If declared ownership totals less than 100% and no
    /// "control prong" individual is named, this error fires.
    #[error("FinCEN CDD beneficial-owner rule violated: {reason}")]
    BeneficialOwnerIncomplete {
        /// Which prong of the rule was missed.
        reason: String,
    },

    /// A split-payment plan failed validation: legs do not sum to source,
    /// or contained a negative leg.
    #[error("invalid split: {reason}")]
    InvalidSplit {
        /// Why the split was rejected.
        reason: String,
    },

    /// A currency mismatch between source payment and a split leg, or
    /// between accounts in an internal transfer.
    #[error("currency mismatch: {0}")]
    CurrencyMismatch(String),

    /// Arithmetic overflow on a Money sum.
    #[error("arithmetic overflow")]
    Overflow,

    /// Screening fired a hit that blocks onboarding.
    #[error("screening blocked: {reason}")]
    ScreeningBlocked {
        /// Why the screening result blocked the operation.
        reason: String,
    },

    /// An onboarding provider (the trait impl) returned an upstream error.
    #[error("provider error: {0}")]
    Provider(String),

    /// A payout-schedule field is out of range (e.g. `day_of_month=32`).
    #[error("invalid payout schedule: {reason}")]
    InvalidPayoutSchedule {
        /// Which field was invalid.
        reason: String,
    },

    /// A liability-model invocation tried to nest `Hybrid` inside `Hybrid`,
    /// which the model does not support.
    #[error("invalid liability model: {reason}")]
    InvalidLiabilityModel {
        /// Why the model was rejected.
        reason: String,
    },

    /// Tax-reporting form generation failed validation.
    #[error("tax reporting error: {reason}")]
    TaxReporting {
        /// What went wrong.
        reason: String,
    },

    /// ToS acceptance was rejected (mismatched hash, missing fields, etc.).
    #[error("tos acceptance invalid: {reason}")]
    TosInvalid {
        /// Why it was rejected.
        reason: String,
    },

    /// Anything bubbled up from `op-core`.
    #[error("op-core error: {0}")]
    Core(#[from] op_core::Error),
}

impl From<op_screening::Error> for Error {
    fn from(value: op_screening::Error) -> Self {
        Self::Provider(format!("screening: {value}"))
    }
}
