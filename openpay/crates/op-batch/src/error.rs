//! Typed error taxonomy for `op-batch`.

use thiserror::Error;

/// Sealed error type. One per crate, per project convention.
#[derive(Debug, Error)]
pub enum Error {
    /// Caller-supplied input violated a precondition (field length,
    /// charset, missing mandatory value, etc.).
    #[error("invalid: {0}")]
    Invalid(String),

    /// Lookup by id missed (e.g. unknown rail in the orchestrator).
    #[error("not found: {0}")]
    NotFound(String),

    /// A fixed-width record was the wrong length when encoded or
    /// when read back. NACHA = 94 chars; Bacs = 100 chars; the
    /// validator emits this with the actual length seen.
    #[error("record length: expected {expected}, got {got}")]
    RecordLength {
        /// Length the spec mandates.
        expected: usize,
        /// Length actually produced or read.
        got: usize,
    },

    /// A NACHA / Bacs / SEPA / SWIFT field violated its scheme rules.
    #[error("field rule: {field}: {reason}")]
    FieldRule {
        /// Field name (scheme-specific).
        field: &'static str,
        /// Human-readable reason.
        reason: String,
    },

    /// An ISO 20022 message couldn't be encoded or decoded.
    #[error("xml: {0}")]
    Xml(String),

    /// A bank submission attempt failed at the filesystem / network
    /// layer (could not write spool file, SFTP refused).
    #[error("submission: {0}")]
    Submission(String),

    /// Reconciliation rejected a row (amount mismatch, unknown
    /// reference, etc.).
    #[error("reconciliation: {0}")]
    Reconciliation(String),

    /// Bubbled-up `op-core` error.
    #[error(transparent)]
    Core(#[from] op_core::Error),

    /// Bubbled-up ISO 20022 error.
    #[error(transparent)]
    Iso20022(#[from] op_iso20022::Error),

    /// IO failure.
    #[error("io: {0}")]
    Io(String),
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

/// Shorthand result alias.
pub type Result<T> = core::result::Result<T, Error>;
