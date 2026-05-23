//! Sealed error type. One variant per failure class.

use thiserror::Error;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes for reconciliation.
#[derive(Debug, Error)]
pub enum Error {
    /// A statement source produced a structurally invalid line
    /// (missing amount, unparseable date, currency we can't map).
    #[error("malformed statement line: {0}")]
    MalformedLine(String),

    /// The webhook payload didn't match the reference settlement
    /// schema. Operators with a different payload shape ship their
    /// own [`crate::ReconciliationSource`].
    #[error("webhook payload not a recognized settlement event: {0}")]
    UnrecognizedWebhook(String),

    /// An ISO 20022 parse / validation failure from `op-iso20022`.
    #[error("iso20022: {0}")]
    Iso20022(String),

    /// JSON (de)serialization failure.
    #[error("json: {0}")]
    Json(String),

    /// A [`crate::ReconciliationStore`] backend failed to persist or
    /// read a task. The string is the backend's own error rendering.
    #[error("reconciliation store backend: {0}")]
    Backend(String),

    /// The caller asked to reconcile a window whose end precedes its
    /// start. Almost always a bug in the caller's date math.
    #[error("invalid window: start {start} is after end {end}")]
    InvalidWindow {
        /// Window start, unix epoch seconds.
        start: u64,
        /// Window end, unix epoch seconds.
        end: u64,
    },
}

impl From<op_iso20022::Error> for Error {
    fn from(e: op_iso20022::Error) -> Self {
        Self::Iso20022(e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e.to_string())
    }
}
