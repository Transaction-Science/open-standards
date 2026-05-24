//! Sealed error enum for the APAC e-wallet adapters.

use thiserror::Error;

/// Result alias used across the crate.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes raised by APAC wallet adapters.
///
/// `Clone` so callers can fan a single error out to multiple sinks
/// (retry bookkeeping, telemetry, ledger) without cloning wrapped
/// causes by hand.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// HTTP / transport-layer failure surfaced by the operator's
    /// injected transport. Adapters never reach the network directly.
    #[error("transport error: {0}")]
    Transport(String),

    /// Provider returned a structured error response.
    #[error("provider rejected: code={code}: {message}")]
    ProviderRejected {
        /// Provider-specific error code (e.g. Alipay `sub_code`,
        /// WeChat `err_code`, UPI `responseCode`).
        code: String,
        /// Human-readable diagnostic.
        message: String,
    },

    /// A request the caller built could not be serialized or violates
    /// the rail's pre-flight constraints (currency mismatch, empty VPA,
    /// negative amount, ...).
    #[error("invalid intent: {0}")]
    InvalidIntent(String),

    /// A response was structurally valid but a required field was
    /// missing.
    #[error("missing field: {0}")]
    MissingField(&'static str),

    /// JSON parse failure on a provider response.
    #[error("parse error: {0}")]
    Parse(String),

    /// EMVCo MPM TLV codec error (malformed length, bad CRC, bad
    /// payload-format indicator, ...).
    #[error("MPM codec error: {0}")]
    Mpm(String),

    /// Webhook / notify-callback signature verification failed.
    #[error("invalid signature")]
    InvalidSignature,

    /// VPA (Virtual Payment Address, India UPI) failed the syntactic
    /// `<handle>@<psp>` form.
    #[error("invalid VPA: {0}")]
    InvalidVpa(String),

    /// The caller asked for a flow the rail does not support
    /// (e.g. mandates on a one-shot QR rail).
    #[error("unsupported flow: {0}")]
    Unsupported(&'static str),

    /// Underlying `op-core` domain error (currency mismatch, overflow).
    #[error(transparent)]
    Core(#[from] op_core::Error),
}
