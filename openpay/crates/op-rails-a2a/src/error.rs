//! Sealed error type for A2A rails.

use thiserror::Error;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All failure modes for A2A operations.
#[derive(Debug, Error)]
pub enum Error {
    /// HTTP / MQ transport failure (DNS, TLS, mTLS handshake, timeout).
    #[error("transport error: {0}")]
    Transport(String),

    /// Rail returned a non-2xx HTTP status (or MQ negative ack).
    #[error("rail rejected: status={status}, code={code}, message={message}")]
    RailRejected {
        /// HTTP status or rail-specific status integer.
        status: u16,
        /// Rail-specific code (PIX `tipo`, `FedNow` status reason, RT1 reject code).
        code: String,
        /// Human-readable message.
        message: String,
    },

    /// ISO 20022 builder rejected the request before sending it (e.g.
    /// remittance >140 chars, UETR not v4).
    #[error("ISO 20022 builder rejected: {0}")]
    Iso20022(#[from] op_iso20022::Error),

    /// The rail returned a status reason code we don't recognize.
    /// Per the verify-before-assume rule, we error rather than guess.
    #[error("unknown rail status: {0}")]
    UnknownStatus(String),

    /// Caller passed a [`PaymentMethod`](op_core::PaymentMethod) variant
    /// the active driver doesn't support (e.g. an EMV card to `FedNow`).
    #[error("unsupported payment method for this A2A rail")]
    UnsupportedMethod,

    /// Caller passed an [`A2aKey`](op_core::A2aKey) variant the active
    /// driver doesn't support (e.g. a PIX key to `FedNow`).
    #[error("unsupported A2A key for {rail}")]
    UnsupportedA2aKey {
        /// Rail name that rejected it.
        rail: &'static str,
    },

    /// Currency or amount mismatch with the rail's allowed set (e.g.
    /// non-EUR to RT1, non-BRL to PIX, non-USD to `FedNow`).
    #[error("currency {got} not supported by rail {rail} (expected {expected})")]
    CurrencyMismatch {
        /// Rail name.
        rail: &'static str,
        /// Currency expected by the rail.
        expected: &'static str,
        /// Currency the caller passed.
        got: String,
    },

    /// Signer rejected the message or returned malformed bytes.
    #[error("signing failed: {0}")]
    Signing(String),

    /// Core layer error (invalid Money etc.).
    #[error(transparent)]
    Core(#[from] op_core::Error),

    /// Driver-side validation rejected the request before sending it.
    #[error("driver validation: {0}")]
    DriverValidation(String),
}
