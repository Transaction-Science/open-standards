//! Sealed error type.

use thiserror::Error;

/// Crate `Result` alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Failure modes for crypto-rail operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    /// Network transport failed (RPC unreachable, websocket
    /// disconnect). Retry-safe at the rail layer.
    #[error("transport: {0}")]
    Transport(String),

    /// Chain rejected the transaction (revert, simulation failure,
    /// insufficient funds, invalid signature). Includes the
    /// chain-side reason if the gateway can surface one.
    #[error("rejected by chain: code={code} message={message}")]
    Rejected {
        /// Chain-supplied error code (e.g. `"insufficient_funds"`,
        /// EVM revert reason, Solana program error).
        code: String,
        /// Human-readable message.
        message: String,
    },

    /// Address doesn't match the chain's format (wrong checksum,
    /// invalid base58, wrong length). Validated by the driver
    /// before broadcast.
    #[error("invalid wallet address for chain `{chain}`: {reason}")]
    InvalidAddress {
        /// Chain identifier.
        chain: String,
        /// Why validation failed.
        reason: String,
    },

    /// Driver received a wallet from a chain it doesn't service
    /// (e.g. an Ethereum address sent to a `usdc-solana` driver).
    #[error("driver does not service chain `{0}`")]
    UnsupportedChain(String),

    /// Driver was asked to settle a token it doesn't service
    /// (e.g. USDC vs USDT confusion).
    #[error("driver does not service token `{0}`")]
    UnsupportedToken(String),

    /// A field required by the gateway is missing.
    #[error("missing required field: {0}")]
    MissingField(&'static str),

    /// Forwarded `op-core` error.
    #[error(transparent)]
    Core(#[from] op_core::Error),

    /// Driver self-reported a validation failure that doesn't fit
    /// the standard taxonomy. Free-form.
    #[error("driver validation: {0}")]
    DriverValidation(String),
}
