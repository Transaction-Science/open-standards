//! Typed errors for the ILP adapter.

use thiserror::Error;

/// All ILP-side errors collapse into a single typed enum so callers can
/// match on the kind without depending on intermediate modules.
#[derive(Debug, Error)]
pub enum IlpError {
    /// OER decode failed — input was truncated, length-determinant was
    /// malformed, or a fixed-size field was the wrong size.
    #[error("oer decode error: {0}")]
    Oer(String),
    /// Packet type byte was not one of `12` / `13` / `14`.
    #[error("unknown packet type: {0}")]
    UnknownPacketType(u8),
    /// A field that was decoded was syntactically valid but semantically
    /// out of range (e.g. a `Prepare` with an empty destination).
    #[error("invalid packet: {0}")]
    InvalidPacket(String),
    /// An ILP address failed scheme / segment / length validation.
    #[error("invalid address: {0}")]
    InvalidAddress(String),
    /// The hashlock condition did not match its fulfillment.
    #[error("condition mismatch")]
    ConditionMismatch,
    /// A STREAM frame referred to an unknown frame type.
    #[error("unknown stream frame: {0}")]
    UnknownStreamFrame(u8),
    /// A BTP packet referenced an unknown packet type.
    #[error("unknown btp packet type: {0}")]
    UnknownBtpPacketType(u8),
    /// An SPSP receiver document was malformed or missing required fields.
    #[error("spsp error: {0}")]
    Spsp(String),
    /// An Open Payments request was malformed or referenced an unknown
    /// resource.
    #[error("open payments error: {0}")]
    OpenPayments(String),
    /// The connector could not find a route for the given destination.
    #[error("no route to {0}")]
    NoRoute(String),
    /// The forwarding decision was blocked by a balance or rate-limit
    /// invariant.
    #[error("balance/rate-limit: {0}")]
    Balance(String),
    /// A currency-conversion step lacked a quote for the given pair.
    #[error("no quote for {from}/{to}")]
    NoQuote {
        /// Source asset code (e.g. `USD`).
        from: String,
        /// Destination asset code (e.g. `XRP`).
        to: String,
    },
    /// JSON encode / decode failed (used by SPSP and Open Payments).
    #[error("json error: {0}")]
    Json(String),
}

impl From<serde_json::Error> for IlpError {
    fn from(e: serde_json::Error) -> Self {
        IlpError::Json(e.to_string())
    }
}

/// Convenience result alias.
pub type Result<T> = core::result::Result<T, IlpError>;
