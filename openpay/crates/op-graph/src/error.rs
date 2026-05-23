//! Error type for the graph-backed stores.
//!
//! Wraps Minigraf's own errors, op-ledger errors, and op-webhook
//! errors so callers can use a single `Result` type.

use thiserror::Error;

/// Crate-local result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// All failure modes for graph-backed store operations.
#[derive(Debug, Error)]
pub enum Error {
    /// The graph backend returned an error (Minigraf storage layer
    /// or parser).
    #[error("graph backend error: {0}")]
    Backend(String),

    /// A required vertex wasn't found.
    #[error("vertex not found: type={vertex_type} id={id}")]
    VertexNotFound {
        /// The expected vertex type.
        vertex_type: String,
        /// The vertex id (UUID).
        id: String,
    },

    /// The vertex had the wrong type. (E.g. we asked for a ledger_tx
    /// at an id but found a webhook_event.)
    #[error("vertex type mismatch: id={id} expected={expected} actual={actual}")]
    VertexTypeMismatch {
        /// The vertex id.
        id: String,
        /// The type the caller expected.
        expected: String,
        /// The type that was stored.
        actual: String,
    },

    /// A vertex was missing a property the caller required.
    #[error("missing property {property} on vertex {vertex_id}")]
    MissingProperty {
        /// The vertex id.
        vertex_id: String,
        /// The property name.
        property: String,
    },

    /// A property value couldn't be converted to the expected Rust
    /// type. This is a schema-drift signal — the graph has data the
    /// crate doesn't recognize.
    #[error(
        "property type mismatch on vertex {vertex_id}: property {property} could not be decoded as {expected_type}"
    )]
    PropertyTypeMismatch {
        /// The vertex id.
        vertex_id: String,
        /// The property.
        property: String,
        /// The Rust type we tried to decode to.
        expected_type: String,
    },

    /// An invariant in the graph was violated (e.g. a ledger
    /// transaction with no debit edges).
    #[error("graph invariant violated: {0}")]
    Invariant(String),

    /// JSON serialization / deserialization error.
    #[error("json error: {0}")]
    Json(String),

    /// Generic invalid-input error.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e.to_string())
    }
}

// Bridge to the op-ledger / op-webhook error types so trait
// implementations can use `?` cleanly.
impl From<Error> for op_ledger::Error {
    fn from(e: Error) -> Self {
        op_ledger::Error::InvalidInput(format!("op-graph: {e}"))
    }
}

impl From<Error> for op_webhook::Error {
    fn from(e: Error) -> Self {
        op_webhook::Error::InvalidInput(format!("op-graph: {e}"))
    }
}
