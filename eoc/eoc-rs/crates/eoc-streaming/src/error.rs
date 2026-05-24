//! Typed errors for streaming primitives.

use thiserror::Error;

/// Errors emitted while parsing, transporting, or accounting a stream.
#[derive(Debug, Error)]
pub enum StreamError {
    /// SSE / WS framing was malformed.
    #[error("framing error: {0}")]
    Framing(String),

    /// JSON deserialization of an event payload failed.
    #[error("parse error: {0}")]
    Parse(String),

    /// An unknown provider event type was observed.
    #[error("unknown event: {0}")]
    UnknownEvent(String),

    /// The stream was cancelled.
    #[error("stream cancelled")]
    Cancelled,

    /// The sink's bounded channel is closed.
    #[error("sink closed")]
    Closed,

    /// The sink rejected a send because of backpressure (non-blocking path).
    #[error("sink full (backpressure)")]
    Backpressure,

    /// The requested `Last-Event-ID` is no longer in the replay window.
    #[error("event {0} no longer resumable")]
    ResumeOutOfRange(String),

    /// The joule meter could not be read.
    #[error("meter error: {0}")]
    Meter(String),

    /// Generic backend / I/O.
    #[error("backend error: {0}")]
    Backend(String),
}

/// Convenience alias.
pub type StreamResult<T> = std::result::Result<T, StreamError>;

impl From<serde_json::Error> for StreamError {
    fn from(e: serde_json::Error) -> Self {
        StreamError::Parse(e.to_string())
    }
}

impl From<eoc_core::Error> for StreamError {
    fn from(e: eoc_core::Error) -> Self {
        StreamError::Backend(e.to_string())
    }
}
