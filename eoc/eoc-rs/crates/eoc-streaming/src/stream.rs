//! The normalized event vocabulary every provider mapper targets.
//!
//! [`Event`] is the lingua franca: providers emit framed bytes, mappers
//! turn those into [`Event`]s, and downstream consumers (cascade,
//! agent, HTTP/WS bridge) only ever see [`Event`]. The [`TokenStream`]
//! is a thin wrapper around a bounded MPSC channel so the sink end can
//! exert backpressure and the receiver end can poll like any other
//! tokio stream.

use serde::{Deserialize, Serialize};

use crate::error::{StreamError, StreamResult};

/// Speaker role for a content block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System / instruction author.
    System,
    /// End user.
    User,
    /// Model output.
    Assistant,
    /// Tool / function response.
    Tool,
}

/// Why the stream stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Model produced an end-of-message token.
    EndTurn,
    /// Model hit `max_tokens` / `max_output_tokens`.
    MaxTokens,
    /// Model emitted a tool call and is awaiting a response.
    ToolUse,
    /// Content was filtered by safety policy.
    ContentFilter,
    /// The stream was cancelled.
    Cancelled,
    /// Provider-specific reason.
    Other(String),
}

/// Normalized stream event.
///
/// Every provider mapper produces zero or more of these per upstream
/// frame. Consumers should treat unknown fields as forward-compatible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// The model started a new message.
    MessageStart {
        /// Optional message id (Anthropic, OpenAI both expose one).
        id: Option<String>,
        /// Speaker role for the message.
        role: Role,
    },
    /// A token-level text delta.
    TextDelta {
        /// Index of the content block this delta belongs to.
        index: u32,
        /// The new text chunk.
        delta: String,
    },
    /// A tool/function call was emitted by the model.
    ToolCallStart {
        /// Index of the content block.
        index: u32,
        /// Provider-issued call id.
        id: String,
        /// Tool name.
        name: String,
    },
    /// Streaming arguments for an in-progress tool call.
    ToolCallDelta {
        /// Index of the content block.
        index: u32,
        /// Provider-issued call id.
        id: String,
        /// Argument JSON fragment.
        arguments_delta: String,
    },
    /// A tool call is complete.
    ToolCallEnd {
        /// Index of the content block.
        index: u32,
        /// Provider-issued call id.
        id: String,
    },
    /// A tool-execution result was injected into the stream.
    ToolResult {
        /// Provider-issued call id this result satisfies.
        id: String,
        /// Result payload (opaque JSON-as-string).
        result: String,
    },
    /// A content block (text or tool call) closed.
    ContentBlockStop {
        /// Index of the content block that just closed.
        index: u32,
    },
    /// Provider liveness ping (Anthropic sends these every ~15s).
    Ping,
    /// The stream is finishing.
    MessageDelta {
        /// Optional partial reason — Anthropic emits this before stop.
        stop_reason: Option<FinishReason>,
    },
    /// Final event — the model has stopped emitting.
    MessageStop {
        /// Why the stream stopped.
        reason: FinishReason,
    },
    /// Provider raised an error mid-stream.
    Error {
        /// Provider-supplied error code.
        code: String,
        /// Human-readable message.
        message: String,
    },
}

/// Anything that can absorb [`Event`]s.
///
/// Implementors decide whether to block, error on backpressure, or drop.
#[async_trait::async_trait]
pub trait Sink: Send + Sync {
    /// Send an event. Returns `Closed` if the sink is shut down.
    async fn send(&self, event: Event) -> StreamResult<()>;
}

/// A bounded MPSC stream of normalized events.
///
/// Construct with [`TokenStream::bounded`]; clone the sink and hand it
/// to mappers; consume via [`TokenStream::recv`].
#[derive(Debug)]
pub struct TokenStream {
    rx: tokio::sync::mpsc::Receiver<Event>,
    tx: tokio::sync::mpsc::Sender<Event>,
}

impl TokenStream {
    /// Construct a bounded stream with the given capacity.
    pub fn bounded(capacity: usize) -> Self {
        let cap = capacity.max(1);
        let (tx, rx) = tokio::sync::mpsc::channel(cap);
        Self { rx, tx }
    }

    /// Clone a [`Sink`] handle for this stream.
    pub fn sink(&self) -> StreamSink {
        StreamSink {
            tx: self.tx.clone(),
        }
    }

    /// Receive the next event, or `None` once all sinks are dropped.
    pub async fn recv(&mut self) -> Option<Event> {
        self.rx.recv().await
    }

    /// Close the receive end. Any further sends will fail with `Closed`.
    pub fn close(&mut self) {
        self.rx.close();
    }
}

/// A clonable sink handle returned by [`TokenStream::sink`].
#[derive(Clone, Debug)]
pub struct StreamSink {
    tx: tokio::sync::mpsc::Sender<Event>,
}

impl StreamSink {
    /// Non-blocking send. Returns `Backpressure` if the channel is full.
    pub fn try_send(&self, event: Event) -> StreamResult<()> {
        use tokio::sync::mpsc::error::TrySendError;
        self.tx.try_send(event).map_err(|e| match e {
            TrySendError::Full(_) => StreamError::Backpressure,
            TrySendError::Closed(_) => StreamError::Closed,
        })
    }
}

#[async_trait::async_trait]
impl Sink for StreamSink {
    async fn send(&self, event: Event) -> StreamResult<()> {
        self.tx
            .send(event)
            .await
            .map_err(|_| StreamError::Closed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn roundtrip_text_delta() {
        let mut s = TokenStream::bounded(4);
        let sink = s.sink();
        sink.send(Event::MessageStart {
            id: Some("m1".into()),
            role: Role::Assistant,
        })
        .await
        .expect("send start");
        sink.send(Event::TextDelta {
            index: 0,
            delta: "hi".into(),
        })
        .await
        .expect("send delta");
        assert!(matches!(s.recv().await, Some(Event::MessageStart { .. })));
        assert!(matches!(s.recv().await, Some(Event::TextDelta { .. })));
        // Closing the receiver forces `recv` to drain to `None` without
        // requiring every cloned sender to be dropped.
        s.close();
        assert!(s.recv().await.is_none());
    }
}
