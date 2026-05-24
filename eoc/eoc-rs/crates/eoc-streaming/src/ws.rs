//! WebSocket framing helpers.
//!
//! This module deliberately does **not** speak the WebSocket wire
//! protocol — that lives in `tokio-tungstenite` and similar crates and
//! is out of scope for an EOC primitive. What we do provide is the
//! transport-agnostic message vocabulary the host bridge translates
//! into real WS frames, plus utilities to serialize a streaming
//! [`crate::stream::Event`] as a WS text payload.

use serde::{Deserialize, Serialize};

use crate::error::StreamResult;
use crate::stream::Event;

/// WebSocket opcode subset relevant to streaming inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WsOpcode {
    /// UTF-8 JSON payload.
    Text,
    /// Binary payload (e.g. tool-result bytes).
    Binary,
    /// Liveness ping (host bridge should reply with `Pong`).
    Ping,
    /// Liveness pong.
    Pong,
    /// Connection close.
    Close,
}

/// A WebSocket frame as carried through the EOC streaming layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WsFrame {
    /// Opcode.
    pub opcode: WsOpcode,
    /// Payload bytes. For `Text` frames the bytes are UTF-8 JSON.
    pub payload: Vec<u8>,
}

impl WsFrame {
    /// Construct a text JSON frame from a normalized [`Event`].
    pub fn text_event(event: &Event) -> StreamResult<Self> {
        let bytes = serde_json::to_vec(event)?;
        Ok(Self {
            opcode: WsOpcode::Text,
            payload: bytes,
        })
    }

    /// Construct a ping frame with an optional payload (echo data).
    pub fn ping(payload: Vec<u8>) -> Self {
        Self {
            opcode: WsOpcode::Ping,
            payload,
        }
    }

    /// Construct a pong frame echoing a ping's payload.
    pub fn pong(payload: Vec<u8>) -> Self {
        Self {
            opcode: WsOpcode::Pong,
            payload,
        }
    }

    /// Construct a close frame.
    pub fn close() -> Self {
        Self {
            opcode: WsOpcode::Close,
            payload: Vec::new(),
        }
    }

    /// Decode a text frame back into an [`Event`].
    pub fn decode_event(&self) -> StreamResult<Event> {
        let s = std::str::from_utf8(&self.payload)
            .map_err(|e| crate::error::StreamError::Framing(e.to_string()))?;
        Ok(serde_json::from_str(s)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::Role;

    #[test]
    fn roundtrip_text_event() {
        let e = Event::MessageStart {
            id: Some("m".into()),
            role: Role::Assistant,
        };
        let frame = WsFrame::text_event(&e).unwrap();
        assert_eq!(frame.opcode, WsOpcode::Text);
        let decoded = frame.decode_event().unwrap();
        assert_eq!(decoded, e);
    }
}
