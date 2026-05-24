//! Anthropic Messages API stream event mapper.
//!
//! Reference event types (as of the 2024-10 / 2025 API revisions):
//!
//! * `message_start` — opens the message, supplies role + id.
//! * `content_block_start` — a content block opens. Block type is either
//!   `text` or `tool_use`.
//! * `content_block_delta` — a delta within a block.
//!   * `text_delta` carries `text`.
//!   * `input_json_delta` carries `partial_json` for a tool-use block.
//! * `content_block_stop` — block closes.
//! * `message_delta` — message-level metadata, including `stop_reason`.
//! * `message_stop` — message terminates.
//! * `ping` — liveness keepalive.
//!
//! Unknown events are surfaced as [`StreamError::UnknownEvent`] so the
//! caller can choose to log + skip, rather than silently dropping data.

use serde_json::Value;

use crate::error::{StreamError, StreamResult};
use crate::sse::SseEvent;
use crate::stream::{Event, FinishReason, Role};

/// Anthropic → [`Event`] mapper.
#[derive(Debug, Default, Clone, Copy)]
pub struct AnthropicMapper;

impl AnthropicMapper {
    /// Map a parsed Anthropic SSE event into zero or more normalized
    /// events. The vast majority of Anthropic frames map 1:1; the only
    /// case that produces zero events is an unrecognized block type
    /// inside `content_block_delta`.
    pub fn map(&self, ev: &SseEvent) -> StreamResult<Vec<Event>> {
        // `ping` events often carry literally `{}` data — that's fine.
        let v: Value = if ev.data.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&ev.data)?
        };
        match ev.event.as_str() {
            "message_start" => {
                let id = v
                    .get("message")
                    .and_then(|m| m.get("id"))
                    .and_then(|i| i.as_str())
                    .map(String::from);
                Ok(vec![Event::MessageStart {
                    id,
                    role: Role::Assistant,
                }])
            }
            "content_block_start" => {
                let index = block_index(&v);
                let block = v.get("content_block");
                let kind = block.and_then(|b| b.get("type")).and_then(|t| t.as_str());
                match kind {
                    Some("tool_use") => {
                        let id = block
                            .and_then(|b| b.get("id"))
                            .and_then(|i| i.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let name = block
                            .and_then(|b| b.get("name"))
                            .and_then(|n| n.as_str())
                            .unwrap_or_default()
                            .to_string();
                        Ok(vec![Event::ToolCallStart { index, id, name }])
                    }
                    _ => Ok(vec![]),
                }
            }
            "content_block_delta" => {
                let index = block_index(&v);
                let delta = v.get("delta");
                let kind = delta.and_then(|d| d.get("type")).and_then(|t| t.as_str());
                match kind {
                    Some("text_delta") => {
                        let text = delta
                            .and_then(|d| d.get("text"))
                            .and_then(|t| t.as_str())
                            .unwrap_or_default()
                            .to_string();
                        Ok(vec![Event::TextDelta {
                            index,
                            delta: text,
                        }])
                    }
                    Some("input_json_delta") => {
                        let partial = delta
                            .and_then(|d| d.get("partial_json"))
                            .and_then(|t| t.as_str())
                            .unwrap_or_default()
                            .to_string();
                        Ok(vec![Event::ToolCallDelta {
                            index,
                            // Anthropic doesn't repeat the id in deltas;
                            // downstream consumers correlate by index.
                            id: String::new(),
                            arguments_delta: partial,
                        }])
                    }
                    _ => Ok(vec![]),
                }
            }
            "content_block_stop" => {
                let index = block_index(&v);
                Ok(vec![Event::ContentBlockStop { index }])
            }
            "message_delta" => {
                let stop_reason = v
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|s| s.as_str())
                    .map(map_stop_reason);
                Ok(vec![Event::MessageDelta { stop_reason }])
            }
            "message_stop" => Ok(vec![Event::MessageStop {
                reason: FinishReason::EndTurn,
            }]),
            "ping" => Ok(vec![Event::Ping]),
            "error" => {
                let code = v
                    .get("error")
                    .and_then(|e| e.get("type"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let message = v
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|s| s.as_str())
                    .unwrap_or_default()
                    .to_string();
                Ok(vec![Event::Error { code, message }])
            }
            other => Err(StreamError::UnknownEvent(other.to_string())),
        }
    }
}

fn block_index(v: &Value) -> u32 {
    v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as u32
}

fn map_stop_reason(s: &str) -> FinishReason {
    match s {
        "end_turn" => FinishReason::EndTurn,
        "max_tokens" => FinishReason::MaxTokens,
        "tool_use" => FinishReason::ToolUse,
        "stop_sequence" => FinishReason::EndTurn,
        other => FinishReason::Other(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_text_delta() {
        let ev = SseEvent {
            event: "content_block_delta".into(),
            data: r#"{"index":0,"delta":{"type":"text_delta","text":"hi"}}"#.into(),
            id: None,
            retry_ms: None,
        };
        let out = AnthropicMapper.map(&ev).unwrap();
        assert_eq!(
            out,
            vec![Event::TextDelta {
                index: 0,
                delta: "hi".into()
            }]
        );
    }

    #[test]
    fn maps_ping() {
        let ev = SseEvent {
            event: "ping".into(),
            data: "{}".into(),
            id: None,
            retry_ms: None,
        };
        assert_eq!(AnthropicMapper.map(&ev).unwrap(), vec![Event::Ping]);
    }
}
