//! OpenAI Chat Completions streaming delta mapper.
//!
//! Reference frame shape:
//!
//! ```jsonc
//! {
//!   "id": "chatcmpl-abc",
//!   "object": "chat.completion.chunk",
//!   "choices": [{
//!     "index": 0,
//!     "delta": { "role": "assistant", "content": "Hi" },
//!     "finish_reason": null
//!   }]
//! }
//! ```
//!
//! The terminating event is the literal SSE payload `[DONE]`, which the
//! caller should detect *before* invoking this mapper (it isn't valid
//! JSON). Tool calls are emitted as `delta.tool_calls[].function.{name,
//! arguments}` fragments; we emit `ToolCallStart` the first time a call
//! id is seen and `ToolCallDelta` for each argument fragment.

use serde_json::Value;
use std::collections::HashSet;
use std::sync::Mutex;

use crate::error::StreamResult;
use crate::stream::{Event, FinishReason, Role};

/// OpenAI → [`Event`] mapper.
///
/// Holds a small amount of state to distinguish the first appearance of
/// a tool call (which gets a `ToolCallStart`) from subsequent argument
/// fragments (`ToolCallDelta`).
#[derive(Debug, Default)]
pub struct OpenAiMapper {
    seen_tool_ids: Mutex<HashSet<String>>,
    started: Mutex<bool>,
}

impl OpenAiMapper {
    /// Construct a fresh mapper.
    pub fn new() -> Self {
        Self::default()
    }

    /// Map a parsed OpenAI chunk (the JSON object after `data: `).
    pub fn map(&self, data: &str) -> StreamResult<Vec<Event>> {
        if data.trim() == "[DONE]" {
            return Ok(vec![Event::MessageStop {
                reason: FinishReason::EndTurn,
            }]);
        }
        let v: Value = serde_json::from_str(data)?;
        let id = v.get("id").and_then(|s| s.as_str()).map(String::from);
        let mut out = Vec::new();

        let mut started_guard = self.started.lock().map_err(poisoned)?;
        if !*started_guard {
            *started_guard = true;
            out.push(Event::MessageStart {
                id: id.clone(),
                role: Role::Assistant,
            });
        }
        drop(started_guard);

        let choices = v.get("choices").and_then(|c| c.as_array());
        let Some(choices) = choices else {
            return Ok(out);
        };
        for choice in choices {
            let index = choice
                .get("index")
                .and_then(|i| i.as_u64())
                .unwrap_or(0) as u32;
            if let Some(delta) = choice.get("delta") {
                if let Some(content) = delta.get("content").and_then(|c| c.as_str())
                    && !content.is_empty()
                {
                    out.push(Event::TextDelta {
                        index,
                        delta: content.to_string(),
                    });
                }
                if let Some(calls) = delta.get("tool_calls").and_then(|c| c.as_array()) {
                    for call in calls {
                        let call_idx = call
                            .get("index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as u32;
                        let call_id = call
                            .get("id")
                            .and_then(|s| s.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let function = call.get("function");
                        let name = function
                            .and_then(|f| f.get("name"))
                            .and_then(|s| s.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let args = function
                            .and_then(|f| f.get("arguments"))
                            .and_then(|s| s.as_str())
                            .unwrap_or_default()
                            .to_string();
                        if !call_id.is_empty() {
                            let mut seen = self.seen_tool_ids.lock().map_err(poisoned)?;
                            if !seen.contains(&call_id) {
                                seen.insert(call_id.clone());
                                out.push(Event::ToolCallStart {
                                    index: call_idx,
                                    id: call_id.clone(),
                                    name: name.clone(),
                                });
                            }
                            drop(seen);
                        }
                        if !args.is_empty() {
                            out.push(Event::ToolCallDelta {
                                index: call_idx,
                                id: call_id,
                                arguments_delta: args,
                            });
                        }
                    }
                }
            }
            if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
                out.push(Event::MessageStop {
                    reason: map_reason(reason),
                });
            }
        }
        Ok(out)
    }
}

fn map_reason(s: &str) -> FinishReason {
    match s {
        "stop" => FinishReason::EndTurn,
        "length" => FinishReason::MaxTokens,
        "tool_calls" | "function_call" => FinishReason::ToolUse,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_string()),
    }
}

fn poisoned<T>(_: T) -> crate::error::StreamError {
    crate::error::StreamError::Backend("mapper state poisoned".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_content_delta() {
        let m = OpenAiMapper::new();
        let frame =
            r#"{"id":"x","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#;
        let out = m.map(frame).unwrap();
        assert!(matches!(out[0], Event::MessageStart { .. }));
        assert!(matches!(out[1], Event::TextDelta { .. }));
    }

    #[test]
    fn maps_done() {
        let m = OpenAiMapper::new();
        let out = m.map("[DONE]").unwrap();
        assert!(matches!(out[0], Event::MessageStop { .. }));
    }
}
