//! Vercel AI SDK streaming protocol mapper.
//!
//! The "AI SDK Data Stream Protocol" frames each event as a single
//! line of the form `<type>:<json>\n`. The current type alphabet
//! relevant to streaming inference includes:
//!
//! * `0:"text-fragment"` — legacy text-delta (string-as-JSON).
//! * `2:[...]`            — data lines (opaque arrays).
//! * `9:{tool-call}`      — full tool call.
//! * `a:{tool-result}`    — tool result.
//! * `d:{finish}`         — finish marker.
//!
//! The SDK has also published a v2 protocol that uses named events
//! (`text-delta`, `tool-call`, `tool-result`, `finish`) wrapped in JSON
//! objects. We accept both forms: a leading `{` is parsed as v2; a
//! leading `<digit>:` is parsed as v1.

use serde_json::Value;

use crate::error::{StreamError, StreamResult};
use crate::stream::{Event, FinishReason, Role};

/// Vercel AI SDK → [`Event`] mapper.
#[derive(Debug, Default, Clone, Copy)]
pub struct VercelAiMapper {
    /// True if we've already emitted a `MessageStart`. The Vercel
    /// protocol has no explicit start event, so we synthesize one on
    /// the first frame; tracking it here would normally require
    /// interior mutability — instead the caller is expected to feed
    /// frames via the explicit [`Self::map_first`] / [`Self::map`]
    /// distinction.
    pub assume_started: bool,
}

impl VercelAiMapper {
    /// Construct a fresh mapper.
    pub fn new() -> Self {
        Self {
            assume_started: false,
        }
    }

    /// Map one Vercel protocol line. Returns zero or more normalized
    /// events. If `assume_started` is `false`, a synthesized
    /// `MessageStart` is prepended and the mapper flips its flag.
    pub fn map(&mut self, line: &str) -> StreamResult<Vec<Event>> {
        let line = line.trim_end_matches('\n');
        if line.is_empty() {
            return Ok(vec![]);
        }
        let mut out = Vec::new();
        if !self.assume_started {
            self.assume_started = true;
            out.push(Event::MessageStart {
                id: None,
                role: Role::Assistant,
            });
        }
        // v2: leading `{` means a JSON object with a `type` field.
        if line.starts_with('{') {
            let v: Value = serde_json::from_str(line)?;
            let ty = v
                .get("type")
                .and_then(|t| t.as_str())
                .ok_or_else(|| StreamError::Parse("missing type".into()))?;
            match ty {
                "text-delta" => {
                    let delta = v
                        .get("textDelta")
                        .or_else(|| v.get("delta"))
                        .and_then(|s| s.as_str())
                        .unwrap_or_default()
                        .to_string();
                    out.push(Event::TextDelta { index: 0, delta });
                }
                "tool-call" => {
                    let id = v
                        .get("toolCallId")
                        .or_else(|| v.get("id"))
                        .and_then(|s| s.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let name = v
                        .get("toolName")
                        .or_else(|| v.get("name"))
                        .and_then(|s| s.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let args = v
                        .get("args")
                        .map(|a| a.to_string())
                        .unwrap_or_default();
                    out.push(Event::ToolCallStart {
                        index: 0,
                        id: id.clone(),
                        name,
                    });
                    if !args.is_empty() {
                        out.push(Event::ToolCallDelta {
                            index: 0,
                            id: id.clone(),
                            arguments_delta: args,
                        });
                    }
                    out.push(Event::ToolCallEnd { index: 0, id });
                }
                "tool-result" => {
                    let id = v
                        .get("toolCallId")
                        .or_else(|| v.get("id"))
                        .and_then(|s| s.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let result = v
                        .get("result")
                        .map(|r| r.to_string())
                        .unwrap_or_default();
                    out.push(Event::ToolResult { id, result });
                }
                "finish" => {
                    let reason = v
                        .get("finishReason")
                        .and_then(|s| s.as_str())
                        .map(map_reason)
                        .unwrap_or(FinishReason::EndTurn);
                    out.push(Event::MessageStop { reason });
                }
                other => return Err(StreamError::UnknownEvent(other.to_string())),
            }
            return Ok(out);
        }
        // v1: `<type>:<json>` line-protocol.
        let (ty, rest) = line.split_once(':').ok_or_else(|| {
            StreamError::Framing("vercel v1 line missing ':'".into())
        })?;
        match ty {
            "0" => {
                let s: String = serde_json::from_str(rest)?;
                out.push(Event::TextDelta {
                    index: 0,
                    delta: s,
                });
            }
            "9" => {
                let v: Value = serde_json::from_str(rest)?;
                let id = v
                    .get("toolCallId")
                    .and_then(|s| s.as_str())
                    .unwrap_or_default()
                    .to_string();
                let name = v
                    .get("toolName")
                    .and_then(|s| s.as_str())
                    .unwrap_or_default()
                    .to_string();
                let args = v.get("args").map(|a| a.to_string()).unwrap_or_default();
                out.push(Event::ToolCallStart {
                    index: 0,
                    id: id.clone(),
                    name,
                });
                if !args.is_empty() {
                    out.push(Event::ToolCallDelta {
                        index: 0,
                        id: id.clone(),
                        arguments_delta: args,
                    });
                }
                out.push(Event::ToolCallEnd { index: 0, id });
            }
            "a" => {
                let v: Value = serde_json::from_str(rest)?;
                let id = v
                    .get("toolCallId")
                    .and_then(|s| s.as_str())
                    .unwrap_or_default()
                    .to_string();
                let result = v
                    .get("result")
                    .map(|r| r.to_string())
                    .unwrap_or_default();
                out.push(Event::ToolResult { id, result });
            }
            "d" => {
                let v: Value = serde_json::from_str(rest)?;
                let reason = v
                    .get("finishReason")
                    .and_then(|s| s.as_str())
                    .map(map_reason)
                    .unwrap_or(FinishReason::EndTurn);
                out.push(Event::MessageStop { reason });
            }
            // Unrecognised v1 prefixes (data lines, etc) are dropped.
            _ => {}
        }
        Ok(out)
    }
}

fn map_reason(s: &str) -> FinishReason {
    match s {
        "stop" | "end_turn" => FinishReason::EndTurn,
        "length" | "max_tokens" => FinishReason::MaxTokens,
        "tool-calls" | "tool_use" => FinishReason::ToolUse,
        "content-filter" | "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_text_fragment() {
        let mut m = VercelAiMapper::new();
        let out = m.map(r#"0:"hi""#).unwrap();
        assert!(matches!(out[0], Event::MessageStart { .. }));
        assert!(matches!(out[1], Event::TextDelta { .. }));
    }

    #[test]
    fn v2_finish() {
        let mut m = VercelAiMapper::new();
        let out = m.map(r#"{"type":"finish","finishReason":"stop"}"#).unwrap();
        assert!(matches!(out.last().unwrap(), Event::MessageStop { .. }));
    }
}
