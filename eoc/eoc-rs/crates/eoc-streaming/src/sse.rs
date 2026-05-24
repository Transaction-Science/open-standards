//! Server-Sent Events framing.
//!
//! Implements the subset of the SSE wire format used by OpenAI and
//! Anthropic: `event:`, `data:`, `id:`, `retry:`, and the empty line
//! terminator. Multi-line `data:` blocks are concatenated with `\n`,
//! per the WHATWG spec. The parser is stateful and incremental so it
//! can be fed arbitrary byte chunks from a transport.

use crate::error::StreamResult;

/// A complete SSE event (i.e. an `event:`/`data:` block terminated by a
/// blank line).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// Event type. SSE defaults to `"message"` when no `event:` line is
    /// supplied; we preserve that.
    pub event: String,
    /// Concatenated `data:` payload (lines joined with `\n`).
    pub data: String,
    /// Optional `id:` (used for `Last-Event-ID` resumption).
    pub id: Option<String>,
    /// Optional `retry:` value in milliseconds.
    pub retry_ms: Option<u64>,
}

impl SseEvent {
    /// Encode this event back into the SSE wire format.
    pub fn encode(&self) -> String {
        let mut out = String::new();
        if !self.event.is_empty() && self.event != "message" {
            out.push_str("event: ");
            out.push_str(&self.event);
            out.push('\n');
        }
        if let Some(id) = &self.id {
            out.push_str("id: ");
            out.push_str(id);
            out.push('\n');
        }
        if let Some(ms) = self.retry_ms {
            out.push_str(&format!("retry: {ms}\n"));
        }
        for line in self.data.split('\n') {
            out.push_str("data: ");
            out.push_str(line);
            out.push('\n');
        }
        out.push('\n');
        out
    }
}

/// Incremental SSE parser. Feed it bytes; pull complete events.
#[derive(Debug, Default)]
pub struct SseParser {
    buf: String,
    cur_event: String,
    cur_data: Vec<String>,
    cur_id: Option<String>,
    cur_retry: Option<u64>,
    ready: Vec<SseEvent>,
}

impl SseParser {
    /// Construct a fresh parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a byte chunk. UTF-8 must be valid at chunk boundaries (the
    /// caller is responsible for accumulating partial code points).
    pub fn feed(&mut self, chunk: &str) -> StreamResult<()> {
        self.buf.push_str(chunk);
        while let Some(nl) = self.buf.find('\n') {
            // Drain one line, including the newline.
            let line: String = self.buf.drain(..=nl).collect();
            // Strip trailing \n (and \r if present).
            let mut line = line.as_str().trim_end_matches('\n');
            line = line.trim_end_matches('\r');
            self.consume_line(line)?;
        }
        Ok(())
    }

    fn consume_line(&mut self, line: &str) -> StreamResult<()> {
        // Empty line = dispatch.
        if line.is_empty() {
            if self.cur_data.is_empty() && self.cur_event.is_empty() && self.cur_id.is_none() {
                // Pure separator with nothing to dispatch.
                return Ok(());
            }
            let event = if self.cur_event.is_empty() {
                "message".to_string()
            } else {
                std::mem::take(&mut self.cur_event)
            };
            let data = self.cur_data.join("\n");
            self.cur_data.clear();
            let id = self.cur_id.take();
            let retry_ms = self.cur_retry.take();
            self.ready.push(SseEvent {
                event,
                data,
                id,
                retry_ms,
            });
            return Ok(());
        }
        // Comments start with ':' — ignore.
        if line.starts_with(':') {
            return Ok(());
        }
        let (field, value) = match line.find(':') {
            Some(i) => {
                let (f, v) = line.split_at(i);
                let v = &v[1..];
                let v = v.strip_prefix(' ').unwrap_or(v);
                (f, v)
            }
            None => (line, ""),
        };
        match field {
            "event" => self.cur_event = value.to_string(),
            "data" => self.cur_data.push(value.to_string()),
            "id" => self.cur_id = Some(value.to_string()),
            "retry" => {
                self.cur_retry = value.parse::<u64>().ok();
            }
            _ => { /* unknown field — spec says ignore */ }
        }
        Ok(())
    }

    /// Drain all events parsed so far.
    pub fn drain(&mut self) -> Vec<SseEvent> {
        std::mem::take(&mut self.ready)
    }

    /// Force-finalize: treat any pending lines as a final event.
    /// Useful when the transport closes without a trailing blank line.
    pub fn finalize(&mut self) -> StreamResult<Vec<SseEvent>> {
        if !self.buf.is_empty() {
            let pending = std::mem::take(&mut self.buf);
            for line in pending.split('\n') {
                let line = line.trim_end_matches('\r');
                self.consume_line(line)?;
            }
        }
        if !self.cur_data.is_empty()
            || !self.cur_event.is_empty()
            || self.cur_id.is_some()
            || self.cur_retry.is_some()
        {
            self.consume_line("")?;
        }
        Ok(self.drain())
    }
}

/// Convenience: parse a complete buffer in one call.
pub fn parse_all(input: &str) -> StreamResult<Vec<SseEvent>> {
    let mut p = SseParser::new();
    p.feed(input)?;
    p.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_anthropic_pair() {
        let wire = "event: ping\ndata: {}\n\n";
        let evs = parse_all(wire).expect("parse");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].event, "ping");
        assert_eq!(evs[0].data, "{}");
    }

    #[test]
    fn multiline_data_joins_with_newline() {
        let wire = "data: a\ndata: b\n\n";
        let evs = parse_all(wire).unwrap();
        assert_eq!(evs[0].data, "a\nb");
    }

    #[test]
    fn round_trip_encode() {
        let e = SseEvent {
            event: "delta".into(),
            data: "x".into(),
            id: Some("42".into()),
            retry_ms: None,
        };
        let wire = e.encode();
        let parsed = parse_all(&wire).unwrap();
        assert_eq!(parsed[0], e);
    }
}
