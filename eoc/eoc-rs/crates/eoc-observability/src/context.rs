//! W3C tracecontext + baggage propagation.
//!
//! Wire format follows the [W3C Trace Context] spec:
//!
//! ```text
//! traceparent: 00-<trace-id 32hex>-<span-id 16hex>-<flags 2hex>
//! tracestate:  vendor1=value1,vendor2=value2
//! baggage:     key1=value1,key2=value2
//! ```
//!
//! [W3C Trace Context]: https://www.w3.org/TR/trace-context/

use crate::error::{ObsError, ObsResult};
use std::collections::BTreeMap;

/// 128-bit trace identifier. All-zero is the invalid sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceId(pub [u8; 16]);

impl TraceId {
    /// The invalid (all-zero) trace id.
    pub const INVALID: TraceId = TraceId([0u8; 16]);

    /// True if this id is the all-zero sentinel.
    pub fn is_invalid(&self) -> bool {
        self.0 == [0u8; 16]
    }

    /// Render as 32 lowercase hex chars.
    pub fn to_hex(&self) -> String {
        hex_encode(&self.0)
    }

    /// Parse from 32 hex chars.
    pub fn from_hex(s: &str) -> ObsResult<Self> {
        if s.len() != 32 {
            return Err(ObsError::Parse(format!(
                "trace_id must be 32 hex chars, got {}",
                s.len()
            )));
        }
        let mut out = [0u8; 16];
        hex_decode(s, &mut out)?;
        Ok(TraceId(out))
    }
}

/// 64-bit span identifier. All-zero is the invalid sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanId(pub [u8; 8]);

impl SpanId {
    /// The invalid (all-zero) span id.
    pub const INVALID: SpanId = SpanId([0u8; 8]);

    /// True if this id is the all-zero sentinel.
    pub fn is_invalid(&self) -> bool {
        self.0 == [0u8; 8]
    }

    /// Render as 16 lowercase hex chars.
    pub fn to_hex(&self) -> String {
        hex_encode(&self.0)
    }

    /// Parse from 16 hex chars.
    pub fn from_hex(s: &str) -> ObsResult<Self> {
        if s.len() != 16 {
            return Err(ObsError::Parse(format!(
                "span_id must be 16 hex chars, got {}",
                s.len()
            )));
        }
        let mut out = [0u8; 8];
        hex_decode(s, &mut out)?;
        Ok(SpanId(out))
    }
}

/// Trace flags. Only the sampled bit is defined by the W3C spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TraceFlags(pub u8);

impl TraceFlags {
    /// `sampled = 0x01`.
    pub const SAMPLED: TraceFlags = TraceFlags(0x01);

    /// True if the sampled bit is set.
    pub fn is_sampled(&self) -> bool {
        (self.0 & 0x01) != 0
    }

    /// Return a copy with the sampled bit forced on.
    pub fn with_sampled(self, on: bool) -> Self {
        if on {
            TraceFlags(self.0 | 0x01)
        } else {
            TraceFlags(self.0 & !0x01)
        }
    }
}

/// W3C span context: trace_id + span_id + flags + tracestate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanContext {
    /// 128-bit trace id.
    pub trace_id: TraceId,
    /// 64-bit span id (this is the *current* span in the call chain).
    pub span_id: SpanId,
    /// Trace flags (currently only `sampled`).
    pub flags: TraceFlags,
    /// Opaque vendor-specific tracestate values, preserved across hops.
    pub trace_state: Vec<(String, String)>,
    /// True if the context came in over the wire (vs. created locally).
    pub remote: bool,
}

impl SpanContext {
    /// Construct a fresh, locally-created context.
    pub fn new(trace_id: TraceId, span_id: SpanId, flags: TraceFlags) -> Self {
        Self {
            trace_id,
            span_id,
            flags,
            trace_state: Vec::new(),
            remote: false,
        }
    }

    /// The invalid all-zero context.
    pub fn invalid() -> Self {
        Self {
            trace_id: TraceId::INVALID,
            span_id: SpanId::INVALID,
            flags: TraceFlags::default(),
            trace_state: Vec::new(),
            remote: false,
        }
    }

    /// True iff trace_id and span_id are both nonzero.
    pub fn is_valid(&self) -> bool {
        !self.trace_id.is_invalid() && !self.span_id.is_invalid()
    }

    /// Encode as W3C `traceparent` header value.
    ///
    /// Format: `00-<32hex>-<16hex>-<2hex>`.
    pub fn to_traceparent(&self) -> String {
        format!(
            "00-{}-{}-{:02x}",
            self.trace_id.to_hex(),
            self.span_id.to_hex(),
            self.flags.0,
        )
    }

    /// Encode `tracestate` header value, or empty string if no entries.
    pub fn to_tracestate(&self) -> String {
        self.trace_state
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Parse a `traceparent` header value. Strict: only version `00`.
    pub fn from_traceparent(hdr: &str) -> ObsResult<Self> {
        let parts: Vec<&str> = hdr.split('-').collect();
        if parts.len() != 4 {
            return Err(ObsError::Parse(format!(
                "traceparent must have 4 fields, got {}",
                parts.len()
            )));
        }
        if parts[0] != "00" {
            return Err(ObsError::Parse(format!(
                "unsupported traceparent version {:?}",
                parts[0]
            )));
        }
        let trace_id = TraceId::from_hex(parts[1])?;
        let span_id = SpanId::from_hex(parts[2])?;
        if parts[3].len() != 2 {
            return Err(ObsError::Parse(format!(
                "flags must be 2 hex chars, got {}",
                parts[3].len()
            )));
        }
        let flags = u8::from_str_radix(parts[3], 16)
            .map_err(|e| ObsError::Parse(format!("bad flags hex: {e}")))?;
        if trace_id.is_invalid() || span_id.is_invalid() {
            return Err(ObsError::Parse(
                "traceparent had zero trace_id or span_id".to_string(),
            ));
        }
        Ok(Self {
            trace_id,
            span_id,
            flags: TraceFlags(flags),
            trace_state: Vec::new(),
            remote: true,
        })
    }

    /// Merge a `tracestate` header into this context. Re-parses on each call.
    pub fn with_tracestate(mut self, hdr: &str) -> Self {
        if hdr.trim().is_empty() {
            return self;
        }
        for entry in hdr.split(',') {
            if let Some((k, v)) = entry.split_once('=') {
                self.trace_state
                    .push((k.trim().to_string(), v.trim().to_string()));
            }
        }
        self
    }
}

/// W3C baggage: ordered key/value pairs propagated alongside trace context.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Baggage {
    entries: BTreeMap<String, String>,
}

impl Baggage {
    /// Empty baggage.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a key/value pair.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.entries.insert(key.into(), value.into());
    }

    /// Look up a key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(|s| s.as_str())
    }

    /// True if no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over `(key, value)` pairs in canonical (sorted) order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Encode as a `baggage` header value. Values are emitted verbatim;
    /// callers must percent-encode beforehand if they contain `,` or `;`.
    pub fn to_header(&self) -> String {
        self.entries
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Parse a `baggage` header value.
    pub fn from_header(hdr: &str) -> ObsResult<Self> {
        let mut out = Self::new();
        if hdr.trim().is_empty() {
            return Ok(out);
        }
        for entry in hdr.split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let (k, v) = entry.split_once('=').ok_or_else(|| {
                ObsError::Parse(format!("malformed baggage entry: {entry:?}"))
            })?;
            out.insert(k.trim(), v.trim());
        }
        Ok(out)
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(nibble(b >> 4));
        s.push(nibble(b & 0x0f));
    }
    s
}

fn nibble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'a' + (n - 10)) as char,
        _ => '0',
    }
}

fn hex_decode(s: &str, out: &mut [u8]) -> ObsResult<()> {
    if s.len() != out.len() * 2 {
        return Err(ObsError::Parse(format!(
            "hex length {} does not fit {} bytes",
            s.len(),
            out.len()
        )));
    }
    let bytes = s.as_bytes();
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = decode_nibble(bytes[i * 2])?;
        let lo = decode_nibble(bytes[i * 2 + 1])?;
        *slot = (hi << 4) | lo;
    }
    Ok(())
}

fn decode_nibble(b: u8) -> ObsResult<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(ObsError::Parse(format!("invalid hex byte: {b:#x}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traceparent_roundtrip() {
        let ctx = SpanContext::new(
            TraceId([
                0x0a, 0xf7, 0x65, 0x19, 0x16, 0xcd, 0x43, 0xdd, 0x84, 0x48, 0xeb, 0x21,
                0x1c, 0x80, 0x31, 0x9c,
            ]),
            SpanId([0xb7, 0xad, 0x6b, 0x71, 0x69, 0x20, 0x33, 0x31]),
            TraceFlags::SAMPLED,
        );
        let hdr = ctx.to_traceparent();
        assert_eq!(
            hdr,
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        );
        let parsed = SpanContext::from_traceparent(&hdr).expect("parse");
        assert_eq!(parsed.trace_id, ctx.trace_id);
        assert_eq!(parsed.span_id, ctx.span_id);
        assert!(parsed.flags.is_sampled());
        assert!(parsed.remote);
    }

    #[test]
    fn baggage_roundtrip() {
        let mut b = Baggage::new();
        b.insert("user.id", "42");
        b.insert("region", "us-east-1");
        let hdr = b.to_header();
        let parsed = Baggage::from_header(&hdr).expect("parse");
        assert_eq!(parsed.get("user.id"), Some("42"));
        assert_eq!(parsed.get("region"), Some("us-east-1"));
    }

    #[test]
    fn rejects_v01_traceparent() {
        assert!(SpanContext::from_traceparent("01-aa-bb-00").is_err());
    }
}
