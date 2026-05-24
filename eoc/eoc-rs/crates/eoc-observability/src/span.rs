//! OpenTelemetry-shaped spans.
//!
//! A [`Span`] is the unit of work in a trace. It owns:
//!
//! - identity ([`SpanContext`])
//! - kind ([`SpanKind`])
//! - status ([`Status`])
//! - attributes (string-keyed scalars)
//! - events (timestamped log records)
//! - links (to sibling traces)
//! - timing (start_unix_nano, end_unix_nano)
//! - joule attribution ([`JouleCost`])
//!
//! Spans are *value types*. A [`crate::processor::BatchSpanProcessor`] takes
//! finished spans and hands them to an [`crate::exporter::Exporter`].

use crate::context::{SpanContext, SpanId, TraceFlags, TraceId};
use eoc_core::JouleCost;
use serde::{Deserialize, Serialize};

/// Span kind, per OTel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpanKind {
    /// Default; an internal operation.
    Internal,
    /// Synchronous outbound call.
    Client,
    /// Synchronous inbound handler.
    Server,
    /// Asynchronous producer.
    Producer,
    /// Asynchronous consumer.
    Consumer,
}

impl Default for SpanKind {
    fn default() -> Self {
        SpanKind::Internal
    }
}

/// Status code per OTel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StatusCode {
    /// Default, unset by SDK.
    Unset,
    /// Operation succeeded.
    Ok,
    /// Operation failed.
    Error,
}

impl Default for StatusCode {
    fn default() -> Self {
        StatusCode::Unset
    }
}

/// Span status: a code plus optional human-readable description.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Status {
    /// Status code.
    pub code: StatusCode,
    /// Description (only meaningful when `code == Error`).
    pub description: String,
}

impl Status {
    /// Status with code `Ok`.
    pub fn ok() -> Self {
        Self {
            code: StatusCode::Ok,
            description: String::new(),
        }
    }

    /// Error status with a description.
    pub fn error(desc: impl Into<String>) -> Self {
        Self {
            code: StatusCode::Error,
            description: desc.into(),
        }
    }
}

/// Attribute value. Mirrors OTel's `AnyValue` but restricted to common types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttrValue {
    /// UTF-8 string.
    String(String),
    /// 64-bit signed integer.
    Int(i64),
    /// IEEE-754 double.
    Float(f64),
    /// Boolean.
    Bool(bool),
}

impl AttrValue {
    /// Convert into a JSON-shaped value for export.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            AttrValue::String(s) => serde_json::Value::String(s.clone()),
            AttrValue::Int(i) => serde_json::json!(i),
            AttrValue::Float(f) => serde_json::json!(f),
            AttrValue::Bool(b) => serde_json::json!(b),
        }
    }
}

impl From<&str> for AttrValue {
    fn from(s: &str) -> Self {
        AttrValue::String(s.to_string())
    }
}
impl From<String> for AttrValue {
    fn from(s: String) -> Self {
        AttrValue::String(s)
    }
}
impl From<i64> for AttrValue {
    fn from(i: i64) -> Self {
        AttrValue::Int(i)
    }
}
impl From<u64> for AttrValue {
    fn from(u: u64) -> Self {
        AttrValue::Int(u as i64)
    }
}
impl From<f64> for AttrValue {
    fn from(f: f64) -> Self {
        AttrValue::Float(f)
    }
}
impl From<bool> for AttrValue {
    fn from(b: bool) -> Self {
        AttrValue::Bool(b)
    }
}

/// One timestamped event attached to a span.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Event name.
    pub name: String,
    /// Unix-epoch nanoseconds.
    pub time_unix_nano: u64,
    /// Attributes attached to this event.
    pub attributes: Vec<(String, AttrValue)>,
}

/// A link from this span to another span (possibly in another trace).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpanLink {
    /// Linked span's trace id (hex).
    pub trace_id: String,
    /// Linked span's span id (hex).
    pub span_id: String,
    /// Link attributes.
    pub attributes: Vec<(String, AttrValue)>,
}

/// A completed (or in-progress) span.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// 32-hex trace id.
    pub trace_id: String,
    /// 16-hex span id.
    pub span_id: String,
    /// 16-hex parent span id, or empty.
    pub parent_span_id: String,
    /// Trace flags (low byte).
    pub trace_flags: u8,
    /// Human-readable name (e.g. `"llm.generate"`).
    pub name: String,
    /// Span kind.
    pub kind: SpanKind,
    /// Start timestamp (unix nanos).
    pub start_unix_nano: u64,
    /// End timestamp (unix nanos), or 0 if still open.
    pub end_unix_nano: u64,
    /// Status.
    pub status: Status,
    /// String-keyed attributes.
    pub attributes: Vec<(String, AttrValue)>,
    /// Events.
    pub events: Vec<SpanEvent>,
    /// Links.
    pub links: Vec<SpanLink>,
    /// Joule attribution, if any.
    pub joule_cost: Option<JouleCost>,
}

impl Span {
    /// Build a span from a [`SpanContext`] and (optional) parent.
    pub fn new(
        ctx: &SpanContext,
        parent: Option<&SpanContext>,
        name: impl Into<String>,
        kind: SpanKind,
        start_unix_nano: u64,
    ) -> Self {
        Self {
            trace_id: ctx.trace_id.to_hex(),
            span_id: ctx.span_id.to_hex(),
            parent_span_id: parent
                .map(|p| p.span_id.to_hex())
                .unwrap_or_default(),
            trace_flags: ctx.flags.0,
            name: name.into(),
            kind,
            start_unix_nano,
            end_unix_nano: 0,
            status: Status::default(),
            attributes: Vec::new(),
            events: Vec::new(),
            links: Vec::new(),
            joule_cost: None,
        }
    }

    /// Reconstruct the [`SpanContext`] for this span.
    pub fn context(&self) -> SpanContext {
        let trace_id = TraceId::from_hex(&self.trace_id).unwrap_or(TraceId::INVALID);
        let span_id = SpanId::from_hex(&self.span_id).unwrap_or(SpanId::INVALID);
        SpanContext::new(trace_id, span_id, TraceFlags(self.trace_flags))
    }

    /// Set an attribute, replacing any existing entry with the same key.
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<AttrValue>) {
        let key = key.into();
        let value = value.into();
        if let Some(slot) = self.attributes.iter_mut().find(|(k, _)| k == &key) {
            slot.1 = value;
        } else {
            self.attributes.push((key, value));
        }
    }

    /// Record an event.
    pub fn add_event(
        &mut self,
        name: impl Into<String>,
        time_unix_nano: u64,
        attributes: Vec<(String, AttrValue)>,
    ) {
        self.events.push(SpanEvent {
            name: name.into(),
            time_unix_nano,
            attributes,
        });
    }

    /// Add a link to a sibling span.
    pub fn add_link(&mut self, ctx: &SpanContext, attributes: Vec<(String, AttrValue)>) {
        self.links.push(SpanLink {
            trace_id: ctx.trace_id.to_hex(),
            span_id: ctx.span_id.to_hex(),
            attributes,
        });
    }

    /// Set the span status.
    pub fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    /// Attach joule cost to the span (typically from `eoc-meter`).
    pub fn set_joule_cost(&mut self, cost: JouleCost) {
        self.joule_cost = Some(cost);
    }

    /// Close the span at the given time. Idempotent: only the first call sets the time.
    pub fn end(&mut self, end_unix_nano: u64) {
        if self.end_unix_nano == 0 {
            self.end_unix_nano = end_unix_nano;
        }
    }

    /// True if `end()` has been called with a nonzero timestamp.
    pub fn is_ended(&self) -> bool {
        self.end_unix_nano != 0
    }

    /// Duration in nanoseconds, or 0 if not yet ended.
    pub fn duration_nanos(&self) -> u64 {
        if self.end_unix_nano == 0 {
            0
        } else {
            self.end_unix_nano.saturating_sub(self.start_unix_nano)
        }
    }
}
