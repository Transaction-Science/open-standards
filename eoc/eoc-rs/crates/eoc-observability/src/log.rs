//! Log records with trace correlation.
//!
//! A [`LogRecord`] is the OTel-shaped log payload: severity + body +
//! attributes + (optional) span context. Loggers emit `LogRecord`s; exporters
//! batch them and ship to a backend.

use crate::context::SpanContext;
use crate::span::AttrValue;
use serde::{Deserialize, Serialize};

/// Severity number, per OTel logs spec (1..=24).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Severity {
    /// 1
    Trace = 1,
    /// 5
    Debug = 5,
    /// 9
    Info = 9,
    /// 13
    Warn = 13,
    /// 17
    Error = 17,
    /// 21
    Fatal = 21,
}

impl Severity {
    /// Canonical string name.
    pub fn name(&self) -> &'static str {
        match self {
            Severity::Trace => "TRACE",
            Severity::Debug => "DEBUG",
            Severity::Info => "INFO",
            Severity::Warn => "WARN",
            Severity::Error => "ERROR",
            Severity::Fatal => "FATAL",
        }
    }
}

/// One OTel log record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRecord {
    /// Unix-epoch nanoseconds.
    pub time_unix_nano: u64,
    /// Severity number.
    pub severity_number: u8,
    /// Severity text (typically `Severity::name()`).
    pub severity_text: String,
    /// Log body (free-form string).
    pub body: String,
    /// 32-hex trace_id, or empty.
    pub trace_id: String,
    /// 16-hex span_id, or empty.
    pub span_id: String,
    /// Trace flags (low byte).
    pub trace_flags: u8,
    /// Structured attributes.
    pub attributes: Vec<(String, AttrValue)>,
}

impl LogRecord {
    /// Build a log record with no trace correlation.
    pub fn new(
        time_unix_nano: u64,
        severity: Severity,
        body: impl Into<String>,
    ) -> Self {
        Self {
            time_unix_nano,
            severity_number: severity as u8,
            severity_text: severity.name().to_string(),
            body: body.into(),
            trace_id: String::new(),
            span_id: String::new(),
            trace_flags: 0,
            attributes: Vec::new(),
        }
    }

    /// Attach a [`SpanContext`] to correlate this log with a trace.
    pub fn with_context(mut self, ctx: &SpanContext) -> Self {
        self.trace_id = ctx.trace_id.to_hex();
        self.span_id = ctx.span_id.to_hex();
        self.trace_flags = ctx.flags.0;
        self
    }

    /// Add a structured attribute.
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<AttrValue>) -> Self {
        self.attributes.push((key.into(), value.into()));
        self
    }

    /// True iff trace_id and span_id are both populated.
    pub fn is_correlated(&self) -> bool {
        !self.trace_id.is_empty() && !self.span_id.is_empty()
    }
}
