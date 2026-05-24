//! LangSmith trace export adapter.
//!
//! LangSmith expects a "run" object per span. We map our [`Span`] onto the
//! minimal LangSmith run schema (`id`, `trace_id`, `parent_run_id`,
//! `name`, `run_type`, `start_time`, `end_time`, `inputs`, `outputs`,
//! `extra`). The result is JSON, hand-off to the user's HTTP transport.

use crate::error::ObsResult;
use crate::exporter::SpanExporter;
use crate::span::{AttrValue, Span, SpanKind, StatusCode};
use std::sync::Mutex;

/// LangSmith run-type, derived from span kind.
fn run_type(kind: SpanKind) -> &'static str {
    match kind {
        SpanKind::Server => "chain",
        SpanKind::Client => "llm",
        SpanKind::Producer => "tool",
        SpanKind::Consumer => "tool",
        SpanKind::Internal => "chain",
    }
}

/// Translate a [`Span`] into a LangSmith run JSON value.
pub fn span_to_run(span: &Span) -> serde_json::Value {
    let mut extra = serde_json::Map::new();
    for (k, v) in span.attributes.iter() {
        extra.insert(k.clone(), v.to_json());
    }
    if let Some(joule) = span.joule_cost {
        extra.insert(
            "eoc.joules.microjoules".to_string(),
            serde_json::json!(joule.microjoules),
        );
        extra.insert(
            "eoc.joules.source".to_string(),
            serde_json::json!(match joule.source {
                eoc_core::JouleSource::Measured => "measured",
                eoc_core::JouleSource::Estimated => "estimated",
            }),
        );
    }

    let events: Vec<serde_json::Value> = span
        .events
        .iter()
        .map(|e| {
            serde_json::json!({
                "name": e.name,
                "time_unix_nano": e.time_unix_nano,
                "attributes": e
                    .attributes
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_json()))
                    .collect::<serde_json::Map<_, _>>(),
            })
        })
        .collect();

    let status = match span.status.code {
        StatusCode::Ok => "success",
        StatusCode::Error => "error",
        StatusCode::Unset => "success",
    };

    let parent = if span.parent_span_id.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::String(span.parent_span_id.clone())
    };

    serde_json::json!({
        "id": span.span_id,
        "trace_id": span.trace_id,
        "parent_run_id": parent,
        "name": span.name,
        "run_type": run_type(span.kind),
        "start_time": ns_to_iso8601(span.start_unix_nano),
        "end_time": if span.end_unix_nano == 0 {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(ns_to_iso8601(span.end_unix_nano))
        },
        "status": status,
        "error": span.status.description,
        "extra": extra,
        "events": events,
    })
}

/// Convert unix-nanos to an RFC3339-ish UTC timestamp string.
///
/// This is intentionally simple: no `chrono` dependency. Returns a string of
/// the form `1970-01-01T00:00:00.000000Z` for nanos within `[1970,9999]`.
fn ns_to_iso8601(nanos: u64) -> String {
    let secs = nanos / 1_000_000_000;
    let sub_micros = (nanos % 1_000_000_000) / 1_000;
    let (y, mo, d, h, mi, se) = unix_to_ymdhms(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{se:02}.{sub_micros:06}Z")
}

/// A tiny `unix_secs -> (y,mo,d,h,mi,se)` implementation, accurate for
/// 1970..=9999. Adapted from civil-from-days (Howard Hinnant style).
fn unix_to_ymdhms(secs_total: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (secs_total / 86_400) as i64;
    let secs = (secs_total % 86_400) as u32;
    let h = secs / 3600;
    let mi = (secs / 60) % 60;
    let se = secs % 60;

    // Convert "days since 1970-01-01" to civil date.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0,399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y_civil = if m <= 2 { y + 1 } else { y };
    (y_civil, m, d, h, mi, se)
}

/// LangSmith-shaped exporter. Stores produced JSON values in memory; users
/// wire up an HTTP `POST` to `${LANGSMITH_ENDPOINT}/runs/batch` themselves.
#[derive(Debug, Default)]
pub struct LangSmithExporter {
    runs: Mutex<Vec<serde_json::Value>>,
}

impl LangSmithExporter {
    /// Construct an empty exporter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot the JSON runs collected so far.
    pub fn finished_runs(&self) -> Vec<serde_json::Value> {
        self.runs.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

impl SpanExporter for LangSmithExporter {
    fn export(&self, batch: &[Span]) -> ObsResult<()> {
        let mut g = self.runs.lock().map_err(|_| {
            crate::error::ObsError::Exporter("langsmith poisoned".to_string())
        })?;
        for s in batch {
            g.push(span_to_run(s));
        }
        Ok(())
    }
}

/// Render an attribute list to a JSON object.
pub fn attrs_to_json(attrs: &[(String, AttrValue)]) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    for (k, v) in attrs.iter() {
        m.insert(k.clone(), v.to_json());
    }
    serde_json::Value::Object(m)
}
