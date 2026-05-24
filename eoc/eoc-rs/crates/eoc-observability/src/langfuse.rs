//! LangFuse trace export adapter.
//!
//! LangFuse models traces as a tree of `observations`, each tagged
//! `GENERATION` | `SPAN` | `EVENT`. We pick:
//!
//! - `GENERATION` if the span has any `gen_ai.*` attribute
//! - `SPAN` otherwise

use crate::error::ObsResult;
use crate::exporter::SpanExporter;
use crate::langsmith::attrs_to_json;
use crate::span::Span;
use std::sync::Mutex;

/// Decide the LangFuse observation type based on attributes.
fn observation_type(span: &Span) -> &'static str {
    let has_genai = span
        .attributes
        .iter()
        .any(|(k, _)| k.starts_with("gen_ai."));
    if has_genai { "GENERATION" } else { "SPAN" }
}

/// Translate a span into a LangFuse observation JSON value.
pub fn span_to_observation(span: &Span) -> serde_json::Value {
    let mut metadata = serde_json::Map::new();
    if let Some(j) = span.joule_cost {
        metadata.insert(
            "eoc.joules.microjoules".to_string(),
            serde_json::json!(j.microjoules),
        );
    }

    let mut model = serde_json::Value::Null;
    let mut usage_input: Option<i64> = None;
    let mut usage_output: Option<i64> = None;
    let mut usage_total: Option<i64> = None;
    for (k, v) in span.attributes.iter() {
        match k.as_str() {
            "gen_ai.response.model" | "gen_ai.request.model" => {
                model = v.to_json();
            }
            "gen_ai.usage.prompt_tokens" => {
                if let crate::span::AttrValue::Int(i) = v {
                    usage_input = Some(*i);
                }
            }
            "gen_ai.usage.completion_tokens" => {
                if let crate::span::AttrValue::Int(i) = v {
                    usage_output = Some(*i);
                }
            }
            "gen_ai.usage.total_tokens" => {
                if let crate::span::AttrValue::Int(i) = v {
                    usage_total = Some(*i);
                }
            }
            _ => {}
        }
    }

    let mut usage = serde_json::Map::new();
    if let Some(i) = usage_input {
        usage.insert("input".to_string(), serde_json::json!(i));
    }
    if let Some(o) = usage_output {
        usage.insert("output".to_string(), serde_json::json!(o));
    }
    if let Some(t) = usage_total {
        usage.insert("total".to_string(), serde_json::json!(t));
    }
    if let Some(j) = span.joule_cost {
        usage.insert(
            "eoc_microjoules".to_string(),
            serde_json::json!(j.microjoules),
        );
    }

    let parent = if span.parent_span_id.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::String(span.parent_span_id.clone())
    };

    let level = match span.status.code {
        crate::span::StatusCode::Error => "ERROR",
        _ => "DEFAULT",
    };

    serde_json::json!({
        "id": span.span_id,
        "trace_id": span.trace_id,
        "parent_observation_id": parent,
        "type": observation_type(span),
        "name": span.name,
        "start_time_unix_nano": span.start_unix_nano,
        "end_time_unix_nano": if span.end_unix_nano == 0 {
            serde_json::Value::Null
        } else {
            serde_json::Value::Number(span.end_unix_nano.into())
        },
        "level": level,
        "status_message": span.status.description,
        "model": model,
        "usage": usage,
        "input": attrs_to_json(&[]),
        "output": serde_json::Value::Null,
        "metadata": metadata,
        "attributes": attrs_to_json(&span.attributes),
    })
}

/// LangFuse-shaped exporter (collects JSON in memory).
#[derive(Debug, Default)]
pub struct LangFuseExporter {
    observations: Mutex<Vec<serde_json::Value>>,
}

impl LangFuseExporter {
    /// Construct an empty exporter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot collected observations.
    pub fn finished_observations(&self) -> Vec<serde_json::Value> {
        self.observations
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }
}

impl SpanExporter for LangFuseExporter {
    fn export(&self, batch: &[Span]) -> ObsResult<()> {
        let mut g = self.observations.lock().map_err(|_| {
            crate::error::ObsError::Exporter("langfuse poisoned".to_string())
        })?;
        for s in batch {
            g.push(span_to_observation(s));
        }
        Ok(())
    }
}
