//! Conventions shared across every schema (Section 3.1).
//!
//! Every schema begins with `schema_version` and ends with `metadata`.
//! The v6 spec uses Pydantic's `dict[str, Any]`; in Rust we use
//! `serde_json::Map<String, Value>`, which is the natural shape for
//! arbitrary tagged JSON without forcing every consumer to roll their
//! own typed extension envelope.

use serde::{Deserialize, Serialize};

/// Free-form metadata bag, always last field of every schema.
pub type Metadata = serde_json::Map<String, serde_json::Value>;

/// Wraps a `&'static str` schema-version tag with serde glue. Schema
/// versions are spec-defined constants (e.g. `"2.0"` for QueryPlan,
/// `"5.0"` for KnowledgeAxes, `"6.0"` for ClaimAttribution).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SchemaVersion(pub String);

impl SchemaVersion {
    pub fn from_static(s: &'static str) -> Self {
        Self(s.to_string())
    }
}

impl From<&'static str> for SchemaVersion {
    fn from(s: &'static str) -> Self {
        Self(s.to_string())
    }
}

impl std::fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
