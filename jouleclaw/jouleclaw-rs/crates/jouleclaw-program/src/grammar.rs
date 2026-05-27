//! Grammar handle for constrained-decoding integration.
//!
//! v0.1 ships a placeholder newtype. When the `decode` feature is enabled in
//! a later phase, this will be a thin re-export of
//! `jouleclaw_decode::GrammarHandle`. The compiler emits one of these per
//! [`Dispatch`](crate::compiler::Dispatch) so that L3+ tiers can mask token
//! probabilities to the output signature's grammar.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Opaque handle into a grammar definition.
///
/// The payload is the JSON Schema for the output signature. Tiers that
/// support constrained decoding compile the schema into a token-mask; tiers
/// that don't ignore the handle and parse the model output post-hoc.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrammarHandle {
    /// The JSON Schema that constrains decoded tokens.
    pub schema: JsonValue,
    /// Optional human-readable label for telemetry.
    pub label: String,
}

impl GrammarHandle {
    /// Build a grammar handle from a JSON Schema.
    pub fn new(schema: JsonValue, label: impl Into<String>) -> Self {
        Self {
            schema,
            label: label.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_round_trips_through_serde() {
        let h = GrammarHandle::new(serde_json::json!({"type": "object"}), "sig");
        let s = serde_json::to_string(&h).expect("serialize");
        let r: GrammarHandle = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(h, r);
    }
}
