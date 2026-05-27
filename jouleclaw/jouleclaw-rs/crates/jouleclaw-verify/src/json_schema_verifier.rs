//! Lightweight JSON-shape verifier.
//!
//! Checks that the output parses as JSON and that every required
//! top-level `(key, expected_type)` is present with the correct
//! coarse type. **Not** a full JSON Schema draft-2020 implementation
//! — for that, reach for the `jsonschema` crate. This verifier
//! exists to gate model output against the most common L3 schema
//! patterns ("the response must be a JSON object with `usage` of
//! type Number and `text` of type String") without pulling in a
//! heavyweight dependency.

use crate::verifier::{OutputVerifier, VerifyResult};

/// Default microjoule cost charged to a schema verifier touch.
/// Bigger than regex because JSON parsing is heavier.
pub const DEFAULT_SCHEMA_COST_UJ: u64 = 200;

/// Coarse JSON types this verifier knows how to demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonType {
    /// `"foo"`
    String,
    /// `42`, `3.14`
    Number,
    /// `true`, `false`
    Bool,
    /// `[…]`
    Array,
    /// `{…}`
    Object,
}

impl JsonType {
    /// Stable wire name used in failure messages.
    pub fn name(self) -> &'static str {
        match self {
            JsonType::String => "string",
            JsonType::Number => "number",
            JsonType::Bool => "bool",
            JsonType::Array => "array",
            JsonType::Object => "object",
        }
    }

    /// Match a parsed `serde_json::Value` against the expected type.
    pub fn matches(self, value: &serde_json::Value) -> bool {
        match (self, value) {
            (JsonType::String, serde_json::Value::String(_)) => true,
            (JsonType::Number, serde_json::Value::Number(_)) => true,
            (JsonType::Bool, serde_json::Value::Bool(_)) => true,
            (JsonType::Array, serde_json::Value::Array(_)) => true,
            (JsonType::Object, serde_json::Value::Object(_)) => true,
            _ => false,
        }
    }
}

/// Requires the output to be a JSON object containing certain
/// top-level keys with certain coarse types.
#[derive(Debug)]
pub struct JsonSchemaVerifier {
    /// Required `(key, expected_type)` pairs. Order matters only for
    /// the first-failure error message.
    required: Vec<(String, JsonType)>,
    /// Verifier name as it appears in the receipt.
    name: String,
    /// Declared microjoule cost.
    cost_uj: u64,
}

impl JsonSchemaVerifier {
    /// Build a new schema verifier requiring `required` top-level
    /// fields. The default name is `verify:json-schema` and the
    /// default cost is [`DEFAULT_SCHEMA_COST_UJ`].
    pub fn new(required: Vec<(String, JsonType)>) -> Self {
        Self {
            required,
            name: "verify:json-schema".to_string(),
            cost_uj: DEFAULT_SCHEMA_COST_UJ,
        }
    }

    /// Override the verifier name (used in receipts). Convention:
    /// prefix with `verify:`.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Override the declared microjoule cost.
    pub fn with_cost_uj(mut self, cost_uj: u64) -> Self {
        self.cost_uj = cost_uj;
        self
    }
}

impl OutputVerifier for JsonSchemaVerifier {
    fn name(&self) -> &str {
        &self.name
    }

    fn verify(&self, output: &[u8]) -> VerifyResult {
        let value: serde_json::Value = match serde_json::from_slice(output) {
            Ok(v) => v,
            Err(e) => return VerifyResult::fail(format!("invalid JSON: {e}")),
        };
        let obj = match value.as_object() {
            Some(o) => o,
            None => return VerifyResult::fail("output is not a JSON object"),
        };
        for (key, expected) in &self.required {
            match obj.get(key) {
                None => {
                    return VerifyResult::fail(format!("missing required field `{key}`"));
                }
                Some(v) if !expected.matches(v) => {
                    return VerifyResult::fail(format!(
                        "field `{key}` has wrong type — expected {}",
                        expected.name()
                    ));
                }
                Some(_) => {}
            }
        }
        VerifyResult::Pass
    }

    fn declared_cost_uj(&self) -> u64 {
        self.cost_uj
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_json_with_required_fields_passes() {
        let v = JsonSchemaVerifier::new(vec![
            ("text".to_string(), JsonType::String),
            ("usage".to_string(), JsonType::Number),
            ("done".to_string(), JsonType::Bool),
        ])
        .named("verify:schema/openai");
        let out = br#"{"text":"hello","usage":42,"done":true,"extra":"ok"}"#;
        assert_eq!(v.verify(out), VerifyResult::Pass);
        assert_eq!(v.name(), "verify:schema/openai");
    }

    #[test]
    fn missing_required_field_fails() {
        let v = JsonSchemaVerifier::new(vec![
            ("text".to_string(), JsonType::String),
            ("usage".to_string(), JsonType::Number),
        ]);
        let out = br#"{"text":"hello"}"#;
        match v.verify(out) {
            VerifyResult::Fail { reason } => {
                assert!(reason.contains("missing required field `usage`"));
            }
            VerifyResult::Pass => panic!("expected Fail"),
        }
    }

    #[test]
    fn wrong_type_fails() {
        let v = JsonSchemaVerifier::new(vec![("usage".to_string(), JsonType::Number)]);
        let out = br#"{"usage":"forty-two"}"#;
        match v.verify(out) {
            VerifyResult::Fail { reason } => {
                assert!(reason.contains("wrong type"));
                assert!(reason.contains("number"));
            }
            VerifyResult::Pass => panic!("expected Fail"),
        }
    }

    #[test]
    fn non_object_top_level_fails() {
        let v = JsonSchemaVerifier::new(vec![]);
        let out = br#"[1, 2, 3]"#;
        match v.verify(out) {
            VerifyResult::Fail { reason } => assert!(reason.contains("not a JSON object")),
            VerifyResult::Pass => panic!("expected Fail"),
        }
    }

    #[test]
    fn invalid_json_fails() {
        let v = JsonSchemaVerifier::new(vec![]);
        let out = b"not json {{{";
        match v.verify(out) {
            VerifyResult::Fail { reason } => assert!(reason.contains("invalid JSON")),
            VerifyResult::Pass => panic!("expected Fail"),
        }
    }

    #[test]
    fn array_and_object_types_match() {
        let v = JsonSchemaVerifier::new(vec![
            ("xs".to_string(), JsonType::Array),
            ("meta".to_string(), JsonType::Object),
        ]);
        let out = br#"{"xs":[1,2],"meta":{"k":"v"}}"#;
        assert_eq!(v.verify(out), VerifyResult::Pass);
    }
}
