//! ACDC schema language — a JSON-Schema dialect.
//!
//! Per ACDC spec §5, schemas are written in a constrained subset of
//! JSON Schema 2020-12 plus the ACDC extension keyword
//! `oneOf:[compactForm, fullForm]` used to mark partially-disclosable
//! attribute groups (consumed by [`crate::selective`]).
//!
//! This module implements a lightweight validator sufficient for the
//! interop test vectors that ship with the spec:
//!
//! * primitive types (`string`, `integer`, `number`, `boolean`,
//!   `object`, `array`);
//! * `required` lists;
//! * nested `properties`;
//! * `format: said` (verifies the value parses as a [`Said`]);
//! * the partial-disclosure `oneOf` ACDC extension.
//!
//! Heavier features (`$ref` resolution, `pattern`, full draft compat)
//! are reserved for future revisions; the substrate's load-bearing rule
//! is that schema enforcement complements (never replaces) SAID
//! verification.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use smart_byte_core::Said;

use crate::error::{AcdcError, Result};

/// JSON Schema dialect tag carried in the ACDC schema `$schema`. We
/// accept any of these for v1.
pub const ACDC_SCHEMA_DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";

/// Identifies which schema dialect a schema declares.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaDialect {
    /// Standard JSON Schema 2020-12.
    Draft202012,
    /// Any other dialect URI — accepted but not specially handled.
    Other(String),
}

/// Constrained set of JSON Schema primitive types understood by this
/// validator.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SchemaType {
    /// JSON object.
    Object,
    /// JSON array.
    Array,
    /// JSON string.
    String,
    /// JSON integer.
    Integer,
    /// JSON number (float).
    Number,
    /// JSON boolean.
    Boolean,
    /// JSON null.
    Null,
}

/// In-memory ACDC schema. The raw JSON is preserved so it can be
/// re-serialised and SAIDed identically to the wire form.
#[derive(Clone, Debug, PartialEq)]
pub struct AcdcSchema {
    /// Schema SAID (the `$id` of the schema document, derived per the
    /// ACDC procedure).
    pub said: Said,
    /// Schema dialect.
    pub dialect: SchemaDialect,
    /// Raw schema body, including `properties`, `required`, etc.
    pub body: serde_json::Map<String, Value>,
}

impl AcdcSchema {
    /// Build a schema from raw JSON. The SAID is computed by JCS over
    /// the schema body with its `$id` replaced by [`crate::SAID_PLACEHOLDER`].
    pub fn from_body(mut body: serde_json::Map<String, Value>) -> Result<Self> {
        let dialect = match body.get("$schema").and_then(|v| v.as_str()) {
            Some(s) if s == ACDC_SCHEMA_DIALECT => SchemaDialect::Draft202012,
            Some(s) => SchemaDialect::Other(s.to_string()),
            None => SchemaDialect::Draft202012,
        };
        // Compute SAID with $id placeheld.
        let original_id = body.insert(
            "$id".into(),
            Value::String(crate::SAID_PLACEHOLDER.into()),
        );
        let bytes = serde_jcs::to_vec(&Value::Object(body.clone()))
            .map_err(|e| AcdcError::Jcs(e.to_string()))?;
        let said = Said::hash(&bytes);
        // Restore (or set) $id to the actual SAID encoded as base32.
        body.insert("$id".into(), Value::String(said.to_base32()));
        let _ = original_id;
        Ok(Self {
            said,
            dialect,
            body,
        })
    }

    /// Verify the schema's `$id` matches its derived SAID.
    pub fn verify_said(&self) -> Result<()> {
        let mut body = self.body.clone();
        body.insert(
            "$id".into(),
            Value::String(crate::SAID_PLACEHOLDER.into()),
        );
        let bytes = serde_jcs::to_vec(&Value::Object(body))
            .map_err(|e| AcdcError::Jcs(e.to_string()))?;
        let computed = Said::hash(&bytes);
        if computed != self.said {
            return Err(AcdcError::SaidMismatch {
                asserted: self.said,
                computed,
            });
        }
        Ok(())
    }
}

/// Validate a JSON object against the schema's `properties` + `required`
/// rules. Returns `Ok(())` if the value satisfies the constraints.
pub fn validate_attributes(
    schema: &AcdcSchema,
    attrs: &serde_json::Map<String, Value>,
) -> Result<()> {
    validate_object(&schema.body, &Value::Object(attrs.clone()))
}

fn validate_object(schema: &serde_json::Map<String, Value>, value: &Value) -> Result<()> {
    // Handle ACDC oneOf partial-disclosure extension. Either branch
    // satisfying the value is enough.
    if let Some(Value::Array(arr)) = schema.get("oneOf") {
        let mut last_err: Option<AcdcError> = None;
        for branch in arr {
            if let Value::Object(b) = branch {
                match validate_object(b, value) {
                    Ok(()) => return Ok(()),
                    Err(e) => last_err = Some(e),
                }
            }
        }
        return Err(last_err.unwrap_or_else(|| {
            AcdcError::SchemaViolation("no oneOf branch matched".into())
        }));
    }

    // Required type.
    if let Some(t) = schema.get("type") {
        check_type(t, value)?;
    }

    // For object schemas: properties + required.
    if matches!(value, Value::Object(_)) {
        let obj = value
            .as_object()
            .ok_or_else(|| AcdcError::SchemaViolation("expected object".into()))?;

        if let Some(Value::Array(req)) = schema.get("required") {
            for r in req {
                let name = r.as_str().ok_or_else(|| {
                    AcdcError::SchemaViolation("required entry must be string".into())
                })?;
                if !obj.contains_key(name) {
                    return Err(AcdcError::SchemaViolation(format!(
                        "missing required property `{name}`"
                    )));
                }
            }
        }

        if let Some(Value::Object(props)) = schema.get("properties") {
            for (name, prop_schema) in props {
                if let Some(v) = obj.get(name) {
                    if let Value::Object(ps) = prop_schema {
                        validate_object(ps, v)?;
                    }
                }
            }
        }
    } else if matches!(value, Value::Array(_)) {
        if let (Some(Value::Object(items_schema)), Some(arr)) =
            (schema.get("items"), value.as_array())
        {
            for v in arr {
                validate_object(items_schema, v)?;
            }
        }
    } else if matches!(value, Value::String(_)) {
        if let Some(Value::String(fmt)) = schema.get("format") {
            if fmt == "said" {
                let s = value
                    .as_str()
                    .ok_or_else(|| AcdcError::SchemaViolation("expected string".into()))?;
                Said::from_base32(s).map_err(|e| {
                    AcdcError::SchemaViolation(format!("format:said invalid: {e}"))
                })?;
            }
        }
    }

    Ok(())
}

fn check_type(t: &Value, value: &Value) -> Result<()> {
    let want = t
        .as_str()
        .ok_or_else(|| AcdcError::SchemaViolation("`type` must be a string".into()))?;
    let ok = match (want, value) {
        ("object", Value::Object(_)) => true,
        ("array", Value::Array(_)) => true,
        ("string", Value::String(_)) => true,
        ("integer", Value::Number(n)) => n.is_i64() || n.is_u64(),
        ("number", Value::Number(_)) => true,
        ("boolean", Value::Bool(_)) => true,
        ("null", Value::Null) => true,
        _ => false,
    };
    if !ok {
        return Err(AcdcError::SchemaViolation(format!(
            "value does not match type {want}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema_with(required: &[&str]) -> AcdcSchema {
        let mut props = serde_json::Map::new();
        props.insert("name".into(), json!({"type": "string"}));
        props.insert("age".into(), json!({"type": "integer"}));
        let body = json!({
            "$schema": ACDC_SCHEMA_DIALECT,
            "title": "Person",
            "type": "object",
            "properties": props,
            "required": required.iter().map(|s| Value::String((*s).into())).collect::<Vec<_>>(),
        });
        AcdcSchema::from_body(body.as_object().expect("obj").clone()).expect("schema")
    }

    #[test]
    fn validates_well_formed() {
        let s = schema_with(&["name"]);
        let mut m = serde_json::Map::new();
        m.insert("name".into(), json!("Alice"));
        m.insert("age".into(), json!(30));
        validate_attributes(&s, &m).expect("ok");
    }

    #[test]
    fn rejects_missing_required() {
        let s = schema_with(&["name", "age"]);
        let mut m = serde_json::Map::new();
        m.insert("name".into(), json!("Alice"));
        let err = validate_attributes(&s, &m).unwrap_err();
        assert!(matches!(err, AcdcError::SchemaViolation(_)));
    }

    #[test]
    fn rejects_bad_type() {
        let s = schema_with(&[]);
        let mut m = serde_json::Map::new();
        m.insert("age".into(), json!("not-a-number"));
        let err = validate_attributes(&s, &m).unwrap_err();
        assert!(matches!(err, AcdcError::SchemaViolation(_)));
    }

    #[test]
    fn schema_said_roundtrips() {
        let s = schema_with(&[]);
        s.verify_said().expect("said matches");
    }
}
