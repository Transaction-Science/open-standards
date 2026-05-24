//! Structured-output validation.
//!
//! Minimal, dependency-free JSON-Schema-subset validator. Supports the
//! draft-07 keywords most commonly emitted by structured-output
//! frameworks (Outlines, Instructor, Guidance):
//!
//! - `type` (`"object"`, `"array"`, `"string"`, `"integer"`, `"number"`,
//!   `"boolean"`, `"null"`).
//! - `properties`, `required`, `additionalProperties` (object).
//! - `items` (array).
//! - `enum`, `const` (any).
//! - `minLength`, `maxLength`, `pattern` (string).
//! - `minimum`, `maximum` (number).
//!
//! Real deployments should plug in a full draft-2020-12 validator
//! (e.g. `jsonschema` crate) behind the same [`StructureValidator`]
//! trait shape.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Result, SafetyError};

/// One validation problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// JSON-Pointer to the offending location.
    pub path: String,
    /// Human-readable message.
    pub message: String,
}

/// Result of validating a value against a schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// True iff the value satisfied the schema.
    pub valid: bool,
    /// Collected errors (empty when [`Self::valid`] is true).
    pub errors: Vec<ValidationError>,
}

/// Validate `value` against `schema`. Both are `serde_json::Value`s.
pub fn validate(value: &Value, schema: &Value) -> ValidationReport {
    let mut errors: Vec<ValidationError> = Vec::new();
    check(value, schema, "", &mut errors);
    ValidationReport {
        valid: errors.is_empty(),
        errors,
    }
}

/// Validate `raw_json` (a string) against `schema` and return an
/// `Err(SafetyError::Structure(..))` on failure.
pub fn validate_string(raw_json: &str, schema: &Value) -> Result<Value> {
    let value: Value = serde_json::from_str(raw_json)?;
    let report = validate(&value, schema);
    if !report.valid {
        let summary = report
            .errors
            .iter()
            .map(|e| format!("{}: {}", e.path, e.message))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(SafetyError::Structure(summary));
    }
    Ok(value)
}

fn check(value: &Value, schema: &Value, path: &str, errors: &mut Vec<ValidationError>) {
    let Some(schema_obj) = schema.as_object() else {
        return;
    };

    if let Some(c) = schema_obj.get("const") {
        if value != c {
            errors.push(ValidationError {
                path: path.into(),
                message: format!("const mismatch: expected {}", c),
            });
        }
    }
    if let Some(Value::Array(allowed)) = schema_obj.get("enum") {
        if !allowed.contains(value) {
            errors.push(ValidationError {
                path: path.into(),
                message: "value not in enum".into(),
            });
        }
    }

    if let Some(Value::String(t)) = schema_obj.get("type") {
        if !type_matches(t, value) {
            errors.push(ValidationError {
                path: path.into(),
                message: format!("expected type {}", t),
            });
            return;
        }
    }

    match value {
        Value::Object(map) => {
            if let Some(Value::Object(props)) = schema_obj.get("properties") {
                for (k, sub_schema) in props {
                    if let Some(child) = map.get(k) {
                        let child_path = format!("{}/{}", path, k);
                        check(child, sub_schema, &child_path, errors);
                    }
                }
            }
            if let Some(Value::Array(req)) = schema_obj.get("required") {
                for r in req {
                    if let Some(name) = r.as_str() {
                        if !map.contains_key(name) {
                            errors.push(ValidationError {
                                path: format!("{}/{}", path, name),
                                message: "required property missing".into(),
                            });
                        }
                    }
                }
            }
            if let Some(Value::Bool(false)) = schema_obj.get("additionalProperties") {
                if let Some(Value::Object(props)) = schema_obj.get("properties") {
                    for k in map.keys() {
                        if !props.contains_key(k) {
                            errors.push(ValidationError {
                                path: format!("{}/{}", path, k),
                                message: "additional property not permitted".into(),
                            });
                        }
                    }
                }
            }
        }
        Value::Array(items) => {
            if let Some(sub_schema) = schema_obj.get("items") {
                for (i, item) in items.iter().enumerate() {
                    let child_path = format!("{}/{}", path, i);
                    check(item, sub_schema, &child_path, errors);
                }
            }
        }
        Value::String(s) => {
            if let Some(n) = schema_obj.get("minLength").and_then(|v| v.as_u64()) {
                if (s.chars().count() as u64) < n {
                    errors.push(ValidationError {
                        path: path.into(),
                        message: format!("string shorter than minLength {}", n),
                    });
                }
            }
            if let Some(n) = schema_obj.get("maxLength").and_then(|v| v.as_u64()) {
                if (s.chars().count() as u64) > n {
                    errors.push(ValidationError {
                        path: path.into(),
                        message: format!("string longer than maxLength {}", n),
                    });
                }
            }
            if let Some(Value::String(pat)) = schema_obj.get("pattern") {
                match regex::Regex::new(pat) {
                    Ok(re) => {
                        if !re.is_match(s) {
                            errors.push(ValidationError {
                                path: path.into(),
                                message: format!("string does not match pattern /{}/", pat),
                            });
                        }
                    }
                    Err(e) => {
                        errors.push(ValidationError {
                            path: path.into(),
                            message: format!("invalid pattern in schema: {}", e),
                        });
                    }
                }
            }
        }
        Value::Number(n) => {
            if let Some(min) = schema_obj.get("minimum").and_then(|v| v.as_f64()) {
                if let Some(actual) = n.as_f64() {
                    if actual < min {
                        errors.push(ValidationError {
                            path: path.into(),
                            message: format!("number {} below minimum {}", actual, min),
                        });
                    }
                }
            }
            if let Some(max) = schema_obj.get("maximum").and_then(|v| v.as_f64()) {
                if let Some(actual) = n.as_f64() {
                    if actual > max {
                        errors.push(ValidationError {
                            path: path.into(),
                            message: format!("number {} above maximum {}", actual, max),
                        });
                    }
                }
            }
        }
        _ => {}
    }
}

fn type_matches(t: &str, v: &Value) -> bool {
    match t {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "integer" => v.as_i64().is_some() || v.as_u64().is_some(),
        "number" => v.is_number(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        _ => true,
    }
}
