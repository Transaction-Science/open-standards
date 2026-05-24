//! Structured-output validator tests.

use eoc_safety::structure::{validate, validate_string};
use serde_json::json;

#[test]
fn valid_object_passes() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer", "minimum": 0, "maximum": 150}
        },
        "required": ["name", "age"],
        "additionalProperties": false
    });
    let value = json!({"name": "Alice", "age": 30});
    let report = validate(&value, &schema);
    assert!(report.valid, "errors: {:?}", report.errors);
}

#[test]
fn missing_required_field_fails() {
    let schema = json!({
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    });
    let value = json!({});
    let report = validate(&value, &schema);
    assert!(!report.valid);
    assert!(report.errors.iter().any(|e| e.message.contains("required")));
}

#[test]
fn wrong_type_fails() {
    let schema = json!({"type": "integer"});
    let value = json!("not an int");
    let report = validate(&value, &schema);
    assert!(!report.valid);
}

#[test]
fn additional_property_rejected() {
    let schema = json!({
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": false
    });
    let value = json!({"a": "x", "b": "y"});
    let report = validate(&value, &schema);
    assert!(!report.valid);
    assert!(report.errors.iter().any(|e| e.path.ends_with("/b")));
}

#[test]
fn enum_matches() {
    let schema = json!({"enum": ["red", "green", "blue"]});
    assert!(validate(&json!("red"), &schema).valid);
    assert!(!validate(&json!("purple"), &schema).valid);
}

#[test]
fn string_pattern_enforced() {
    let schema = json!({"type": "string", "pattern": "^\\d{3}-\\d{4}$"});
    assert!(validate(&json!("555-1234"), &schema).valid);
    assert!(!validate(&json!("nope"), &schema).valid);
}

#[test]
fn validate_string_returns_value() {
    let schema = json!({"type": "object", "properties": {"x": {"type": "integer"}}});
    let v = validate_string(r#"{"x": 7}"#, &schema).expect("ok");
    assert_eq!(v["x"], json!(7));
}

#[test]
fn validate_string_surfaces_error() {
    let schema = json!({"type": "object", "required": ["x"]});
    let err = validate_string("{}", &schema).expect_err("must err");
    let msg = format!("{err}");
    assert!(msg.contains("invalid"), "got: {msg}");
}
