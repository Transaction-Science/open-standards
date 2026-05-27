//! Runtime values flowing through a compiled program.
//!
//! A [`Record`] is a `HashMap<String, Value>`. A [`Value`] is the runtime
//! complement of [`FieldType`](crate::signature::FieldType): a small enum over
//! scalar, list, and free-JSON values.
//!
//! [`Value`] is intentionally *not* `serde_json::Value` directly so that the
//! runtime can keep a typed Bool/Int/Float/Text distinction without leaning on
//! serde_json's number coercion rules. JSON-typed fields stash their payload
//! in [`Value::Json`] and let the caller schema-check.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::{Error, Result};
use crate::signature::FieldType;

/// Runtime value carried by a [`Record`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Value {
    Text(String),
    Bool(bool),
    Int(i64),
    Float(f64),
    List(Vec<Value>),
    Json(JsonValue),
    /// String value bound by a [`FieldType::OneOf`] enum.
    Enum(String),
}

impl Value {
    /// Short human label for use in error messages.
    pub fn label(&self) -> &'static str {
        match self {
            Value::Text(_) => "text",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::List(_) => "list",
            Value::Json(_) => "json",
            Value::Enum(_) => "enum",
        }
    }

    /// Convenience text constructor.
    pub fn text(s: impl Into<String>) -> Self {
        Value::Text(s.into())
    }

    /// Convenience enum constructor.
    pub fn enum_variant(s: impl Into<String>) -> Self {
        Value::Enum(s.into())
    }

    /// Lower a value to JSON for serialization-friendly logging.
    pub fn to_json(&self) -> JsonValue {
        match self {
            Value::Text(s) => JsonValue::String(s.clone()),
            Value::Bool(b) => JsonValue::Bool(*b),
            Value::Int(i) => JsonValue::Number((*i).into()),
            Value::Float(f) => serde_json::Number::from_f64(*f)
                .map(JsonValue::Number)
                .unwrap_or(JsonValue::Null),
            Value::List(items) => JsonValue::Array(items.iter().map(Value::to_json).collect()),
            Value::Json(v) => v.clone(),
            Value::Enum(s) => JsonValue::String(s.clone()),
        }
    }

    /// Check this value against a declared [`FieldType`].
    pub fn matches(&self, ty: &FieldType) -> bool {
        match (self, ty) {
            (Value::Text(_), FieldType::Text) => true,
            (Value::Bool(_), FieldType::Bool) => true,
            (Value::Int(_), FieldType::Int) => true,
            (Value::Float(_), FieldType::Float) => true,
            (Value::Int(_), FieldType::Float) => true, // ints widen
            (Value::List(items), FieldType::List(inner)) => {
                items.iter().all(|v| v.matches(inner))
            }
            (Value::Json(_), FieldType::Json(_)) => true,
            (Value::Enum(s), FieldType::OneOf(variants)) => variants.iter().any(|v| v == s),
            (Value::Text(s), FieldType::OneOf(variants)) => variants.iter().any(|v| v == s),
            _ => false,
        }
    }

    /// Type-check a value against a field type, returning a typed error if
    /// the value doesn't match.
    pub fn check(&self, ty: &FieldType, module: &str, field: &str) -> Result<()> {
        if self.matches(ty) {
            Ok(())
        } else {
            Err(Error::TypeMismatch {
                module: module.to_string(),
                field: field.to_string(),
                expected: ty.label(),
                actual: self.label().to_string(),
            })
        }
    }
}

/// Bag of named values flowing through a compiled program.
///
/// Records are passed *by value* between modules — the runner accumulates a
/// single global record by merging each module's outputs back in.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub values: HashMap<String, Value>,
}

impl Record {
    /// Empty record.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a value, returning the previous one if any.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) -> Option<Value> {
        self.values.insert(key.into(), value)
    }

    /// Look up a value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    /// Number of bindings.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether this record holds no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Merge `other` into self, overwriting on conflict. Used by the runner
    /// to fold a module's outputs back into the global program record.
    pub fn merge(&mut self, other: Record) {
        for (k, v) in other.values {
            self.values.insert(k, v);
        }
    }

    /// Convenience builder.
    pub fn with(mut self, key: impl Into<String>, value: Value) -> Self {
        self.insert(key, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_round_trip_basic() {
        let mut r = Record::new();
        r.insert("a", Value::text("hi"));
        r.insert("b", Value::Int(7));
        assert_eq!(r.get("a"), Some(&Value::Text("hi".into())));
        assert_eq!(r.get("b"), Some(&Value::Int(7)));
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn merge_overwrites() {
        let mut a = Record::new();
        a.insert("x", Value::Int(1));
        let mut b = Record::new();
        b.insert("x", Value::Int(2));
        a.merge(b);
        assert_eq!(a.get("x"), Some(&Value::Int(2)));
    }

    #[test]
    fn value_matches_basic_types() {
        assert!(Value::Text("a".into()).matches(&FieldType::Text));
        assert!(Value::Bool(true).matches(&FieldType::Bool));
        assert!(Value::Int(3).matches(&FieldType::Int));
        assert!(Value::Float(1.0).matches(&FieldType::Float));
        assert!(Value::Int(3).matches(&FieldType::Float)); // int widens
        assert!(!Value::Float(1.0).matches(&FieldType::Int));
        assert!(!Value::Text("a".into()).matches(&FieldType::Int));
    }

    #[test]
    fn value_matches_one_of() {
        let ty = FieldType::OneOf(vec!["a".into(), "b".into()]);
        assert!(Value::Enum("a".into()).matches(&ty));
        assert!(Value::Text("a".into()).matches(&ty));
        assert!(!Value::Enum("z".into()).matches(&ty));
    }

    #[test]
    fn value_matches_list_recursively() {
        let ty = FieldType::List(Box::new(FieldType::Int));
        let ok = Value::List(vec![Value::Int(1), Value::Int(2)]);
        let bad = Value::List(vec![Value::Int(1), Value::Text("x".into())]);
        assert!(ok.matches(&ty));
        assert!(!bad.matches(&ty));
    }

    #[test]
    fn value_check_returns_type_mismatch() {
        let err = Value::Text("x".into())
            .check(&FieldType::Int, "m", "f")
            .unwrap_err();
        assert!(matches!(err, Error::TypeMismatch { .. }));
    }

    #[test]
    fn value_to_json_round_trips_through_serde() {
        let v = Value::List(vec![Value::Int(1), Value::Bool(true)]);
        let j = v.to_json();
        assert_eq!(j, serde_json::json!([1, true]));
    }
}
