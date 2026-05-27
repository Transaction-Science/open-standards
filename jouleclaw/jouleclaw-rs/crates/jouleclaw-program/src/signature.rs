//! Typed signatures for JouleClaw programs.
//!
//! A [`Signature`] declares the typed input/output contract of a single
//! [`Module`](crate::module::Module). Signatures are the unit the compiler
//! reasons over — they get rendered into prompts, JSON schemas, and (when
//! `decode` is enabled) grammar-constrained decoding masks.
//!
//! Two design choices worth calling out:
//!
//! * **Names matter.** Field names appear in the rendered prompt, in the JSON
//!   schema, and in the wiring graph. Renaming a field is a breaking change.
//! * **Types are deliberately small.** v0.1 covers what DSPy covers in
//!   practice (text, bool, int, float, list, enum, free JSON). Tensors,
//!   embeddings, and tier-specific value types live one layer down in the
//!   cascade, not here.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::{Error, Result};

/// One typed field on a signature.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Field {
    /// Field name. Must be non-empty and unique within its
    /// inputs-or-outputs list.
    pub name: String,
    /// The field's declared type.
    pub ty: FieldType,
    /// Human-readable description. Rendered into the prompt for model-tier
    /// dispatches; ignored by deterministic tiers.
    pub description: String,
}

impl Field {
    /// Build a text field with the given name and description.
    pub fn text(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: FieldType::Text,
            description: description.into(),
        }
    }

    /// Build a bool field with the given name and description.
    pub fn bool(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: FieldType::Bool,
            description: description.into(),
        }
    }

    /// Build an int field with the given name and description.
    pub fn int(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: FieldType::Int,
            description: description.into(),
        }
    }

    /// Build a float field with the given name and description.
    pub fn float(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: FieldType::Float,
            description: description.into(),
        }
    }

    /// Build a `one_of` (enum) field over the supplied variants.
    pub fn one_of<I, S>(name: impl Into<String>, description: impl Into<String>, variants: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            name: name.into(),
            ty: FieldType::OneOf(variants.into_iter().map(Into::into).collect()),
            description: description.into(),
        }
    }

    /// Build a list-of-`inner` field.
    pub fn list(
        name: impl Into<String>,
        description: impl Into<String>,
        inner: FieldType,
    ) -> Self {
        Self {
            name: name.into(),
            ty: FieldType::List(Box::new(inner)),
            description: description.into(),
        }
    }

    /// Build a JSON field with the supplied schema.
    pub fn json(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: JsonValue,
    ) -> Self {
        Self {
            name: name.into(),
            ty: FieldType::Json(schema),
            description: description.into(),
        }
    }
}

/// Declared type of a signature field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FieldType {
    Text,
    Bool,
    Int,
    Float,
    List(Box<FieldType>),
    /// Free JSON with the supplied JSON Schema.
    Json(JsonValue),
    /// Enum over a fixed set of string variants.
    OneOf(Vec<String>),
}

impl FieldType {
    /// Short human label for use in error messages.
    pub fn label(&self) -> String {
        match self {
            FieldType::Text => "text".into(),
            FieldType::Bool => "bool".into(),
            FieldType::Int => "int".into(),
            FieldType::Float => "float".into(),
            FieldType::List(inner) => format!("list<{}>", inner.label()),
            FieldType::Json(_) => "json".into(),
            FieldType::OneOf(variants) => format!("one_of[{}]", variants.join("|")),
        }
    }

    /// Lower this type into a JSON Schema fragment.
    ///
    /// The compiler stitches these fragments together into the full grammar
    /// surface exposed via [`Compiler::compile_to_jsonschema`].
    ///
    /// [`Compiler::compile_to_jsonschema`]: crate::compiler::Compiler::compile_to_jsonschema
    pub fn to_jsonschema(&self) -> JsonValue {
        match self {
            FieldType::Text => serde_json::json!({ "type": "string" }),
            FieldType::Bool => serde_json::json!({ "type": "boolean" }),
            FieldType::Int => serde_json::json!({ "type": "integer" }),
            FieldType::Float => serde_json::json!({ "type": "number" }),
            FieldType::List(inner) => {
                serde_json::json!({ "type": "array", "items": inner.to_jsonschema() })
            }
            FieldType::Json(schema) => schema.clone(),
            FieldType::OneOf(variants) => {
                serde_json::json!({ "type": "string", "enum": variants })
            }
        }
    }
}

/// A typed input/output contract for a single module.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Signature {
    pub name: String,
    pub inputs: Vec<Field>,
    pub outputs: Vec<Field>,
    /// Instruction string. Rendered into the prompt for model-tier dispatches.
    pub instruction: String,
}

impl Signature {
    /// Build a signature.
    pub fn new(
        name: impl Into<String>,
        instruction: impl Into<String>,
        inputs: Vec<Field>,
        outputs: Vec<Field>,
    ) -> Self {
        Self {
            name: name.into(),
            inputs,
            outputs,
            instruction: instruction.into(),
        }
    }

    /// Validate this signature in isolation.
    ///
    /// Checks that:
    ///
    /// * inputs and outputs are non-empty,
    /// * no field name appears twice in the same list,
    /// * every field name is non-empty.
    pub fn validate(&self) -> Result<()> {
        if self.inputs.is_empty() {
            return Err(Error::SignatureMissingInputs {
                name: self.name.clone(),
            });
        }
        if self.outputs.is_empty() {
            return Err(Error::SignatureMissingOutputs {
                name: self.name.clone(),
            });
        }
        check_unique_names(&self.name, &self.inputs)?;
        check_unique_names(&self.name, &self.outputs)?;
        Ok(())
    }

    /// Look up an input field by name.
    pub fn input(&self, name: &str) -> Option<&Field> {
        self.inputs.iter().find(|f| f.name == name)
    }

    /// Look up an output field by name.
    pub fn output(&self, name: &str) -> Option<&Field> {
        self.outputs.iter().find(|f| f.name == name)
    }
}

fn check_unique_names(sig_name: &str, fields: &[Field]) -> Result<()> {
    for (i, f) in fields.iter().enumerate() {
        if f.name.is_empty() {
            return Err(Error::EmptyFieldName {
                field: format!("(unnamed field at index {})", i),
            });
        }
        if fields.iter().skip(i + 1).any(|other| other.name == f.name) {
            return Err(Error::DuplicateField {
                name: sig_name.to_string(),
                field: f.name.clone(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer_sig() -> Signature {
        Signature::new(
            "answer",
            "Answer the question.",
            vec![Field::text("question", "the user's question")],
            vec![Field::text("answer", "the answer text")],
        )
    }

    #[test]
    fn signature_validates_minimal() {
        assert!(answer_sig().validate().is_ok());
    }

    #[test]
    fn signature_rejects_no_inputs() {
        let sig = Signature::new(
            "s",
            "i",
            vec![],
            vec![Field::text("out", "")],
        );
        assert!(matches!(
            sig.validate(),
            Err(Error::SignatureMissingInputs { .. })
        ));
    }

    #[test]
    fn signature_rejects_no_outputs() {
        let sig = Signature::new(
            "s",
            "i",
            vec![Field::text("in", "")],
            vec![],
        );
        assert!(matches!(
            sig.validate(),
            Err(Error::SignatureMissingOutputs { .. })
        ));
    }

    #[test]
    fn signature_rejects_duplicate_inputs() {
        let sig = Signature::new(
            "s",
            "i",
            vec![Field::text("x", ""), Field::int("x", "")],
            vec![Field::text("y", "")],
        );
        assert!(matches!(
            sig.validate(),
            Err(Error::DuplicateField { .. })
        ));
    }

    #[test]
    fn signature_rejects_empty_field_name() {
        let sig = Signature::new(
            "s",
            "i",
            vec![Field::text("", "")],
            vec![Field::text("y", "")],
        );
        assert!(matches!(
            sig.validate(),
            Err(Error::EmptyFieldName { .. })
        ));
    }

    #[test]
    fn jsonschema_lowering_covers_each_variant() {
        assert_eq!(FieldType::Text.to_jsonschema()["type"], "string");
        assert_eq!(FieldType::Bool.to_jsonschema()["type"], "boolean");
        assert_eq!(FieldType::Int.to_jsonschema()["type"], "integer");
        assert_eq!(FieldType::Float.to_jsonschema()["type"], "number");

        let list = FieldType::List(Box::new(FieldType::Int)).to_jsonschema();
        assert_eq!(list["type"], "array");
        assert_eq!(list["items"]["type"], "integer");

        let one_of = FieldType::OneOf(vec!["a".into(), "b".into()]).to_jsonschema();
        assert_eq!(one_of["type"], "string");
        assert_eq!(one_of["enum"], serde_json::json!(["a", "b"]));
    }

    #[test]
    fn field_lookup_works() {
        let sig = answer_sig();
        assert!(sig.input("question").is_some());
        assert!(sig.output("answer").is_some());
        assert!(sig.input("nope").is_none());
    }

    #[test]
    fn field_constructors_yield_expected_types() {
        assert_eq!(Field::text("a", "").ty, FieldType::Text);
        assert_eq!(Field::bool("a", "").ty, FieldType::Bool);
        assert_eq!(Field::int("a", "").ty, FieldType::Int);
        assert_eq!(Field::float("a", "").ty, FieldType::Float);
        match Field::list("a", "", FieldType::Text).ty {
            FieldType::List(inner) => assert_eq!(*inner, FieldType::Text),
            other => panic!("expected list, got {:?}", other),
        }
        match Field::one_of("a", "", ["x", "y"]).ty {
            FieldType::OneOf(v) => assert_eq!(v, vec!["x".to_string(), "y".to_string()]),
            other => panic!("expected one_of, got {:?}", other),
        }
    }
}
