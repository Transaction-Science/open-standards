//! Canonical tool schema — name, description, JSON Schema for arguments
//! and (optionally) returns. Generated from Rust types via `schemars`.
//!
//! Operators define a tool's argument shape as an ordinary Rust struct
//! with `serde` annotations + `#[derive(schemars::JsonSchema)]` and call
//! [`schema_for`] to produce the canonical [`ToolSchema`]. From that
//! canonical form, vendor-specific shapes are derived by the
//! `*_schema` modules.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonSchemaValue;

use schemars::JsonSchema;

/// The canonical, vendor-agnostic description of a single tool.
///
/// `parameters` and `returns` are JSON Schema documents (Draft 2020-12
/// compatible). Vendor adapters re-shape this into the wire layout each
/// provider expects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name — matched against `ToolCallRequest::name`.
    pub name: String,
    /// Free-form description shown to the model.
    pub description: String,
    /// JSON Schema for the tool's argument object.
    pub parameters: JsonSchemaValue,
    /// Optional JSON Schema for the tool's return value (advisory; many
    /// vendor APIs do not yet consume this).
    pub returns: Option<JsonSchemaValue>,
}

impl ToolSchema {
    /// Construct a schema directly (for tests / non-Rust-derived tools).
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: JsonSchemaValue,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            returns: None,
        }
    }

    /// Attach a return schema.
    pub fn with_returns(mut self, returns: JsonSchemaValue) -> Self {
        self.returns = Some(returns);
        self
    }
}

/// Derive a canonical [`ToolSchema`] from a Rust argument type `T`.
///
/// `T` must implement `schemars::JsonSchema`. The resulting `parameters`
/// is the JSON Schema produced by `schemars` for `T`.
///
/// ```ignore
/// use schemars::JsonSchema;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Serialize, Deserialize, JsonSchema)]
/// struct AddArgs { a: i64, b: i64 }
///
/// let s = eoc_tools::schema_for::<AddArgs>("add", "Sum two integers.");
/// assert_eq!(s.name, "add");
/// ```
pub fn schema_for<T: JsonSchema>(
    name: impl Into<String>,
    description: impl Into<String>,
) -> ToolSchema {
    let schema = schemars::schema_for!(T);
    // schemars::Schema -> serde_json::Value cannot fail for a well-formed
    // schema; if it ever does, fall back to an empty object schema so the
    // caller still receives a usable ToolSchema.
    let parameters = serde_json::to_value(&schema)
        .unwrap_or_else(|_| serde_json::json!({"type": "object"}));
    ToolSchema::new(name, description, parameters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, JsonSchema)]
    struct AddArgs {
        a: i64,
        b: i64,
    }

    #[test]
    fn derives_schema_from_rust_type() {
        let s = schema_for::<AddArgs>("add", "Sum two integers.");
        assert_eq!(s.name, "add");
        assert!(s.parameters.is_object());
        // schemars Draft 2020-12 output contains a `properties` object.
        let params = s.parameters.as_object().unwrap();
        let props = params
            .get("properties")
            .and_then(|v| v.as_object())
            .expect("properties present");
        assert!(props.contains_key("a"));
        assert!(props.contains_key("b"));
    }
}
