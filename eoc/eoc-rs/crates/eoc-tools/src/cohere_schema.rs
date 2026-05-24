//! Cohere tool-use / connectors adapter.
//!
//! Reference: <https://docs.cohere.com/docs/tool-use>
//!
//! Cohere Chat v1 ships a distinct shape:
//!
//! Request `tools`:
//!
//! ```json
//! { "name": "...", "description": "...",
//!   "parameter_definitions": {
//!     "<param>": { "type": "str", "description": "...", "required": true }
//!   } }
//! ```
//!
//! Response `tool_calls`:
//!
//! ```json
//! { "tool_calls": [{ "name": "...", "parameters": { ... } }] }
//! ```
//!
//! Cohere uses a flatter parameter-definitions object rather than a full
//! JSON Schema. We translate from the canonical JSON Schema by reading
//! its top-level `properties` and `required` arrays.

use serde_json::{Map, Value, json};

use crate::error::{ToolError, ToolResult};
use crate::schema::ToolSchema;
use crate::tool::ToolCallRequest;

fn json_type_to_cohere(json_type: &str) -> &'static str {
    match json_type {
        "string" => "str",
        "integer" => "int",
        "number" => "float",
        "boolean" => "bool",
        "array" => "list",
        "object" => "dict",
        _ => "str",
    }
}

/// Translate canonical schema to Cohere tool-definition JSON.
pub fn to_cohere_tool(schema: &ToolSchema) -> Value {
    let mut param_defs = Map::new();
    let required: Vec<String> = schema
        .parameters
        .get("required")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    if let Some(props) = schema.parameters.get("properties").and_then(|v| v.as_object()) {
        for (name, sub) in props {
            let ty = sub
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("string");
            let desc = sub
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            param_defs.insert(
                name.clone(),
                json!({
                    "description": desc,
                    "type": json_type_to_cohere(ty),
                    "required": required.iter().any(|r| r == name),
                }),
            );
        }
    }

    json!({
        "name": schema.name,
        "description": schema.description,
        "parameter_definitions": Value::Object(param_defs),
    })
}

/// Emit the `tools` array for a Cohere chat request.
pub fn tools_array(schemas: &[&ToolSchema]) -> Value {
    Value::Array(schemas.iter().map(|s| to_cohere_tool(s)).collect())
}

/// Format a tool result as a Cohere `tool_results` entry.
pub fn tool_result_entry(call: &ToolCallRequest, output: &Value) -> Value {
    json!({
        "call": {
            "name": call.name,
            "parameters": call.args,
        },
        "outputs": [ output ]
    })
}

/// Extract canonical [`ToolCallRequest`]s from a Cohere chat response.
pub fn parse_tool_calls(response: &Value) -> ToolResult<Vec<ToolCallRequest>> {
    let calls = response
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ToolError::VendorParse("missing `tool_calls`".to_string()))?;

    let mut out = Vec::new();
    for c in calls {
        let name = c
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::VendorParse("tool_call missing `name`".into()))?
            .to_string();
        let args = c.get("parameters").cloned().unwrap_or(json!({}));
        out.push(ToolCallRequest {
            id: String::new(),
            name,
            args,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_properties_to_parameter_definitions() {
        let schema = ToolSchema::new(
            "get_weather",
            "fetch weather",
            json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "fahrenheit": {"type": "boolean", "description": "Use F"}
                },
                "required": ["city"]
            }),
        );
        let t = to_cohere_tool(&schema);
        assert_eq!(t["name"], "get_weather");
        let defs = &t["parameter_definitions"];
        assert_eq!(defs["city"]["type"], "str");
        assert_eq!(defs["city"]["required"], true);
        assert_eq!(defs["fahrenheit"]["type"], "bool");
        assert_eq!(defs["fahrenheit"]["required"], false);
    }

    #[test]
    fn parses_tool_calls() {
        let resp = json!({
            "tool_calls": [
                {"name": "search", "parameters": {"q": "rust"}}
            ]
        });
        let calls = parse_tool_calls(&resp).unwrap();
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].args, json!({"q": "rust"}));
    }
}
