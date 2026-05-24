//! Google Gemini `functionDeclarations` adapter.
//!
//! Reference: <https://ai.google.dev/gemini-api/docs/function-calling>
//!
//! Request shape:
//!
//! ```json
//! { "tools": [{
//!     "functionDeclarations": [
//!       { "name": "...", "description": "...", "parameters": { ... } }
//!     ]
//!   }] }
//! ```
//!
//! Response shape: candidates contain `content.parts[*].functionCall`
//! with `{ "name": "...", "args": { ... } }`. Gemini does not surface a
//! call id, so we synthesise an empty string.

use serde_json::{Value, json};

use crate::error::{ToolError, ToolResult};
use crate::schema::ToolSchema;
use crate::tool::ToolCallRequest;

/// Translate a canonical [`ToolSchema`] to a Gemini `FunctionDeclaration`.
pub fn to_function_declaration(schema: &ToolSchema) -> Value {
    json!({
        "name": schema.name,
        "description": schema.description,
        "parameters": schema.parameters,
    })
}

/// Emit the `tools` array Gemini expects (a single object wrapping
/// `functionDeclarations`).
pub fn tools_array(schemas: &[&ToolSchema]) -> Value {
    let decls: Vec<Value> = schemas.iter().map(|s| to_function_declaration(s)).collect();
    json!([{ "functionDeclarations": decls }])
}

/// Format a tool result as a Gemini `functionResponse` content part.
pub fn function_response_part(name: &str, output: &Value) -> Value {
    json!({
        "functionResponse": {
            "name": name,
            "response": output,
        }
    })
}

/// Extract canonical [`ToolCallRequest`]s from a Gemini
/// `generateContent` response body.
pub fn parse_tool_calls(response: &Value) -> ToolResult<Vec<ToolCallRequest>> {
    let candidates = response
        .get("candidates")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ToolError::VendorParse("missing `candidates`".to_string()))?;

    let mut out = Vec::new();
    for cand in candidates {
        let parts = cand
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|v| v.as_array());
        let Some(parts) = parts else { continue };
        for part in parts {
            if let Some(fc) = part.get("functionCall") {
                let name = fc
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ToolError::VendorParse("functionCall missing `name`".into())
                    })?
                    .to_string();
                let args = fc.get("args").cloned().unwrap_or(json!({}));
                out.push(ToolCallRequest {
                    id: String::new(),
                    name,
                    args,
                });
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_declarations_in_tools_array() {
        let s = ToolSchema::new("add", "sums", json!({"type": "object"}));
        let t = tools_array(&[&s]);
        let arr = t.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        let decls = arr[0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(decls[0]["name"], "add");
    }

    #[test]
    fn parses_function_calls() {
        let resp = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"functionCall": {"name": "add", "args": {"a": 1, "b": 2}}}
                    ]
                }
            }]
        });
        let calls = parse_tool_calls(&resp).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "add");
        assert_eq!(calls[0].args, json!({"a": 1, "b": 2}));
    }
}
