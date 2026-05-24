//! Anthropic `tool_use` adapter.
//!
//! Reference: <https://docs.anthropic.com/en/docs/build-with-claude/tool-use>
//!
//! On the request side, Anthropic accepts a top-level `tools` array
//! with entries shaped:
//!
//! ```json
//! { "name": "...", "description": "...", "input_schema": { ... } }
//! ```
//!
//! On the response side, tool calls appear as `content` blocks of type
//! `tool_use`:
//!
//! ```json
//! { "type": "tool_use", "id": "toolu_...", "name": "...", "input": { ... } }
//! ```

use serde_json::{Value, json};

use crate::error::{ToolError, ToolResult};
use crate::schema::ToolSchema;
use crate::tool::ToolCallRequest;

/// Translate a canonical [`ToolSchema`] into Anthropic's tool-definition
/// JSON.
pub fn to_anthropic_tool(schema: &ToolSchema) -> Value {
    json!({
        "name": schema.name,
        "description": schema.description,
        "input_schema": schema.parameters,
    })
}

/// Emit the full `tools` array for an Anthropic Messages request body.
pub fn tools_array(schemas: &[&ToolSchema]) -> Value {
    Value::Array(schemas.iter().map(|s| to_anthropic_tool(s)).collect())
}

/// Format a tool result as the Anthropic `tool_result` content block that
/// must be sent back to the model in the next `user` message.
pub fn tool_result_block(tool_use_id: &str, output: &Value) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": output.to_string(),
    })
}

/// Extract canonical [`ToolCallRequest`]s from an Anthropic Messages
/// response body.
pub fn parse_tool_calls(response: &Value) -> ToolResult<Vec<ToolCallRequest>> {
    let content = response
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or_else(|| ToolError::VendorParse("missing `content` array".to_string()))?;

    let mut out = Vec::new();
    for block in content {
        let ty = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if ty != "tool_use" {
            continue;
        }
        let id = block
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let name = block
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::VendorParse("tool_use missing `name`".to_string()))?
            .to_string();
        let args = block.get("input").cloned().unwrap_or(json!({}));
        out.push(ToolCallRequest { id, name, args });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_schema_to_input_schema() {
        let s = ToolSchema::new("get_weather", "fetch weather", json!({"type": "object"}));
        let t = to_anthropic_tool(&s);
        assert_eq!(t["name"], "get_weather");
        assert_eq!(t["input_schema"], json!({"type": "object"}));
    }

    #[test]
    fn parses_tool_use_blocks() {
        let resp = json!({
            "id": "msg_1",
            "content": [
                {"type": "text", "text": "thinking…"},
                {"type": "tool_use", "id": "toolu_1", "name": "add",
                 "input": {"a": 1, "b": 2}}
            ]
        });
        let calls = parse_tool_calls(&resp).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "toolu_1");
        assert_eq!(calls[0].name, "add");
        assert_eq!(calls[0].args, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn tool_result_block_shape() {
        let b = tool_result_block("toolu_1", &json!({"sum": 3}));
        assert_eq!(b["type"], "tool_result");
        assert_eq!(b["tool_use_id"], "toolu_1");
    }
}
