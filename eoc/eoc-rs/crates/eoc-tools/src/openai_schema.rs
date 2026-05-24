//! OpenAI `tools` adapter (chat completions / responses API).
//!
//! Reference: <https://platform.openai.com/docs/guides/function-calling>
//!
//! Request shape:
//!
//! ```json
//! { "type": "function",
//!   "function": { "name": "...", "description": "...", "parameters": { ... } } }
//! ```
//!
//! Response shape (chat completions): `choices[0].message.tool_calls` is
//! an array of `{ "id": "call_...", "type": "function",
//! "function": { "name": "...", "arguments": "<stringified JSON>" } }`.

use serde_json::{Value, json};

use crate::error::{ToolError, ToolResult};
use crate::schema::ToolSchema;
use crate::tool::ToolCallRequest;

/// Translate a canonical [`ToolSchema`] into the OpenAI tool-definition
/// JSON.
pub fn to_openai_tool(schema: &ToolSchema) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters,
        }
    })
}

/// Emit the full `tools` array for an OpenAI Chat Completions request.
pub fn tools_array(schemas: &[&ToolSchema]) -> Value {
    Value::Array(schemas.iter().map(|s| to_openai_tool(s)).collect())
}

/// Format a tool result as the OpenAI `tool` role message that must be
/// fed back in the next request.
pub fn tool_result_message(tool_call_id: &str, name: &str, output: &Value) -> Value {
    json!({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": output.to_string(),
    })
}

/// Extract canonical [`ToolCallRequest`]s from an OpenAI Chat
/// Completions response.
pub fn parse_tool_calls(response: &Value) -> ToolResult<Vec<ToolCallRequest>> {
    let choices = response
        .get("choices")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ToolError::VendorParse("missing `choices`".to_string()))?;

    let mut out = Vec::new();
    for choice in choices {
        let calls = choice
            .get("message")
            .and_then(|m| m.get("tool_calls"))
            .and_then(|v| v.as_array());
        let Some(calls) = calls else {
            continue;
        };
        for c in calls {
            let id = c
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let func = c
                .get("function")
                .ok_or_else(|| ToolError::VendorParse("tool_call missing `function`".into()))?;
            let name = func
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::VendorParse("function missing `name`".into()))?
                .to_string();
            // Arguments are sent as a JSON-encoded string per OpenAI's
            // spec; decode best-effort and fall through to raw if not
            // valid JSON.
            let args_raw = func.get("arguments").cloned().unwrap_or(json!("{}"));
            let args = match args_raw {
                Value::String(s) => {
                    serde_json::from_str(&s).unwrap_or(Value::String(s))
                }
                other => other,
            };
            out.push(ToolCallRequest { id, name, args });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_schema_to_function_param() {
        let s = ToolSchema::new("add", "sums", json!({"type": "object"}));
        let t = to_openai_tool(&s);
        assert_eq!(t["type"], "function");
        assert_eq!(t["function"]["name"], "add");
    }

    #[test]
    fn parses_chat_completion_tool_calls() {
        let resp = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": "{\"a\":1,\"b\":2}"
                        }
                    }]
                }
            }]
        });
        let calls = parse_tool_calls(&resp).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "add");
        assert_eq!(calls[0].args, json!({"a": 1, "b": 2}));
    }
}
