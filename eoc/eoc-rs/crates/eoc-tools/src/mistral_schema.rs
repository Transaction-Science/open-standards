//! Mistral function-calling adapter.
//!
//! Reference: <https://docs.mistral.ai/capabilities/function_calling/>
//!
//! Mistral's wire shape is essentially identical to OpenAI's chat
//! completions tools schema (an unsurprising consequence of the
//! OpenAI-compatible API surface most providers now ship). We re-export
//! the OpenAI translators here so callers can address each vendor with
//! its proper module name without re-implementing identical code.

use serde_json::Value;

use crate::error::ToolResult;
use crate::schema::ToolSchema;
use crate::tool::ToolCallRequest;

/// Translate canonical schema to Mistral tool-definition JSON.
pub fn to_mistral_tool(schema: &ToolSchema) -> Value {
    crate::openai_schema::to_openai_tool(schema)
}

/// Emit the `tools` array for a Mistral chat completions request.
pub fn tools_array(schemas: &[&ToolSchema]) -> Value {
    crate::openai_schema::tools_array(schemas)
}

/// Format a tool result as the `tool` role message Mistral expects.
pub fn tool_result_message(tool_call_id: &str, name: &str, output: &Value) -> Value {
    crate::openai_schema::tool_result_message(tool_call_id, name, output)
}

/// Extract canonical [`ToolCallRequest`]s from a Mistral chat
/// completions response.
pub fn parse_tool_calls(response: &Value) -> ToolResult<Vec<ToolCallRequest>> {
    crate::openai_schema::parse_tool_calls(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_mistral_style_tool_calls() {
        let resp = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_m1",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"id\":42}"
                        }
                    }]
                }
            }]
        });
        let calls = parse_tool_calls(&resp).unwrap();
        assert_eq!(calls[0].name, "lookup");
        assert_eq!(calls[0].args, json!({"id": 42}));
    }
}
