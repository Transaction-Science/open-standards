//! Per-vendor round-trip: define a Rust tool type → generate canonical
//! [`ToolSchema`] → translate to vendor format → parse a vendor response
//! shaped against that tool → execute → result returned.

use async_trait::async_trait;
use eoc_tools::{
    Tool, ToolRegistry, ToolResult, ToolSchema,
    anthropic_schema, cohere_schema, google_schema, mistral_schema, openai_schema, schema_for,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Serialize, Deserialize, JsonSchema)]
struct AddArgs {
    a: i64,
    b: i64,
}

struct AddTool {
    schema: ToolSchema,
}

#[async_trait]
impl Tool for AddTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }
    async fn invoke(&self, args: Value) -> ToolResult<Value> {
        let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
        let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
        Ok(json!({"sum": a + b}))
    }
}

fn make_registry() -> ToolRegistry {
    let schema = schema_for::<AddArgs>("add", "Sum two integers.");
    let mut reg = ToolRegistry::new();
    reg.register(AddTool { schema });
    reg
}

#[tokio::test]
async fn anthropic_round_trip() {
    let reg = make_registry();
    let schema = reg.schemas()[0];
    let vendor_tool = anthropic_schema::to_anthropic_tool(schema);
    assert_eq!(vendor_tool["name"], "add");

    // Synthetic vendor response.
    let resp = json!({
        "content": [
            {"type": "text", "text": "let me compute"},
            {"type": "tool_use", "id": "toolu_1", "name": "add",
             "input": {"a": 7, "b": 35}}
        ]
    });
    let calls = anthropic_schema::parse_tool_calls(&resp).unwrap();
    assert_eq!(calls.len(), 1);
    let out = reg.dispatch(&calls[0].name, calls[0].args.clone()).await.unwrap();
    assert_eq!(out, json!({"sum": 42}));
}

#[tokio::test]
async fn openai_round_trip() {
    let reg = make_registry();
    let schema = reg.schemas()[0];
    let vendor_tool = openai_schema::to_openai_tool(schema);
    assert_eq!(vendor_tool["function"]["name"], "add");

    let resp = json!({
        "choices": [{
            "message": {
                "tool_calls": [{
                    "id": "call_42",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{\"a\":7,\"b\":35}"}
                }]
            }
        }]
    });
    let calls = openai_schema::parse_tool_calls(&resp).unwrap();
    let out = reg.dispatch(&calls[0].name, calls[0].args.clone()).await.unwrap();
    assert_eq!(out, json!({"sum": 42}));
}

#[tokio::test]
async fn google_round_trip() {
    let reg = make_registry();
    let schema = reg.schemas()[0];
    let vendor_tools = google_schema::tools_array(&[schema]);
    assert_eq!(vendor_tools[0]["functionDeclarations"][0]["name"], "add");

    let resp = json!({
        "candidates": [{
            "content": {
                "parts": [
                    {"functionCall": {"name": "add", "args": {"a": 7, "b": 35}}}
                ]
            }
        }]
    });
    let calls = google_schema::parse_tool_calls(&resp).unwrap();
    let out = reg.dispatch(&calls[0].name, calls[0].args.clone()).await.unwrap();
    assert_eq!(out, json!({"sum": 42}));
}

#[tokio::test]
async fn mistral_round_trip() {
    let reg = make_registry();
    let schema = reg.schemas()[0];
    let vendor_tool = mistral_schema::to_mistral_tool(schema);
    assert_eq!(vendor_tool["function"]["name"], "add");

    let resp = json!({
        "choices": [{
            "message": {
                "tool_calls": [{
                    "id": "call_m",
                    "function": {"name": "add", "arguments": "{\"a\":7,\"b\":35}"}
                }]
            }
        }]
    });
    let calls = mistral_schema::parse_tool_calls(&resp).unwrap();
    let out = reg.dispatch(&calls[0].name, calls[0].args.clone()).await.unwrap();
    assert_eq!(out, json!({"sum": 42}));
}

#[tokio::test]
async fn cohere_round_trip() {
    let reg = make_registry();
    let schema = reg.schemas()[0];
    let vendor_tool = cohere_schema::to_cohere_tool(schema);
    assert_eq!(vendor_tool["name"], "add");

    let resp = json!({
        "tool_calls": [
            {"name": "add", "parameters": {"a": 7, "b": 35}}
        ]
    });
    let calls = cohere_schema::parse_tool_calls(&resp).unwrap();
    let out = reg.dispatch(&calls[0].name, calls[0].args.clone()).await.unwrap();
    assert_eq!(out, json!({"sum": 42}));
}
