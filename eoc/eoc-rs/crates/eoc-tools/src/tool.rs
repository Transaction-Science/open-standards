//! The [`Tool`] trait, the canonical [`ToolCallRequest`] form, and
//! the [`ToolRegistry`] dispatcher.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{ToolError, ToolResult};
use crate::schema::ToolSchema;

/// An invocable tool. Implementations are typically thin wrappers over
/// a sandboxed primitive (shell, http, file …) or a domain function.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The canonical schema describing this tool.
    fn schema(&self) -> &ToolSchema;

    /// Invoke the tool with the supplied JSON-shaped arguments. The
    /// returned value is opaque JSON that will be fed back to the
    /// model as the tool result.
    async fn invoke(&self, args: Value) -> ToolResult<Value>;
}

/// A canonical tool-call request — vendor-agnostic. Each vendor adapter
/// extracts these from the provider's response shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallRequest {
    /// Vendor-supplied call id (used for correlating result back into
    /// the model's context). Empty string when the vendor does not
    /// surface a stable id.
    pub id: String,
    /// Tool name — must match a registered [`Tool::schema().name`].
    pub name: String,
    /// JSON arguments object.
    pub args: Value,
}

/// A registry of named tools.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool. The registry indexes by `tool.schema().name`.
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        let name = tool.schema().name.clone();
        self.tools.insert(name, Box::new(tool));
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Lookup by name.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|b| b.as_ref())
    }

    /// All registered schemas — used by vendor adapters to emit the
    /// `tools` array on outgoing requests.
    pub fn schemas(&self) -> Vec<&ToolSchema> {
        self.tools.values().map(|t| t.schema()).collect()
    }

    /// Dispatch a call by name.
    pub async fn dispatch(&self, name: &str, args: Value) -> ToolResult<Value> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::NotRegistered(name.to_string()))?;
        tool.invoke(args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ToolSchema;
    use serde_json::json;

    struct EchoTool {
        schema: ToolSchema,
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn schema(&self) -> &ToolSchema {
            &self.schema
        }
        async fn invoke(&self, args: Value) -> ToolResult<Value> {
            Ok(args)
        }
    }

    #[tokio::test]
    async fn dispatch_runs_registered_tool() {
        let mut reg = ToolRegistry::new();
        reg.register(EchoTool {
            schema: ToolSchema::new("echo", "echoes", json!({"type": "object"})),
        });
        let out = reg.dispatch("echo", json!({"a": 1})).await.unwrap();
        assert_eq!(out, json!({"a": 1}));
    }

    #[tokio::test]
    async fn dispatch_unknown_tool_errors() {
        let reg = ToolRegistry::new();
        let err = reg.dispatch("nope", json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::NotRegistered(_)));
    }
}
