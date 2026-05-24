//! `SqlTool` — guarded SQL execution.
//!
//! This crate doesn't ship a database driver — `SqlTool` carries the
//! connection string and a *driver callback* the operator supplies.
//! That keeps the dependency footprint small and lets each deployment
//! plug in whichever driver (sqlx, tokio-postgres, libsql, duckdb) it
//! already uses.
//!
//! Write protection: `allow_writes = false` causes `INSERT`, `UPDATE`,
//! `DELETE`, `CREATE`, `DROP`, `ALTER`, `TRUNCATE`, `GRANT`, `REVOKE`,
//! and `REPLACE` statements to be rejected before the driver runs.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::error::{ToolError, ToolResult};
use crate::schema::ToolSchema;
use crate::tool::Tool;

/// Driver callback signature.
pub type SqlDriver = Arc<
    dyn Fn(
            String,
            String,
        ) -> Pin<Box<dyn std::future::Future<Output = ToolResult<Vec<Value>>> + Send>>
        + Send
        + Sync,
>;

/// Guarded SQL tool.
pub struct SqlTool {
    /// Driver-specific connection string (DSN).
    pub connection_string: String,
    /// If false, write statements are refused before reaching the driver.
    pub allow_writes: bool,
    /// Row-count ceiling — returned rows are truncated.
    pub max_rows: usize,
    schema: ToolSchema,
    driver: SqlDriver,
}

impl SqlTool {
    /// Construct.
    pub fn new(
        connection_string: impl Into<String>,
        allow_writes: bool,
        max_rows: usize,
        driver: SqlDriver,
    ) -> Self {
        Self {
            connection_string: connection_string.into(),
            allow_writes,
            max_rows,
            driver,
            schema: ToolSchema::new(
                "sql",
                "Run a SQL query. Read-only unless allow_writes is set.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }),
            ),
        }
    }
}

fn looks_like_write(sql: &str) -> bool {
    let head = sql.trim_start().to_ascii_uppercase();
    const WRITE_VERBS: &[&str] = &[
        "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "TRUNCATE", "GRANT", "REVOKE",
        "REPLACE", "MERGE",
    ];
    WRITE_VERBS.iter().any(|v| head.starts_with(v))
}

#[async_trait]
impl Tool for SqlTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    async fn invoke(&self, args: Value) -> ToolResult<Value> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments {
                tool: "sql".into(),
                reason: "`query` (string) required".into(),
            })?
            .to_string();
        if !self.allow_writes && looks_like_write(&query) {
            return Err(ToolError::SandboxDenied {
                tool: "sql".into(),
                reason: "write statement rejected: allow_writes=false".into(),
            });
        }
        let rows = (self.driver)(self.connection_string.clone(), query).await?;
        let truncated = rows.len() > self.max_rows;
        let final_rows: Vec<Value> = rows.into_iter().take(self.max_rows).collect();
        Ok(json!({
            "rows": final_rows,
            "row_count": final_rows.len(),
            "truncated": truncated
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_driver() -> SqlDriver {
        Arc::new(|_dsn, q| {
            Box::pin(async move { Ok(vec![json!({"query": q})]) })
        })
    }

    #[tokio::test]
    async fn rejects_writes_when_read_only() {
        let tool = SqlTool::new("dsn", false, 100, echo_driver());
        let err = tool
            .invoke(json!({"query": "INSERT INTO t VALUES (1)"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::SandboxDenied { .. }));
    }

    #[tokio::test]
    async fn allows_writes_when_opted_in() {
        let tool = SqlTool::new("dsn", true, 100, echo_driver());
        let v = tool
            .invoke(json!({"query": "INSERT INTO t VALUES (1)"}))
            .await
            .unwrap();
        assert_eq!(v["row_count"], 1);
    }

    #[tokio::test]
    async fn allows_select_when_read_only() {
        let tool = SqlTool::new("dsn", false, 100, echo_driver());
        let v = tool
            .invoke(json!({"query": "SELECT 1"}))
            .await
            .unwrap();
        assert_eq!(v["row_count"], 1);
    }
}
