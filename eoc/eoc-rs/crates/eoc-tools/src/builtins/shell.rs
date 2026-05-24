//! `ShellTool` — sandboxed subprocess execution.
//!
//! Allowlist-only. The default allowlist is empty, meaning *every* call
//! is denied until the operator explicitly enrols binary names. Path
//! components are stripped from the command before the allowlist check
//! to prevent `/usr/bin/ls` -> `ls` evasion.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::timeout;

use crate::error::{ToolError, ToolResult};
use crate::schema::ToolSchema;
use crate::tool::Tool;

/// Sandboxed shell-exec tool.
pub struct ShellTool {
    /// Whitelisted command basenames (e.g. `"ls"`, `"git"`).
    pub allowed_commands: HashSet<String>,
    /// Working directory for the child process.
    pub working_dir: PathBuf,
    /// Per-call wall-clock timeout.
    pub timeout: Duration,
    schema: ToolSchema,
}

impl ShellTool {
    /// Construct with the given allowlist + working dir + timeout.
    pub fn new(
        allowed_commands: HashSet<String>,
        working_dir: PathBuf,
        timeout_dur: Duration,
    ) -> Self {
        Self {
            allowed_commands,
            working_dir,
            timeout: timeout_dur,
            schema: ToolSchema::new(
                "shell",
                "Execute a shell command. Allowlisted commands only.",
                json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command basename (e.g. ls, git)."},
                        "args": {"type": "array", "items": {"type": "string"}, "description": "Argument list."}
                    },
                    "required": ["command"]
                }),
            ),
        }
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    async fn invoke(&self, args: Value) -> ToolResult<Value> {
        let raw_cmd = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments {
                tool: "shell".into(),
                reason: "`command` (string) is required".into(),
            })?;
        // Strip path components so /usr/bin/ls is checked as "ls".
        let basename = std::path::Path::new(raw_cmd)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(raw_cmd);
        if !self.allowed_commands.contains(basename) {
            return Err(ToolError::SandboxDenied {
                tool: "shell".into(),
                reason: format!("command `{basename}` is not on the allowlist"),
            });
        }

        let arg_list: Vec<String> = args
            .get("args")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let mut cmd = Command::new(basename);
        cmd.args(&arg_list).current_dir(&self.working_dir);

        let out = timeout(self.timeout, cmd.output())
            .await
            .map_err(|_| ToolError::Timeout("shell".into()))??;

        Ok(json!({
            "status": out.status.code(),
            "stdout": String::from_utf8_lossy(&out.stdout).into_owned(),
            "stderr": String::from_utf8_lossy(&out.stderr).into_owned(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn denies_command_not_on_allowlist() {
        let tool = ShellTool::new(
            HashSet::new(),
            std::env::temp_dir(),
            Duration::from_secs(1),
        );
        let err = tool
            .invoke(json!({"command": "ls", "args": ["-la"]}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::SandboxDenied { .. }));
    }

    #[tokio::test]
    async fn denies_path_prefixed_command_not_on_allowlist() {
        // /usr/bin/ls should be checked as "ls" against the (empty)
        // allowlist and therefore denied.
        let tool = ShellTool::new(
            HashSet::new(),
            std::env::temp_dir(),
            Duration::from_secs(1),
        );
        let err = tool
            .invoke(json!({"command": "/usr/bin/ls"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::SandboxDenied { .. }));
    }
}
