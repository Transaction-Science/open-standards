//! `FileReadTool` / `FileWriteTool` — sandboxed filesystem access.
//!
//! Every call canonicalises the requested path and asserts it sits
//! under one of the allowlisted roots. Symlink escapes are caught by
//! the canonicalisation step.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::error::{ToolError, ToolResult};
use crate::schema::ToolSchema;
use crate::tool::Tool;

fn within_allowed(target: &Path, roots: &[PathBuf]) -> bool {
    let target = match target.canonicalize() {
        Ok(p) => p,
        Err(_) => return false,
    };
    roots.iter().any(|root| {
        let root = match root.canonicalize() {
            Ok(p) => p,
            Err(_) => return false,
        };
        target.starts_with(root)
    })
}

fn within_allowed_for_write(target: &Path, roots: &[PathBuf]) -> bool {
    // For writes the file may not exist yet; check the parent.
    let parent = match target.parent() {
        Some(p) => p,
        None => return false,
    };
    let parent = match parent.canonicalize() {
        Ok(p) => p,
        Err(_) => return false,
    };
    roots.iter().any(|root| {
        let root = match root.canonicalize() {
            Ok(p) => p,
            Err(_) => return false,
        };
        parent.starts_with(root)
    })
}

/// Read a file from an allowlisted root.
pub struct FileReadTool {
    /// Allowlisted root directories.
    pub allowed_paths: Vec<PathBuf>,
    /// Cap on bytes returned to the model.
    pub max_bytes: usize,
    schema: ToolSchema,
}

impl FileReadTool {
    /// Construct.
    pub fn new(allowed_paths: Vec<PathBuf>, max_bytes: usize) -> Self {
        Self {
            allowed_paths,
            max_bytes,
            schema: ToolSchema::new(
                "file_read",
                "Read a file from an allowlisted path.",
                json!({
                    "type": "object",
                    "properties": { "path": {"type": "string"} },
                    "required": ["path"]
                }),
            ),
        }
    }
}

#[async_trait]
impl Tool for FileReadTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }
    async fn invoke(&self, args: Value) -> ToolResult<Value> {
        let path = args.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::InvalidArguments {
                tool: "file_read".into(),
                reason: "`path` required".into(),
            }
        })?;
        let p = Path::new(path);
        if !within_allowed(p, &self.allowed_paths) {
            return Err(ToolError::SandboxDenied {
                tool: "file_read".into(),
                reason: format!("path `{path}` is outside the allowlist"),
            });
        }
        let mut f = tokio::fs::File::open(p).await?;
        let mut buf = Vec::with_capacity(self.max_bytes.min(64 * 1024));
        let mut chunk = vec![0u8; 16 * 1024];
        let mut total = 0usize;
        loop {
            if total >= self.max_bytes {
                break;
            }
            let n = f.read(&mut chunk).await?;
            if n == 0 {
                break;
            }
            let take = n.min(self.max_bytes - total);
            buf.extend_from_slice(&chunk[..take]);
            total += take;
        }
        let truncated = {
            let meta = tokio::fs::metadata(p).await?;
            meta.len() as usize > total
        };
        let body = String::from_utf8_lossy(&buf).into_owned();
        Ok(json!({"content": body, "truncated": truncated}))
    }
}

/// Write a file under an allowlisted root.
pub struct FileWriteTool {
    /// Allowlisted roots.
    pub allowed_paths: Vec<PathBuf>,
    /// Cap on bytes written.
    pub max_bytes: usize,
    schema: ToolSchema,
}

impl FileWriteTool {
    /// Construct.
    pub fn new(allowed_paths: Vec<PathBuf>, max_bytes: usize) -> Self {
        Self {
            allowed_paths,
            max_bytes,
            schema: ToolSchema::new(
                "file_write",
                "Write a file under an allowlisted path.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }),
            ),
        }
    }
}

#[async_trait]
impl Tool for FileWriteTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }
    async fn invoke(&self, args: Value) -> ToolResult<Value> {
        let path = args.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::InvalidArguments {
                tool: "file_write".into(),
                reason: "`path` required".into(),
            }
        })?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments {
                tool: "file_write".into(),
                reason: "`content` required".into(),
            })?;
        if content.len() > self.max_bytes {
            return Err(ToolError::SandboxDenied {
                tool: "file_write".into(),
                reason: format!(
                    "content exceeds max_bytes ({} > {})",
                    content.len(),
                    self.max_bytes
                ),
            });
        }
        let p = Path::new(path);
        if !within_allowed_for_write(p, &self.allowed_paths) {
            return Err(ToolError::SandboxDenied {
                tool: "file_write".into(),
                reason: format!("path `{path}` is outside the allowlist"),
            });
        }
        let mut f = tokio::fs::File::create(p).await?;
        f.write_all(content.as_bytes()).await?;
        f.flush().await?;
        Ok(json!({"bytes_written": content.len()}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn read_outside_allowlist_denied() {
        let tool = FileReadTool::new(vec![std::env::temp_dir()], 1024);
        let err = tool
            .invoke(json!({"path": "/etc/hosts"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::SandboxDenied { .. }));
    }

    #[tokio::test]
    async fn read_inside_allowlist_works() {
        let dir = std::env::temp_dir();
        let f = dir.join("eoc-tools-test-read.txt");
        {
            let mut h = std::fs::File::create(&f).unwrap();
            h.write_all(b"hello").unwrap();
        }
        let tool = FileReadTool::new(vec![dir.clone()], 1024);
        let v = tool
            .invoke(json!({"path": f.to_string_lossy()}))
            .await
            .unwrap();
        assert_eq!(v["content"], "hello");
    }
}
