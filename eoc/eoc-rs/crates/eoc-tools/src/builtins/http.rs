//! `HttpTool` — sandboxed HTTP requests.
//!
//! Allowlist on hosts (matched against the URL's host component). The
//! body is bounded to `max_response_bytes` to keep the model context
//! small.

use std::collections::HashSet;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::error::{ToolError, ToolResult};
use crate::schema::ToolSchema;
use crate::tool::Tool;

/// HTTP method subset.
#[derive(Debug, Clone, Copy)]
pub enum HttpMethod {
    /// GET.
    Get,
    /// POST.
    Post,
    /// PUT.
    Put,
    /// DELETE.
    Delete,
}

impl HttpMethod {
    fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "GET" => Some(Self::Get),
            "POST" => Some(Self::Post),
            "PUT" => Some(Self::Put),
            "DELETE" => Some(Self::Delete),
            _ => None,
        }
    }
}

/// Sandboxed HTTP fetch tool.
pub struct HttpTool {
    /// Allowlisted hosts (e.g. `"api.example.com"`).
    pub allowed_hosts: HashSet<String>,
    /// Body cap.
    pub max_response_bytes: usize,
    /// Per-call wall-clock timeout.
    pub timeout: Duration,
    schema: ToolSchema,
    client: reqwest::Client,
}

impl HttpTool {
    /// Construct.
    pub fn new(
        allowed_hosts: HashSet<String>,
        max_response_bytes: usize,
        timeout_dur: Duration,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout_dur)
            .build()
            .unwrap_or_default();
        Self {
            allowed_hosts,
            max_response_bytes,
            timeout: timeout_dur,
            client,
            schema: ToolSchema::new(
                "http",
                "Make an HTTP request to an allowlisted host.",
                json!({
                    "type": "object",
                    "properties": {
                        "method": {"type": "string", "enum": ["GET","POST","PUT","DELETE"]},
                        "url": {"type": "string"},
                        "headers": {"type": "object"},
                        "body": {"type": "string"}
                    },
                    "required": ["method", "url"]
                }),
            ),
        }
    }

    fn check_host(&self, url: &str) -> ToolResult<()> {
        let parsed = url::Url::parse(url).map_err(|e| ToolError::InvalidArguments {
            tool: "http".into(),
            reason: format!("invalid url: {e}"),
        })?;
        let host = parsed
            .host_str()
            .ok_or_else(|| ToolError::InvalidArguments {
                tool: "http".into(),
                reason: "url has no host".into(),
            })?;
        if !self.allowed_hosts.contains(host) {
            return Err(ToolError::SandboxDenied {
                tool: "http".into(),
                reason: format!("host `{host}` is not on the allowlist"),
            });
        }
        Ok(())
    }
}

#[async_trait]
impl Tool for HttpTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    async fn invoke(&self, args: Value) -> ToolResult<Value> {
        let method_str = args
            .get("method")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments {
                tool: "http".into(),
                reason: "`method` is required".into(),
            })?;
        let method = HttpMethod::parse(method_str).ok_or_else(|| {
            ToolError::InvalidArguments {
                tool: "http".into(),
                reason: format!("unsupported method `{method_str}`"),
            }
        })?;
        let url = args.get("url").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::InvalidArguments {
                tool: "http".into(),
                reason: "`url` is required".into(),
            }
        })?;
        self.check_host(url)?;

        let mut req = match method {
            HttpMethod::Get => self.client.get(url),
            HttpMethod::Post => self.client.post(url),
            HttpMethod::Put => self.client.put(url),
            HttpMethod::Delete => self.client.delete(url),
        };
        if let Some(headers) = args.get("headers").and_then(|v| v.as_object()) {
            for (k, v) in headers {
                if let Some(s) = v.as_str() {
                    req = req.header(k.as_str(), s);
                }
            }
        }
        if let Some(body) = args.get("body").and_then(|v| v.as_str()) {
            req = req.body(body.to_string());
        }

        let resp = req.send().await?;
        let status = resp.status().as_u16();
        let body_bytes = resp.bytes().await?;
        let truncated_len = body_bytes.len().min(self.max_response_bytes);
        let body = String::from_utf8_lossy(&body_bytes[..truncated_len]).into_owned();

        Ok(json!({
            "status": status,
            "body": body,
            "truncated": body_bytes.len() > self.max_response_bytes
        }))
    }
}

// Tiny URL parser shim so we don't pull in the `url` crate as a hard
// dep. We accept only http/https schemes.
mod url {
    use super::{ToolError, ToolResult};

    pub struct Url {
        host: Option<String>,
    }
    impl Url {
        pub fn parse(s: &str) -> ToolResult<Self> {
            let rest = s
                .strip_prefix("https://")
                .or_else(|| s.strip_prefix("http://"))
                .ok_or_else(|| ToolError::InvalidArguments {
                    tool: "http".into(),
                    reason: "url must start with http:// or https://".into(),
                })?;
            // host is up to next '/' or '?' or end, optionally stripping
            // userinfo and port.
            let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
            let authority = &rest[..end];
            let authority = authority.rsplit_once('@').map(|x| x.1).unwrap_or(authority);
            let host = authority.split_once(':').map(|x| x.0).unwrap_or(authority);
            if host.is_empty() {
                return Err(ToolError::InvalidArguments {
                    tool: "http".into(),
                    reason: "url has no host".into(),
                });
            }
            Ok(Self {
                host: Some(host.to_string()),
            })
        }
        pub fn host_str(&self) -> Option<&str> {
            self.host.as_deref()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn denies_host_not_on_allowlist() {
        let tool = HttpTool::new(HashSet::new(), 64_000, Duration::from_secs(2));
        let err = tool
            .invoke(json!({"method": "GET", "url": "https://example.com/"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::SandboxDenied { .. }));
    }

    #[tokio::test]
    async fn rejects_unsupported_scheme() {
        let tool = HttpTool::new(HashSet::new(), 64_000, Duration::from_secs(2));
        let err = tool
            .invoke(json!({"method": "GET", "url": "ftp://example.com/"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments { .. }));
    }
}
