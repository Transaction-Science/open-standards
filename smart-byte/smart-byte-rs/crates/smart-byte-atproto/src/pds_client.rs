//! XRPC client for talking to a Personal Data Server (PDS).
//!
//! The PDS exposes [XRPC](https://atproto.com/specs/xrpc) endpoints at
//! `https://<host>/xrpc/<method>` with query parameters for queries and
//! JSON bodies for procedures. This client is *thin*: it wires
//! [`XrpcRequest`] / [`XrpcResponse`] from [`crate::lexicon`] through
//! `reqwest` and translates HTTP into [`AtprotoError`].
//!
//! Authentication: the client carries an optional bearer token (the
//! PDS `accessJwt`). It is the caller's responsibility to obtain it via
//! `com.atproto.server.createSession`.

use reqwest::Client;
use serde_json::Value;

use crate::error::AtprotoError;
use crate::lexicon::{XrpcKind, XrpcRequest, XrpcResponse};

/// Default PDS host for the public Bluesky network.
pub const DEFAULT_PDS: &str = "https://bsky.social";

/// Thin XRPC client.
pub struct PdsClient {
    client: Client,
    /// Origin (`https://host[:port]`). No trailing slash.
    origin: String,
    /// Optional bearer token.
    token: Option<String>,
}

impl PdsClient {
    /// Construct a client pointing at `origin`. The origin is the host
    /// portion, e.g. `https://bsky.social`.
    pub fn new(origin: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .user_agent("smart-byte-atproto/0.1")
                .build()
                .unwrap_or_else(|_| Client::new()),
            origin: origin.into().trim_end_matches('/').to_string(),
            token: None,
        }
    }

    /// Construct a client with a caller-supplied [`reqwest::Client`].
    pub fn with_client(client: Client, origin: impl Into<String>) -> Self {
        Self {
            client,
            origin: origin.into().trim_end_matches('/').to_string(),
            token: None,
        }
    }

    /// Set the bearer access token for subsequent calls.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Replace the current token (or clear it with `None`).
    pub fn set_token(&mut self, token: Option<String>) {
        self.token = token;
    }

    /// Compute the URL for an XRPC method.
    pub fn url_for(&self, method: &str) -> String {
        format!("{}/xrpc/{method}", self.origin)
    }

    /// Send an XRPC request and decode the JSON response envelope.
    pub async fn call(
        &self,
        req: &XrpcRequest,
    ) -> Result<XrpcResponse, AtprotoError> {
        let url = self.url_for(&req.method);
        let request = match req.kind {
            XrpcKind::Query => {
                let mut rb = self.client.get(&url);
                for (k, v) in &req.params {
                    if let Some(s) = v.as_str() {
                        rb = rb.query(&[(k.as_str(), s)]);
                    } else {
                        rb = rb.query(&[(k.as_str(), v.to_string().as_str())]);
                    }
                }
                rb
            }
            XrpcKind::Procedure => {
                let body = req.input.clone().unwrap_or(Value::Null);
                self.client.post(&url).json(&body)
            }
        };
        let request = if let Some(t) = &self.token {
            request.bearer_auth(t)
        } else {
            request
        };
        let resp = request
            .send()
            .await
            .map_err(|e| AtprotoError::Network(format!("{url}: {e}")))?;
        let status = resp.status().as_u16();
        let bytes = resp.bytes().await.map_err(|e| {
            AtprotoError::Network(format!("body read {url}: {e}"))
        })?;
        let mut envelope = if bytes.is_empty() {
            XrpcResponse {
                status,
                output: None,
                error: None,
                message: None,
            }
        } else {
            let val: Value = serde_json::from_slice(&bytes)?;
            if let Some(obj) = val.as_object() {
                let error =
                    obj.get("error").and_then(|v| v.as_str()).map(String::from);
                let message = obj
                    .get("message")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                if error.is_some() {
                    XrpcResponse {
                        status,
                        output: None,
                        error,
                        message,
                    }
                } else {
                    XrpcResponse {
                        status,
                        output: Some(val),
                        error: None,
                        message: None,
                    }
                }
            } else {
                XrpcResponse {
                    status,
                    output: Some(val),
                    error: None,
                    message: None,
                }
            }
        };
        if envelope.error.is_none() && status >= 400 {
            envelope.error = Some(format!("HTTP{status}"));
        }
        Ok(envelope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_for_method() {
        let c = PdsClient::new("https://bsky.social/");
        assert_eq!(
            c.url_for("com.atproto.server.createSession"),
            "https://bsky.social/xrpc/com.atproto.server.createSession"
        );
    }
}
