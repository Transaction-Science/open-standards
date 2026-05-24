//! Lexicon as typed JSON-RPC.
//!
//! AT Protocol's *Lexicon* is the schema language for records and XRPC
//! methods. This module models the subset that matters for federation:
//!
//! * [`LexiconRecord`] — a record body carrying a `$type` discriminator
//!   and arbitrary JSON fields. The known shapes are
//!   [`PostRecord`] (`app.bsky.feed.post`),
//!   [`LikeRecord`] (`app.bsky.feed.like`),
//!   [`RepoCommitRecord`] (`com.atproto.repo.commit`).
//! * [`XrpcRequest`] / [`XrpcResponse`] — the on-the-wire envelope for an
//!   XRPC call (`com.atproto.server.createSession`,
//!   `com.atproto.sync.getRepo`, etc.).
//!
//! The crate deliberately stays generic on fields beyond the well-known
//! ones: AT Protocol Lexicon is open-world and operators routinely
//! extend record types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::AtprotoError;

/// A Lexicon record. The `$type` field carries the schema NSID; the
/// remaining fields are kept as raw JSON for downstream typed access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexiconRecord {
    /// Schema NSID (e.g. `"app.bsky.feed.post"`).
    #[serde(rename = "$type")]
    pub type_: String,
    /// All remaining fields as JSON.
    #[serde(flatten)]
    pub fields: serde_json::Map<String, Value>,
}

impl LexiconRecord {
    /// Construct a record from a `$type` and an arbitrary JSON body.
    pub fn new(
        type_: impl Into<String>,
        body: serde_json::Map<String, Value>,
    ) -> Self {
        Self {
            type_: type_.into(),
            fields: body,
        }
    }

    /// Try to read a top-level field as a string.
    pub fn field_str(&self, key: &str) -> Option<&str> {
        self.fields.get(key).and_then(|v| v.as_str())
    }

    /// Validate that the record's `$type` matches `expected`.
    pub fn require_type(&self, expected: &str) -> Result<(), AtprotoError> {
        if self.type_ == expected {
            Ok(())
        } else {
            Err(AtprotoError::Lexicon(format!(
                "expected $type {expected}, got {}",
                self.type_
            )))
        }
    }
}

/// `app.bsky.feed.post` — a Bluesky post.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostRecord {
    /// Text body of the post.
    pub text: String,
    /// ISO-8601 creation timestamp.
    #[serde(rename = "createdAt")]
    pub created_at: String,
    /// Optional language hint (`["en"]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub langs: Option<Vec<String>>,
}

/// `app.bsky.feed.like` — a Bluesky like.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LikeRecord {
    /// AT URI of the subject being liked.
    pub subject: SubjectRef,
    /// ISO-8601 creation timestamp.
    #[serde(rename = "createdAt")]
    pub created_at: String,
}

/// A reference to another record by URI + CID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectRef {
    /// AT URI of the subject (`at://<did>/<collection>/<rkey>`).
    pub uri: String,
    /// CID of the subject record's IPLD block.
    pub cid: String,
}

/// `com.atproto.repo.commit` — an unsigned repo commit shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoCommitRecord {
    /// Repo DID.
    pub did: String,
    /// Commit format version (currently `3`).
    pub version: u32,
    /// CID of the MST root.
    pub data: String,
    /// CID of the previous commit, or `None` for genesis.
    pub prev: Option<String>,
}

/// An XRPC request envelope.
///
/// XRPC distinguishes *queries* (HTTP GET, idempotent) from *procedures*
/// (HTTP POST, with side effects). Both forms carry an NSID method name
/// and a parameter / body map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XrpcRequest {
    /// XRPC kind: query or procedure.
    pub kind: XrpcKind,
    /// Method NSID, e.g. `"com.atproto.server.createSession"`.
    pub method: String,
    /// Query string parameters (queries).
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub params: serde_json::Map<String, Value>,
    /// Request body (procedures).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
}

/// XRPC method kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum XrpcKind {
    /// Read-only query (HTTP GET).
    Query,
    /// State-changing procedure (HTTP POST).
    Procedure,
}

/// An XRPC response envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XrpcResponse {
    /// HTTP status the server reported.
    pub status: u16,
    /// Output JSON, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    /// Structured XRPC error code, if the server reported one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Human-readable error message, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl XrpcResponse {
    /// Convert into a `Result`, mapping non-`None` `error` into
    /// [`AtprotoError::Xrpc`].
    pub fn into_result(self) -> Result<Option<Value>, AtprotoError> {
        if let Some(code) = self.error {
            return Err(AtprotoError::Xrpc {
                code,
                message: self.message.unwrap_or_default(),
            });
        }
        Ok(self.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn post_record_roundtrip() {
        let post = PostRecord {
            text: "hello atproto".into(),
            created_at: "2026-05-24T00:00:00Z".into(),
            langs: Some(vec!["en".into()]),
        };
        let s = serde_json::to_string(&post).unwrap();
        let back: PostRecord = serde_json::from_str(&s).unwrap();
        assert_eq!(back.text, "hello atproto");
    }

    #[test]
    fn lexicon_record_dollar_type() {
        let r = LexiconRecord::new(
            "app.bsky.feed.post",
            json!({"text": "hi"}).as_object().unwrap().clone(),
        );
        assert!(r.require_type("app.bsky.feed.post").is_ok());
        assert!(r.require_type("app.bsky.feed.like").is_err());
        assert_eq!(r.field_str("text"), Some("hi"));
    }

    #[test]
    fn xrpc_error_into_result() {
        let resp = XrpcResponse {
            status: 400,
            output: None,
            error: Some("InvalidRequest".into()),
            message: Some("bad".into()),
        };
        assert!(matches!(
            resp.into_result(),
            Err(AtprotoError::Xrpc { .. })
        ));
    }
}
