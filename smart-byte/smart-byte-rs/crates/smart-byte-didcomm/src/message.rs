//! DIDComm v2 plaintext message envelope (DIF DIDComm Messaging v2.1 § 3).
//!
//! A DIDComm message is a JSON object with a fixed set of envelope
//! attributes and a `body` of arbitrary protocol-specific JSON. Optional
//! `attachments` carry binary payloads outside the body.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use smart_byte_did::Did;

/// A DIDComm v2 message in its plaintext form.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DidcommMessage {
    /// Globally unique message id (DIDComm v2 § 3.1).
    pub id: String,
    /// Message type URI, of the form
    /// `<protocol-uri>/<version>/<message-name>` (DIDComm v2 § 3.1).
    #[serde(rename = "type")]
    pub type_: String,
    /// Sender DID. Optional for anoncrypt and plaintext.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub from: Option<Did>,
    /// Recipient DIDs (DIDComm v2 § 3.2 — `to` is an array).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub to: Vec<Did>,
    /// Thread id (`thid`) — id of the first message in the thread.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub thid: Option<String>,
    /// Parent thread id (`pthid`) — for nested protocols.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub pthid: Option<String>,
    /// Creation timestamp.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub created_time: Option<DateTime<Utc>>,
    /// Expiry timestamp. Recipients SHOULD discard expired messages.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub expires_time: Option<DateTime<Utc>>,
    /// Application body — protocol-specific JSON.
    #[serde(default)]
    pub body: serde_json::Value,
    /// Out-of-line binary attachments (DIDComm v2 § 5).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<Attachment>,
}

impl DidcommMessage {
    /// Build a new message with a freshly-minted UUID id.
    pub fn new(type_: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            type_: type_.into(),
            from: None,
            to: vec![],
            thid: None,
            pthid: None,
            created_time: Some(Utc::now()),
            expires_time: None,
            body: serde_json::Value::Object(Default::default()),
            attachments: vec![],
        }
    }

    /// Set `from` and return the message (builder style).
    pub fn from_did(mut self, from: Did) -> Self {
        self.from = Some(from);
        self
    }

    /// Set `to` and return the message (builder style).
    pub fn to_dids(mut self, to: Vec<Did>) -> Self {
        self.to = to;
        self
    }

    /// Set `thid` and return the message (builder style).
    pub fn thread(mut self, thid: impl Into<String>) -> Self {
        self.thid = Some(thid.into());
        self
    }

    /// Set `body` and return the message (builder style).
    pub fn body(mut self, body: serde_json::Value) -> Self {
        self.body = body;
        self
    }

    /// `true` if `expires_time` is set and is in the past.
    pub fn is_expired(&self) -> bool {
        matches!(self.expires_time, Some(t) if t < Utc::now())
    }
}

/// A DIDComm v2 attachment (§ 5).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Attachment {
    /// Optional attachment id (referenced from the body by `@id`).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub id: Option<String>,
    /// Human-friendly description.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub description: Option<String>,
    /// Filename.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub filename: Option<String>,
    /// MIME type.
    #[serde(rename = "media_type", skip_serializing_if = "Option::is_none", default)]
    pub media_type: Option<String>,
    /// Attachment format identifier (e.g. `dif/credential-manifest@v1.0`).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub format: Option<String>,
    /// Last-modified timestamp.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub lastmod_time: Option<DateTime<Utc>>,
    /// Byte count.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub byte_count: Option<u64>,
    /// Attachment data — base64, JSON, or links.
    pub data: AttachmentData,
}

/// Attachment data variants (DIDComm v2 § 5.2).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AttachmentData {
    /// Inline base64 data.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub base64: Option<String>,
    /// Inline JSON data.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub json: Option<serde_json::Value>,
    /// External links the recipient can fetch.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<String>,
    /// SHA-256 hash (multihash, base64url) for link integrity.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub hash: Option<String>,
}

impl AttachmentData {
    /// Wrap an inline JSON payload.
    pub fn from_json(v: serde_json::Value) -> Self {
        Self {
            base64: None,
            json: Some(v),
            links: vec![],
            hash: None,
        }
    }

    /// Wrap an inline base64 payload.
    pub fn from_base64(b64: impl Into<String>) -> Self {
        Self {
            base64: Some(b64.into()),
            json: None,
            links: vec![],
            hash: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_message_round_trip() {
        let alice: Did = "did:example:alice".parse().unwrap();
        let bob: Did = "did:example:bob".parse().unwrap();
        let m = DidcommMessage::new("https://didcomm.org/basicmessage/2.0/message")
            .from_did(alice.clone())
            .to_dids(vec![bob.clone()])
            .body(serde_json::json!({"content": "hello"}));
        let j = serde_json::to_string(&m).unwrap();
        let back: DidcommMessage = serde_json::from_str(&j).unwrap();
        assert_eq!(back.from.unwrap(), alice);
        assert_eq!(back.to, vec![bob]);
        assert_eq!(back.body["content"], "hello");
    }
}
