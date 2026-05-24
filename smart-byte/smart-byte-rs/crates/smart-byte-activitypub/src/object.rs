//! ActivityStreams 2.0 Objects.
//!
//! Objects are the addressable content nouns in ActivityPub. The most
//! common one on the deployed federation is a Mastodon-flavoured
//! `Note`, but the same shape carries `Article`, `Image`, and `Video`.
//!
//! Mastodon extensions modelled here:
//!
//! * `sensitive` — content-warning gate for media + text
//! * `summary` — used by Mastodon as the CW string
//! * `conversation` — toot-specific conversation pointer
//! * `blurhash` — perceptual hash for images / videos

use crate::error::{ActivityPubError, Result};
use crate::vocabulary::{ObjectKind, AS2_CONTEXT, PUBLIC};
use serde::{Deserialize, Serialize};

/// An attachment on an Object — typically an image or video on a Note.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Attachment {
    /// AS2 type, e.g. `"Document"`, `"Image"`, `"Video"`.
    #[serde(rename = "type")]
    pub type_field: String,
    /// IRI of the media.
    pub url: String,
    /// Media MIME type.
    #[serde(rename = "mediaType", default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Alt text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Mastodon blurhash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blurhash: Option<String>,
}

/// An AS2 Object — Notes, Articles, Images, Videos, and Tombstones all
/// share this shape. Object-kind-specific fields are merged into one
/// struct because that matches how the wire JSON arrives.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Object {
    /// JSON-LD context. Only required at the top level of a delivered
    /// document, but we always carry it for round-trip stability.
    #[serde(rename = "@context", default, skip_serializing_if = "Option::is_none")]
    pub context: Option<serde_json::Value>,
    /// Object IRI.
    pub id: String,
    /// AS2 `type`.
    #[serde(rename = "type")]
    pub type_field: String,
    /// IRI of the actor that authored / owns the object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attributed_to: Option<String>,
    /// Plain-text or HTML content (Mastodon Notes use HTML here).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Short summary; Mastodon uses this as the content warning string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Plain title for Articles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// `to` recipients.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub to: Vec<String>,
    /// `cc` recipients.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cc: Vec<String>,
    /// `bcc` recipients (stripped on delivery).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bcc: Vec<String>,
    /// In-reply-to IRI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_reply_to: Option<String>,
    /// Publication timestamp (RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published: Option<String>,
    /// Update timestamp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated: Option<String>,
    /// Attachments — images / videos / docs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachment: Vec<Attachment>,
    /// Mastodon `sensitive` flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensitive: Option<bool>,
    /// Mastodon `conversation` pointer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,
    /// IRI for the object's replies collection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replies: Option<String>,
    /// MIME type — used by Article and friends.
    #[serde(rename = "mediaType", default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// IRI of the binary, used for Image / Video.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Blurhash for Image / Video.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blurhash: Option<String>,
    /// Tombstone — previous type before deletion.
    #[serde(rename = "formerType", default, skip_serializing_if = "Option::is_none")]
    pub former_type: Option<String>,
    /// Tombstone — deletion timestamp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deleted: Option<String>,
}

impl Object {
    /// Internal helper — fresh empty object with the given id + type.
    fn empty(id: impl Into<String>, kind: ObjectKind) -> Self {
        Self {
            context: None,
            id: id.into(),
            type_field: kind.as_type().to_string(),
            attributed_to: None,
            content: None,
            summary: None,
            name: None,
            to: Vec::new(),
            cc: Vec::new(),
            bcc: Vec::new(),
            in_reply_to: None,
            published: None,
            updated: None,
            attachment: Vec::new(),
            sensitive: None,
            conversation: None,
            replies: None,
            media_type: None,
            url: None,
            blurhash: None,
            former_type: None,
            deleted: None,
        }
    }

    /// Construct a Mastodon-shaped public Note.
    pub fn note(
        id: impl Into<String>,
        attributed_to: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        let mut obj = Self::empty(id, ObjectKind::Note);
        obj.attributed_to = Some(attributed_to.into());
        obj.content = Some(content.into());
        obj.to = vec![PUBLIC.to_string()];
        obj
    }

    /// Construct an Article.
    pub fn article(
        id: impl Into<String>,
        attributed_to: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        let mut obj = Self::empty(id, ObjectKind::Article);
        obj.attributed_to = Some(attributed_to.into());
        obj.name = Some(name.into());
        obj.content = Some(content.into());
        obj
    }

    /// Construct an Image standalone object.
    pub fn image(
        id: impl Into<String>,
        attributed_to: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        let mut obj = Self::empty(id, ObjectKind::Image);
        obj.attributed_to = Some(attributed_to.into());
        obj.url = Some(url.into());
        obj
    }

    /// Construct a Video standalone object.
    pub fn video(
        id: impl Into<String>,
        attributed_to: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        let mut obj = Self::empty(id, ObjectKind::Video);
        obj.attributed_to = Some(attributed_to.into());
        obj.url = Some(url.into());
        obj
    }

    /// Construct a Tombstone for an Object that has been Deleted.
    pub fn tombstone(id: impl Into<String>, former_type: impl Into<String>) -> Self {
        let mut obj = Self::empty(id, ObjectKind::Tombstone);
        obj.former_type = Some(former_type.into());
        obj
    }

    /// Attach the canonical AS2 context — used when an object is
    /// itself the top-level document (e.g. served from its own IRI).
    pub fn with_context(mut self) -> Self {
        self.context = Some(serde_json::Value::String(AS2_CONTEXT.into()));
        self
    }

    /// Address this object to the public collection.
    pub fn public(mut self) -> Self {
        if !self.to.iter().any(|t| t == PUBLIC) {
            self.to.push(PUBLIC.to_string());
        }
        self
    }

    /// Add a `cc` recipient.
    pub fn cc(mut self, iri: impl Into<String>) -> Self {
        self.cc.push(iri.into());
        self
    }

    /// Mark as Mastodon-`sensitive` with a CW summary.
    pub fn with_sensitive(mut self, summary: impl Into<String>) -> Self {
        self.sensitive = Some(true);
        self.summary = Some(summary.into());
        self
    }

    /// Return the typed [`ObjectKind`] if recognised.
    pub fn kind(&self) -> Option<ObjectKind> {
        ObjectKind::parse(&self.type_field)
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self> {
        let obj: Object = serde_json::from_str(s)?;
        if obj.id.is_empty() {
            return Err(ActivityPubError::Vocabulary("object missing id".into()));
        }
        if ObjectKind::parse(&obj.type_field).is_none() {
            return Err(ActivityPubError::Vocabulary(format!(
                "unknown object type {}",
                obj.type_field
            )));
        }
        Ok(obj)
    }

    /// True when this object is addressed to the public collection.
    pub fn is_public(&self) -> bool {
        self.to.iter().chain(self.cc.iter()).any(|t| t == PUBLIC)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_is_public_by_default() {
        let n = Object::note(
            "https://example.test/notes/1",
            "https://example.test/users/alice",
            "hello world",
        );
        assert!(n.is_public());
        assert_eq!(n.kind(), Some(ObjectKind::Note));
    }

    #[test]
    fn json_roundtrip_note() -> Result<()> {
        let n = Object::note(
            "https://example.test/notes/1",
            "https://example.test/users/alice",
            "hello",
        )
        .cc("https://example.test/users/alice/followers")
        .with_sensitive("CW: greetings");
        let json = n.to_json()?;
        let m = Object::from_json(&json)?;
        assert_eq!(n, m);
        Ok(())
    }

    #[test]
    fn tombstone_carries_former_type() {
        let t = Object::tombstone("https://example.test/notes/1", "Note");
        assert_eq!(t.kind(), Some(ObjectKind::Tombstone));
        assert_eq!(t.former_type.as_deref(), Some("Note"));
    }
}
