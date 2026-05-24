//! ActivityPub Actors (ActivityPub §4.1).
//!
//! An Actor is the addressable identity in ActivityPub. Every server
//! that participates in federation publishes an Actor document at a
//! stable IRI. The document carries the actor's inbox / outbox endpoints,
//! the public key used to verify HTTP Signatures, and a handful of
//! Mastodon-flavoured extensions (`featured`, `discoverable`).
//!
//! This module models the document as a strongly-typed struct so the
//! rest of the crate does not have to grep through JSON.

use crate::error::{ActivityPubError, Result};
use crate::vocabulary::{ActorKind, AS2_CONTEXT, SECURITY_CONTEXT};
use serde::{Deserialize, Serialize};

/// Public key block on an Actor document.
///
/// The PEM is whatever the actor publishes; for Mastodon-style
/// federation it is an RSA SPKI key, but we keep it as an opaque string
/// so the same shape works for Ed25519 and others.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PublicKey {
    /// Unique key id, typically the actor IRI followed by `#main-key`.
    pub id: String,
    /// IRI of the actor that owns this key.
    pub owner: String,
    /// PEM-encoded public key.
    #[serde(rename = "publicKeyPem")]
    pub public_key_pem: String,
}

/// Endpoint collection — Mastodon publishes a `sharedInbox` here.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Endpoints {
    /// Optional shared inbox for fan-out delivery.
    #[serde(rename = "sharedInbox", skip_serializing_if = "Option::is_none")]
    pub shared_inbox: Option<String>,
}

/// An ActivityPub Actor document.
///
/// Field order follows the JSON-LD `@context` + `id` + `type` convention.
/// `Option` fields are skipped when serialising so the output matches
/// the canonical Mastodon-flavoured shape that other servers expect.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Actor {
    /// JSON-LD context. We always emit both AS2 and the security vocab.
    #[serde(rename = "@context")]
    pub context: serde_json::Value,
    /// Stable IRI for this actor.
    pub id: String,
    /// AS2 `type` — one of Person / Application / Group / Organization / Service.
    #[serde(rename = "type")]
    pub type_field: String,
    /// `preferredUsername` — the local part of the acct (Webfinger).
    #[serde(rename = "preferredUsername")]
    pub preferred_username: String,
    /// Human-readable display name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Profile summary / bio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Inbox endpoint (POST target for incoming activities).
    pub inbox: String,
    /// Outbox endpoint.
    pub outbox: String,
    /// Followers collection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub followers: Option<String>,
    /// Following collection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub following: Option<String>,
    /// Mastodon "featured" collection — pinned posts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub featured: Option<String>,
    /// Endpoints object — typically contains `sharedInbox`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoints: Option<Endpoints>,
    /// Public key used for HTTP Signature verification.
    #[serde(rename = "publicKey", default, skip_serializing_if = "Option::is_none")]
    pub public_key: Option<PublicKey>,
    /// Mastodon `discoverable` flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discoverable: Option<bool>,
    /// Account creation timestamp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published: Option<String>,
}

impl Actor {
    /// Construct a fresh Actor with the canonical context and the
    /// minimum required fields populated.
    pub fn new(
        kind: ActorKind,
        id: impl Into<String>,
        preferred_username: impl Into<String>,
        inbox: impl Into<String>,
        outbox: impl Into<String>,
    ) -> Self {
        let context = serde_json::json!([AS2_CONTEXT, SECURITY_CONTEXT]);
        Self {
            context,
            id: id.into(),
            type_field: kind.as_type().to_string(),
            preferred_username: preferred_username.into(),
            name: None,
            summary: None,
            inbox: inbox.into(),
            outbox: outbox.into(),
            followers: None,
            following: None,
            featured: None,
            endpoints: None,
            public_key: None,
            discoverable: None,
            published: None,
        }
    }

    /// Return the [`ActorKind`] if `type` is one we recognise.
    pub fn kind(&self) -> Option<ActorKind> {
        ActorKind::parse(&self.type_field)
    }

    /// Resolve which inbox to deliver to for this actor — prefer the
    /// shared inbox when one is published (Mastodon convention).
    pub fn delivery_inbox(&self) -> &str {
        match &self.endpoints {
            Some(Endpoints {
                shared_inbox: Some(s),
            }) if !s.is_empty() => s,
            _ => &self.inbox,
        }
    }

    /// Serialise the actor to canonical JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Parse an actor from JSON, requiring at minimum the AS2 context.
    pub fn from_json(s: &str) -> Result<Self> {
        let actor: Actor = serde_json::from_str(s)?;
        actor.validate()?;
        Ok(actor)
    }

    /// Validate structural invariants — required fields and a sane
    /// context. Performed on every parse so downstream consumers can
    /// rely on the parsed shape.
    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty() {
            return Err(ActivityPubError::Vocabulary("actor missing id".into()));
        }
        if self.inbox.is_empty() {
            return Err(ActivityPubError::Vocabulary("actor missing inbox".into()));
        }
        if self.outbox.is_empty() {
            return Err(ActivityPubError::Vocabulary("actor missing outbox".into()));
        }
        if ActorKind::parse(&self.type_field).is_none() {
            return Err(ActivityPubError::Vocabulary(format!(
                "unknown actor type {}",
                self.type_field
            )));
        }
        if !context_contains_as2(&self.context) {
            return Err(ActivityPubError::InvalidContext(
                "missing AS2 context".into(),
            ));
        }
        Ok(())
    }
}

fn context_contains_as2(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(s) => s == AS2_CONTEXT,
        serde_json::Value::Array(arr) => arr.iter().any(context_contains_as2),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Actor {
        let mut a = Actor::new(
            ActorKind::Person,
            "https://example.test/users/alice",
            "alice",
            "https://example.test/users/alice/inbox",
            "https://example.test/users/alice/outbox",
        );
        a.endpoints = Some(Endpoints {
            shared_inbox: Some("https://example.test/inbox".into()),
        });
        a
    }

    #[test]
    fn json_roundtrip() -> Result<()> {
        let a = sample();
        let json = a.to_json()?;
        let b = Actor::from_json(&json)?;
        assert_eq!(a, b);
        Ok(())
    }

    #[test]
    fn shared_inbox_preferred() {
        let a = sample();
        assert_eq!(a.delivery_inbox(), "https://example.test/inbox");
    }

    #[test]
    fn rejects_missing_context() {
        let bad = serde_json::json!({
            "@context": "https://example.test/other",
            "id": "https://example.test/users/bob",
            "type": "Person",
            "preferredUsername": "bob",
            "inbox": "https://example.test/users/bob/inbox",
            "outbox": "https://example.test/users/bob/outbox"
        });
        let res = Actor::from_json(&bad.to_string());
        assert!(res.is_err());
    }
}
