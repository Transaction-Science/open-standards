//! ActivityStreams Activities and builders.
//!
//! An Activity is a wrapper that records *who* did *what* to *which
//! object*, with addressing for delivery. ActivityPub §6 enumerates the
//! activity types every server is expected to understand; this module
//! models the union of fields they share, plus typed constructors for
//! each kind.

use crate::error::{ActivityPubError, Result};
use crate::object::Object;
use crate::vocabulary::{ActivityKind, AS2_CONTEXT, PUBLIC};
use serde::{Deserialize, Serialize};

/// `object` / `target` field — either an inline Object or just an IRI.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ActivityObject {
    /// Bare IRI string.
    Iri(String),
    /// Full inline object.
    Embedded(Box<Object>),
}

impl ActivityObject {
    /// Construct from any string-ish IRI.
    pub fn iri(s: impl Into<String>) -> Self {
        ActivityObject::Iri(s.into())
    }

    /// Construct from an inline Object.
    pub fn embedded(o: Object) -> Self {
        ActivityObject::Embedded(Box::new(o))
    }

    /// Return the object's IRI (the `id` field if embedded).
    pub fn iri_str(&self) -> &str {
        match self {
            ActivityObject::Iri(s) => s.as_str(),
            ActivityObject::Embedded(o) => o.id.as_str(),
        }
    }
}

/// An AS2 Activity.
///
/// We deliberately keep one struct for all twelve activity types and
/// distinguish on the `type` string. That mirrors the wire format and
/// keeps the inbox dispatcher simple.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Activity {
    /// JSON-LD context.
    #[serde(rename = "@context")]
    pub context: serde_json::Value,
    /// Activity IRI.
    pub id: String,
    /// AS2 `type`.
    #[serde(rename = "type")]
    pub type_field: String,
    /// IRI of the actor performing the activity.
    pub actor: String,
    /// Primary object — required for Create/Update/Delete/Follow/Like/etc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object: Option<ActivityObject>,
    /// `target` — used for Move and Add/Remove activities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// `to` recipients (resolved by [`crate::addressing`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub to: Vec<String>,
    /// `cc` recipients.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cc: Vec<String>,
    /// `bcc` recipients — stripped on delivery.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bcc: Vec<String>,
    /// Publication timestamp (RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published: Option<String>,
    /// Optional natural-language summary (Mastodon uses this for Flag).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

impl Activity {
    /// Internal builder — every typed constructor flows through this.
    fn build(
        kind: ActivityKind,
        id: impl Into<String>,
        actor: impl Into<String>,
        object: Option<ActivityObject>,
    ) -> Self {
        Self {
            context: serde_json::Value::String(AS2_CONTEXT.into()),
            id: id.into(),
            type_field: kind.as_type().to_string(),
            actor: actor.into(),
            object,
            target: None,
            to: Vec::new(),
            cc: Vec::new(),
            bcc: Vec::new(),
            published: None,
            summary: None,
        }
    }

    /// `Create` an inline Object.
    pub fn create(id: impl Into<String>, actor: impl Into<String>, object: Object) -> Self {
        let mut a = Self::build(
            ActivityKind::Create,
            id,
            actor,
            Some(ActivityObject::embedded(object.clone())),
        );
        // Mirror the object's addressing onto the activity (ActivityPub §6.2).
        a.to = object.to.clone();
        a.cc = object.cc.clone();
        a
    }

    /// `Follow` another actor.
    pub fn follow(
        id: impl Into<String>,
        actor: impl Into<String>,
        target_actor: impl Into<String>,
    ) -> Self {
        let target = target_actor.into();
        let mut a = Self::build(
            ActivityKind::Follow,
            id,
            actor,
            Some(ActivityObject::iri(target.clone())),
        );
        a.to = vec![target];
        a
    }

    /// `Accept` a prior activity (typically a Follow).
    pub fn accept(
        id: impl Into<String>,
        actor: impl Into<String>,
        prior: ActivityObject,
    ) -> Self {
        let prior_actor = match &prior {
            ActivityObject::Embedded(o) => o.attributed_to.clone(),
            _ => None,
        };
        let mut a = Self::build(ActivityKind::Accept, id, actor, Some(prior));
        if let Some(to) = prior_actor {
            a.to = vec![to];
        }
        a
    }

    /// `Reject` a prior activity.
    pub fn reject(id: impl Into<String>, actor: impl Into<String>, prior: ActivityObject) -> Self {
        Self::build(ActivityKind::Reject, id, actor, Some(prior))
    }

    /// `Announce` (boost) another object.
    pub fn announce(
        id: impl Into<String>,
        actor: impl Into<String>,
        object_iri: impl Into<String>,
    ) -> Self {
        let mut a = Self::build(
            ActivityKind::Announce,
            id,
            actor,
            Some(ActivityObject::iri(object_iri)),
        );
        a.to = vec![PUBLIC.to_string()];
        a
    }

    /// `Like` an object.
    pub fn like(
        id: impl Into<String>,
        actor: impl Into<String>,
        object_iri: impl Into<String>,
    ) -> Self {
        Self::build(
            ActivityKind::Like,
            id,
            actor,
            Some(ActivityObject::iri(object_iri)),
        )
    }

    /// `Update` an object (publish a new version).
    pub fn update(id: impl Into<String>, actor: impl Into<String>, object: Object) -> Self {
        Self::build(
            ActivityKind::Update,
            id,
            actor,
            Some(ActivityObject::embedded(object)),
        )
    }

    /// `Delete` an object (server replaces it with a Tombstone).
    pub fn delete(
        id: impl Into<String>,
        actor: impl Into<String>,
        object_iri: impl Into<String>,
    ) -> Self {
        Self::build(
            ActivityKind::Delete,
            id,
            actor,
            Some(ActivityObject::iri(object_iri)),
        )
    }

    /// `Undo` a prior activity.
    pub fn undo(id: impl Into<String>, actor: impl Into<String>, prior: ActivityObject) -> Self {
        Self::build(ActivityKind::Undo, id, actor, Some(prior))
    }

    /// `Move` an actor to a new IRI.
    pub fn move_to(
        id: impl Into<String>,
        actor: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        let mut a = Self::build(
            ActivityKind::Move,
            id,
            actor,
            Some(ActivityObject::iri(target.into())),
        );
        a.target = a.object.as_ref().map(|o| o.iri_str().to_string());
        a
    }

    /// `Block` another actor.
    pub fn block(
        id: impl Into<String>,
        actor: impl Into<String>,
        target_actor: impl Into<String>,
    ) -> Self {
        Self::build(
            ActivityKind::Block,
            id,
            actor,
            Some(ActivityObject::iri(target_actor)),
        )
    }

    /// `Flag` (report) one or more objects/actors with a free-form reason.
    pub fn flag(
        id: impl Into<String>,
        actor: impl Into<String>,
        target_iri: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        let mut a = Self::build(
            ActivityKind::Flag,
            id,
            actor,
            Some(ActivityObject::iri(target_iri)),
        );
        a.summary = Some(reason.into());
        a
    }

    /// Return the typed [`ActivityKind`] if the `type` is recognised.
    pub fn kind(&self) -> Option<ActivityKind> {
        ActivityKind::parse(&self.type_field)
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Parse from JSON, validating that the activity type is recognised.
    pub fn from_json(s: &str) -> Result<Self> {
        let a: Activity = serde_json::from_str(s)?;
        if a.id.is_empty() {
            return Err(ActivityPubError::Vocabulary("activity missing id".into()));
        }
        if a.actor.is_empty() {
            return Err(ActivityPubError::Vocabulary(
                "activity missing actor".into(),
            ));
        }
        if ActivityKind::parse(&a.type_field).is_none() {
            return Err(ActivityPubError::UnknownActivity(a.type_field.clone()));
        }
        Ok(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follow_addresses_target() {
        let f = Activity::follow(
            "https://a.test/activities/1",
            "https://a.test/users/alice",
            "https://b.test/users/bob",
        );
        assert_eq!(f.kind(), Some(ActivityKind::Follow));
        assert_eq!(f.to, vec!["https://b.test/users/bob".to_string()]);
    }

    #[test]
    fn create_mirrors_object_addressing() {
        let note = Object::note(
            "https://a.test/notes/1",
            "https://a.test/users/alice",
            "hi",
        )
        .cc("https://a.test/users/alice/followers");
        let c = Activity::create(
            "https://a.test/activities/c1",
            "https://a.test/users/alice",
            note,
        );
        assert!(c.to.iter().any(|t| t == PUBLIC));
        assert!(c
            .cc
            .iter()
            .any(|t| t == "https://a.test/users/alice/followers"));
    }

    #[test]
    fn json_roundtrip() -> Result<()> {
        let a = Activity::like(
            "https://a.test/activities/2",
            "https://a.test/users/alice",
            "https://b.test/notes/9",
        );
        let s = a.to_json()?;
        let b = Activity::from_json(&s)?;
        assert_eq!(a, b);
        Ok(())
    }
}
