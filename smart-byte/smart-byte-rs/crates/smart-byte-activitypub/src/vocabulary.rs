//! ActivityStreams 2.0 vocabulary constants.
//!
//! These are the JSON-LD `@context` URIs and the small set of string
//! constants that recur across `type`, addressing, and discovery
//! documents. Centralising them avoids drift between the actor, object,
//! activity and inbox modules.
//!
//! References:
//! * ActivityStreams 2.0 Core — <https://www.w3.org/TR/activitystreams-core/>
//! * ActivityStreams 2.0 Vocabulary — <https://www.w3.org/TR/activitystreams-vocabulary/>
//! * ActivityPub — <https://www.w3.org/TR/activitypub/>

/// Canonical ActivityStreams 2.0 namespace.
pub const AS2_CONTEXT: &str = "https://www.w3.org/ns/activitystreams";

/// W3C Security vocabulary; required to publish `publicKey` on an actor
/// (used for HTTP Signature verification).
pub const SECURITY_CONTEXT: &str = "https://w3id.org/security/v1";

/// Mastodon extension namespace — provides `featured`, `blurhash`,
/// `sensitive`, `conversation` and friends.
pub const TOOT_CONTEXT: &str = "http://joinmastodon.org/ns#";

/// The magic `to`/`cc` value that means "everyone, fully public".
/// See ActivityPub §5.6.
pub const PUBLIC: &str = "https://www.w3.org/ns/activitystreams#Public";

/// Webfinger media type per RFC 7033 §10.2.
pub const WEBFINGER_MEDIA_TYPE: &str = "application/jrd+json";

/// ActivityPub media type for incoming and outgoing `application/activity+json`.
pub const ACTIVITY_JSON: &str = "application/activity+json";

/// JSON-LD media type with the activitystreams profile parameter.
pub const ACTIVITY_LD_JSON: &str =
    "application/ld+json; profile=\"https://www.w3.org/ns/activitystreams\"";

/// Activity types recognised by this adapter.
///
/// The set follows ActivityStreams §3.5 + the side-effect rules in
/// ActivityPub §6. We bundle them in one enum so the inbox handler can
/// dispatch with a single match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivityKind {
    /// Create — publish a new object (e.g. a Note).
    Create,
    /// Update — modify an existing object.
    Update,
    /// Delete — tombstone an object.
    Delete,
    /// Follow — request to follow an actor.
    Follow,
    /// Accept — accept a Follow (or other) request.
    Accept,
    /// Reject — reject a Follow (or other) request.
    Reject,
    /// Announce — boost / share another object.
    Announce,
    /// Like — favourite an object.
    Like,
    /// Undo — reverse a prior activity (Follow, Like, Announce…).
    Undo,
    /// Move — actor moved to a new account.
    Move,
    /// Block — actor blocks another actor.
    Block,
    /// Flag — report an object or actor.
    Flag,
}

impl ActivityKind {
    /// Render the kind as the AS2 `type` string.
    pub fn as_type(self) -> &'static str {
        match self {
            ActivityKind::Create => "Create",
            ActivityKind::Update => "Update",
            ActivityKind::Delete => "Delete",
            ActivityKind::Follow => "Follow",
            ActivityKind::Accept => "Accept",
            ActivityKind::Reject => "Reject",
            ActivityKind::Announce => "Announce",
            ActivityKind::Like => "Like",
            ActivityKind::Undo => "Undo",
            ActivityKind::Move => "Move",
            ActivityKind::Block => "Block",
            ActivityKind::Flag => "Flag",
        }
    }

    /// Parse the kind from an AS2 `type` string.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "Create" => Some(ActivityKind::Create),
            "Update" => Some(ActivityKind::Update),
            "Delete" => Some(ActivityKind::Delete),
            "Follow" => Some(ActivityKind::Follow),
            "Accept" => Some(ActivityKind::Accept),
            "Reject" => Some(ActivityKind::Reject),
            "Announce" => Some(ActivityKind::Announce),
            "Like" => Some(ActivityKind::Like),
            "Undo" => Some(ActivityKind::Undo),
            "Move" => Some(ActivityKind::Move),
            "Block" => Some(ActivityKind::Block),
            "Flag" => Some(ActivityKind::Flag),
            _ => None,
        }
    }
}

/// Object types recognised by this adapter — limited to the ones with
/// observable side effects on a Mastodon-style server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectKind {
    /// A short text post (Mastodon-style toot).
    Note,
    /// A long-form article.
    Article,
    /// An attached or standalone image.
    Image,
    /// An attached or standalone video.
    Video,
    /// A tombstone left behind by a Delete activity.
    Tombstone,
}

impl ObjectKind {
    /// Render the kind as the AS2 `type` string.
    pub fn as_type(self) -> &'static str {
        match self {
            ObjectKind::Note => "Note",
            ObjectKind::Article => "Article",
            ObjectKind::Image => "Image",
            ObjectKind::Video => "Video",
            ObjectKind::Tombstone => "Tombstone",
        }
    }

    /// Parse the kind from an AS2 `type` string.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "Note" => Some(ObjectKind::Note),
            "Article" => Some(ObjectKind::Article),
            "Image" => Some(ObjectKind::Image),
            "Video" => Some(ObjectKind::Video),
            "Tombstone" => Some(ObjectKind::Tombstone),
            _ => None,
        }
    }
}

/// Actor types recognised by this adapter (ActivityStreams §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActorKind {
    /// A real-world person.
    Person,
    /// An automated bot / service.
    Application,
    /// A community / group.
    Group,
    /// An organisation.
    Organization,
    /// A service endpoint (often used by relays).
    Service,
}

impl ActorKind {
    /// Render the kind as the AS2 `type` string.
    pub fn as_type(self) -> &'static str {
        match self {
            ActorKind::Person => "Person",
            ActorKind::Application => "Application",
            ActorKind::Group => "Group",
            ActorKind::Organization => "Organization",
            ActorKind::Service => "Service",
        }
    }

    /// Parse the kind from an AS2 `type` string.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "Person" => Some(ActorKind::Person),
            "Application" => Some(ActorKind::Application),
            "Group" => Some(ActorKind::Group),
            "Organization" => Some(ActorKind::Organization),
            "Service" => Some(ActorKind::Service),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_kind_roundtrip() {
        for kind in [
            ActivityKind::Create,
            ActivityKind::Update,
            ActivityKind::Delete,
            ActivityKind::Follow,
            ActivityKind::Accept,
            ActivityKind::Reject,
            ActivityKind::Announce,
            ActivityKind::Like,
            ActivityKind::Undo,
            ActivityKind::Move,
            ActivityKind::Block,
            ActivityKind::Flag,
        ] {
            assert_eq!(ActivityKind::parse(kind.as_type()), Some(kind));
        }
    }

    #[test]
    fn object_kind_roundtrip() {
        for kind in [
            ObjectKind::Note,
            ObjectKind::Article,
            ObjectKind::Image,
            ObjectKind::Video,
            ObjectKind::Tombstone,
        ] {
            assert_eq!(ObjectKind::parse(kind.as_type()), Some(kind));
        }
    }

    #[test]
    fn actor_kind_roundtrip() {
        for kind in [
            ActorKind::Person,
            ActorKind::Application,
            ActorKind::Group,
            ActorKind::Organization,
            ActorKind::Service,
        ] {
            assert_eq!(ActorKind::parse(kind.as_type()), Some(kind));
        }
    }
}
