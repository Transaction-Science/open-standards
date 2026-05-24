//! ActivityPub (Mastodon-compatible) federation adapter for Smart Byte.
//!
//! ActivityPub is the W3C 2018 Recommendation that powers the so-called
//! "Fediverse": Mastodon, Pleroma, Misskey, Pixelfed, GoToSocial and a
//! long tail of compatible servers. This crate ingests that deployed
//! footprint into the Smart Byte substrate so other crates can publish,
//! deliver, and consume Activities without taking a runtime dependency
//! on any specific server implementation.
//!
//! ## Layout
//!
//! * **Vocabulary** ([`vocabulary`]) — AS2 namespaces + the activity,
//!   actor, and object kind enums.
//! * **Actor** ([`actor`]) — `Person` / `Application` / `Group` /
//!   `Organization` / `Service` plus the `publicKey` + `endpoints`
//!   block.
//! * **Object** ([`object`]) — `Note` / `Article` / `Image` / `Video` /
//!   `Tombstone`, with Mastodon extensions (`sensitive`, `summary`,
//!   `conversation`, `blurhash`).
//! * **Activity** ([`activity`]) — typed builders for the twelve
//!   activities every server is expected to understand: `Create`,
//!   `Update`, `Delete`, `Follow`, `Accept`, `Reject`, `Announce`,
//!   `Like`, `Undo`, `Move`, `Block`, `Flag`.
//! * **Collection** ([`collection`]) — `OrderedCollection` and
//!   `OrderedCollectionPage`.
//! * **Webfinger** ([`webfinger`]) — RFC 7033 user discovery; the
//!   `acct:` ↔ actor-IRI bridge.
//! * **HTTP Signatures** ([`http_sig`]) — draft-cavage-12 +
//!   RFC 9421 helpers, with Ed25519 sign / verify and the
//!   `Digest: SHA-256=…` body hash.
//! * **Addressing** ([`addressing`]) — `Public` + `to` / `cc` / `bcc`
//!   resolution and shared-inbox selection.
//! * **Inbox** ([`inbox`]) — pure side-effect rules for every activity
//!   kind, expressed as transitions over an in-memory state.
//! * **Outbox** ([`outbox`]) — local publication log + delivery queue.
//!
//! ## What's intentionally scoped out
//!
//! * HTTP I/O. The signing helpers are pure; transports live in the
//!   caller's runtime crate.
//! * RSA HTTP Signatures. We sign / verify Ed25519 only; cavage's
//!   `rsa-sha256` is left as a parse-only path (the parameter parser
//!   accepts it, the signer rejects it). Servers that need RSA wrap
//!   their own.
//! * Full JSON-LD processing. We treat `@context` as an unparsed value
//!   used only for AS2-presence checks. Mastodon does the same.
//! * Persistence. Inbox / outbox state structures are in-memory; a
//!   real server checkpoints them to disk.

#![forbid(unsafe_code)]

pub mod activity;
pub mod actor;
pub mod addressing;
pub mod collection;
pub mod error;
pub mod http_sig;
pub mod inbox;
pub mod object;
pub mod outbox;
pub mod vocabulary;
pub mod webfinger;

pub use activity::{Activity, ActivityObject};
pub use actor::{Actor, Endpoints, PublicKey};
pub use addressing::{
    delivery_inboxes, resolve_audience, strip_bcc, CollectionResolver, ResolvedAudience,
    StaticResolver,
};
pub use collection::{OrderedCollection, OrderedCollectionPage};
pub use error::{ActivityPubError, Result};
pub use http_sig::{sign_ed25519, verify_ed25519, Digest, SignatureParams, SigningString};
pub use inbox::{InboxEffect, InboxState};
pub use object::{Attachment, Object};
pub use outbox::{Outbox, PendingDelivery};
pub use vocabulary::{
    ActivityKind, ActorKind, ObjectKind, ACTIVITY_JSON, ACTIVITY_LD_JSON, AS2_CONTEXT, PUBLIC,
    SECURITY_CONTEXT, TOOT_CONTEXT, WEBFINGER_MEDIA_TYPE,
};
pub use webfinger::{discovery_url, parse_acct, Jrd, Link};
