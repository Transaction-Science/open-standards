//! Addressing — resolving `to`, `cc`, `bcc`, and the magic `Public`
//! collection (ActivityPub §5.6).
//!
//! Each Activity carries a set of addressing fields. To deliver it we
//! flatten those into a deduplicated list of *inboxes*, expanding the
//! special `Public` IRI and any followers / collections we know about.

use crate::actor::Actor;
use crate::activity::Activity;
use crate::vocabulary::PUBLIC;
use std::collections::BTreeSet;

/// A resolver from collection IRI → list of member actor IRIs.
///
/// Inboxes are usually computed from local followers / following
/// collections, so this trait is the extension point a server plugs
/// into. The default in-crate implementation is the
/// [`StaticResolver`] used in tests.
pub trait CollectionResolver {
    /// Return the list of actor IRIs in the given collection, or
    /// `None` if the IRI is not a collection we know about.
    fn resolve(&self, iri: &str) -> Option<Vec<String>>;
}

/// In-memory resolver — primarily for tests. Maps each known
/// collection IRI to a fixed set of actor IRIs.
#[derive(Debug, Default, Clone)]
pub struct StaticResolver {
    /// Underlying map.
    pub map: std::collections::BTreeMap<String, Vec<String>>,
}

impl StaticResolver {
    /// Construct an empty resolver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a collection IRI → actor list.
    pub fn insert(&mut self, iri: impl Into<String>, members: Vec<String>) {
        self.map.insert(iri.into(), members);
    }
}

impl CollectionResolver for StaticResolver {
    fn resolve(&self, iri: &str) -> Option<Vec<String>> {
        self.map.get(iri).cloned()
    }
}

/// Result of resolving an Activity's addressing — a deduplicated set
/// of actor IRIs (the targets) and a flag for whether the activity is
/// public.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAudience {
    /// True when the Activity is addressed to the magic Public IRI.
    pub is_public: bool,
    /// Concrete actor IRIs the Activity targets (excluding `Public`).
    pub actors: BTreeSet<String>,
}

/// Resolve an Activity's audience into concrete actor IRIs.
///
/// `to` and `cc` are expanded via the resolver; `bcc` is included in
/// the delivery list but *stripped* from the activity that gets
/// posted to each inbox (ActivityPub §5.6).
pub fn resolve_audience<R: CollectionResolver>(
    activity: &Activity,
    resolver: &R,
) -> ResolvedAudience {
    let mut actors = BTreeSet::new();
    let mut is_public = false;
    for field in activity
        .to
        .iter()
        .chain(activity.cc.iter())
        .chain(activity.bcc.iter())
    {
        if field == PUBLIC {
            is_public = true;
            continue;
        }
        if let Some(members) = resolver.resolve(field) {
            for m in members {
                actors.insert(m);
            }
        } else {
            actors.insert(field.clone());
        }
    }
    ResolvedAudience { is_public, actors }
}

/// Strip `bcc` from an Activity before serialising for delivery.
pub fn strip_bcc(activity: &Activity) -> Activity {
    let mut clone = activity.clone();
    clone.bcc.clear();
    clone
}

/// Compute the set of unique inboxes to deliver to, given a mapping
/// from actor IRI to their loaded [`Actor`] document.
///
/// Prefers the shared inbox where one is published — the standard
/// Mastodon optimisation.
pub fn delivery_inboxes<'a, I>(actors: I) -> BTreeSet<String>
where
    I: IntoIterator<Item = &'a Actor>,
{
    actors
        .into_iter()
        .map(|a| a.delivery_inbox().to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::{Actor, Endpoints};
    use crate::vocabulary::ActorKind;

    #[test]
    fn resolves_public_and_collection() {
        let mut r = StaticResolver::new();
        r.insert(
            "https://a.test/users/alice/followers",
            vec![
                "https://b.test/users/bob".to_string(),
                "https://c.test/users/carol".to_string(),
            ],
        );
        let a = Activity::create(
            "https://a.test/activities/1",
            "https://a.test/users/alice",
            crate::object::Object::note(
                "https://a.test/notes/1",
                "https://a.test/users/alice",
                "hi",
            )
            .cc("https://a.test/users/alice/followers"),
        );
        let res = resolve_audience(&a, &r);
        assert!(res.is_public);
        assert!(res.actors.contains("https://b.test/users/bob"));
        assert!(res.actors.contains("https://c.test/users/carol"));
    }

    #[test]
    fn inbox_prefers_shared() {
        let mut a = Actor::new(
            ActorKind::Person,
            "https://b.test/users/bob",
            "bob",
            "https://b.test/users/bob/inbox",
            "https://b.test/users/bob/outbox",
        );
        a.endpoints = Some(Endpoints {
            shared_inbox: Some("https://b.test/inbox".into()),
        });
        let set = delivery_inboxes([&a]);
        assert!(set.contains("https://b.test/inbox"));
    }

    #[test]
    fn strip_bcc_removes_field() {
        let mut a = Activity::create(
            "https://a.test/activities/1",
            "https://a.test/users/alice",
            crate::object::Object::note(
                "https://a.test/notes/1",
                "https://a.test/users/alice",
                "hi",
            ),
        );
        a.bcc.push("https://b.test/users/bob".into());
        let stripped = strip_bcc(&a);
        assert!(stripped.bcc.is_empty());
        assert!(!a.bcc.is_empty());
    }
}
