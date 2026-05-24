//! Inbox semantics — receiving Activities and applying side effects.
//!
//! ActivityPub §7 describes what a server SHOULD do when a remote
//! activity is POSTed to its inbox. This module models those rules as
//! pure transitions over an in-memory [`InboxState`].
//!
//! The state is intentionally minimal — followers, the local object
//! store, blocks, and reports. Real servers persist these to disk;
//! the side-effect logic stays the same.

use crate::activity::{Activity, ActivityObject};
use crate::error::{ActivityPubError, Result};
use crate::object::Object;
use crate::vocabulary::ActivityKind;
use std::collections::{BTreeMap, BTreeSet};

/// State a server keeps to implement inbox semantics.
///
/// Each map is keyed by IRI. The intent is to let a server replay an
/// inbox log against a fresh `InboxState` and recover its full state.
#[derive(Debug, Default, Clone)]
pub struct InboxState {
    /// `target_actor → set(follower_actor)`.
    pub followers: BTreeMap<String, BTreeSet<String>>,
    /// Pending Follow activities awaiting Accept / Reject.
    /// Keyed by activity id.
    pub pending_follows: BTreeMap<String, Activity>,
    /// Objects we have received, keyed by id.
    pub objects: BTreeMap<String, Object>,
    /// `object_iri → set(liker_actor)`.
    pub likes: BTreeMap<String, BTreeSet<String>>,
    /// `object_iri → set(boost_activity_id)`.
    pub announces: BTreeMap<String, BTreeSet<String>>,
    /// `actor → set(blocked_actor)`.
    pub blocks: BTreeMap<String, BTreeSet<String>>,
    /// `old_actor_iri → new_actor_iri` from Move activities.
    pub moves: BTreeMap<String, String>,
    /// Flags (reports) received, in arrival order.
    pub flags: Vec<Activity>,
}

impl InboxState {
    /// Fresh empty inbox.
    pub fn new() -> Self {
        Self::default()
    }

    /// Dispatch a received Activity, applying its side effect.
    ///
    /// Returns an [`InboxEffect`] describing what changed. Activities
    /// that violate preconditions return [`ActivityPubError::Rejected`].
    pub fn handle(&mut self, activity: &Activity) -> Result<InboxEffect> {
        let kind = activity
            .kind()
            .ok_or_else(|| ActivityPubError::UnknownActivity(activity.type_field.clone()))?;
        match kind {
            ActivityKind::Create => self.handle_create(activity),
            ActivityKind::Update => self.handle_update(activity),
            ActivityKind::Delete => self.handle_delete(activity),
            ActivityKind::Follow => self.handle_follow(activity),
            ActivityKind::Accept => self.handle_accept(activity),
            ActivityKind::Reject => self.handle_reject(activity),
            ActivityKind::Announce => self.handle_announce(activity),
            ActivityKind::Like => self.handle_like(activity),
            ActivityKind::Undo => self.handle_undo(activity),
            ActivityKind::Move => self.handle_move(activity),
            ActivityKind::Block => self.handle_block(activity),
            ActivityKind::Flag => self.handle_flag(activity),
        }
    }

    fn embedded_object<'a>(&self, a: &'a Activity) -> Result<&'a Object> {
        match &a.object {
            Some(ActivityObject::Embedded(o)) => Ok(o.as_ref()),
            _ => Err(ActivityPubError::Rejected(format!(
                "{} requires inline object",
                a.type_field
            ))),
        }
    }

    fn object_iri<'a>(&self, a: &'a Activity) -> Result<&'a str> {
        a.object
            .as_ref()
            .map(|o| o.iri_str())
            .ok_or_else(|| ActivityPubError::Rejected(format!("{} missing object", a.type_field)))
    }

    fn handle_create(&mut self, a: &Activity) -> Result<InboxEffect> {
        let obj = self.embedded_object(a)?;
        if obj.attributed_to.as_deref() != Some(a.actor.as_str()) {
            return Err(ActivityPubError::Rejected(
                "Create attributedTo must match actor".into(),
            ));
        }
        self.objects.insert(obj.id.clone(), obj.clone());
        Ok(InboxEffect::ObjectStored(obj.id.clone()))
    }

    fn handle_update(&mut self, a: &Activity) -> Result<InboxEffect> {
        let obj = self.embedded_object(a)?;
        let existing = self.objects.get(&obj.id);
        if let Some(existing) = existing {
            if existing.attributed_to != Some(a.actor.clone()) {
                return Err(ActivityPubError::Rejected(
                    "Update must come from the original author".into(),
                ));
            }
        }
        self.objects.insert(obj.id.clone(), obj.clone());
        Ok(InboxEffect::ObjectUpdated(obj.id.clone()))
    }

    fn handle_delete(&mut self, a: &Activity) -> Result<InboxEffect> {
        let iri = self.object_iri(a)?.to_string();
        // Replace with Tombstone — ActivityPub §6.4.
        let former_type = self
            .objects
            .get(&iri)
            .map(|o| o.type_field.clone())
            .unwrap_or_else(|| "Object".to_string());
        let tomb = Object::tombstone(iri.clone(), former_type);
        self.objects.insert(iri.clone(), tomb);
        Ok(InboxEffect::ObjectDeleted(iri))
    }

    fn handle_follow(&mut self, a: &Activity) -> Result<InboxEffect> {
        // Stash it as pending until we Accept or Reject.
        self.pending_follows.insert(a.id.clone(), a.clone());
        Ok(InboxEffect::FollowPending(a.id.clone()))
    }

    fn handle_accept(&mut self, a: &Activity) -> Result<InboxEffect> {
        let prior = match &a.object {
            Some(ActivityObject::Embedded(o)) => {
                if o.type_field == "Follow" {
                    let follower = o.attributed_to.clone().ok_or_else(|| {
                        ActivityPubError::Rejected(
                            "Accept(Follow) missing follower attribution".into(),
                        )
                    })?;
                    self.followers
                        .entry(a.actor.clone())
                        .or_default()
                        .insert(follower.clone());
                    return Ok(InboxEffect::FollowAccepted {
                        target: a.actor.clone(),
                        follower,
                    });
                }
                None
            }
            Some(ActivityObject::Iri(iri)) => {
                // Look up the pending follow we stashed.
                self.pending_follows.remove(iri)
            }
            None => None,
        };
        let pending = prior.ok_or_else(|| {
            ActivityPubError::Rejected("Accept did not reference a Follow we issued".into())
        })?;
        if pending.type_field != "Follow" {
            return Err(ActivityPubError::Rejected(
                "Accept targeted a non-Follow activity".into(),
            ));
        }
        // The actor of the Accept must be the target of the original Follow.
        let original_target = pending
            .object
            .as_ref()
            .map(|o| o.iri_str().to_string())
            .unwrap_or_default();
        if original_target != a.actor {
            return Err(ActivityPubError::Rejected(format!(
                "Accept actor {} does not match Follow target {original_target}",
                a.actor
            )));
        }
        self.followers
            .entry(a.actor.clone())
            .or_default()
            .insert(pending.actor.clone());
        Ok(InboxEffect::FollowAccepted {
            target: a.actor.clone(),
            follower: pending.actor,
        })
    }

    fn handle_reject(&mut self, a: &Activity) -> Result<InboxEffect> {
        if let Some(ActivityObject::Iri(iri)) = &a.object {
            self.pending_follows.remove(iri);
        }
        Ok(InboxEffect::FollowRejected(a.id.clone()))
    }

    fn handle_announce(&mut self, a: &Activity) -> Result<InboxEffect> {
        let iri = self.object_iri(a)?.to_string();
        self.announces
            .entry(iri.clone())
            .or_default()
            .insert(a.id.clone());
        Ok(InboxEffect::Boosted {
            object: iri,
            activity: a.id.clone(),
        })
    }

    fn handle_like(&mut self, a: &Activity) -> Result<InboxEffect> {
        let iri = self.object_iri(a)?.to_string();
        self.likes
            .entry(iri.clone())
            .or_default()
            .insert(a.actor.clone());
        Ok(InboxEffect::Liked {
            object: iri,
            actor: a.actor.clone(),
        })
    }

    fn handle_undo(&mut self, a: &Activity) -> Result<InboxEffect> {
        let prior = a
            .object
            .as_ref()
            .ok_or_else(|| ActivityPubError::Rejected("Undo missing target".into()))?;
        match prior {
            ActivityObject::Embedded(o) => {
                match o.type_field.as_str() {
                    "Follow" => {
                        let target = o
                            .id
                            .clone(); // Follow's id (we only get the activity by reference normally)
                        // Remove follower from target's set.
                        if let Some(target_actor) = o.attributed_to.as_deref() {
                            if target_actor == a.actor {
                                // The Follow's actor is undoing themselves.
                                if let Some(ActivityObject::Iri(tgt)) =
                                    self.pending_follows.get(&target).and_then(|p| p.object.clone())
                                {
                                    if let Some(set) = self.followers.get_mut(&tgt) {
                                        set.remove(&a.actor);
                                    }
                                }
                            }
                        }
                        self.pending_follows.remove(&target);
                        Ok(InboxEffect::Undone)
                    }
                    "Like" => {
                        if let Some(target) = &o.in_reply_to {
                            if let Some(set) = self.likes.get_mut(target) {
                                set.remove(&a.actor);
                            }
                        }
                        Ok(InboxEffect::Undone)
                    }
                    "Announce" => Ok(InboxEffect::Undone),
                    "Block" => {
                        if let Some(target) = &o.in_reply_to {
                            if let Some(set) = self.blocks.get_mut(&a.actor) {
                                set.remove(target);
                            }
                        }
                        Ok(InboxEffect::Undone)
                    }
                    _ => Ok(InboxEffect::Undone),
                }
            }
            ActivityObject::Iri(_) => Ok(InboxEffect::Undone),
        }
    }

    fn handle_move(&mut self, a: &Activity) -> Result<InboxEffect> {
        let target = a
            .target
            .clone()
            .or_else(|| a.object.as_ref().map(|o| o.iri_str().to_string()))
            .ok_or_else(|| ActivityPubError::Rejected("Move missing target".into()))?;
        self.moves.insert(a.actor.clone(), target.clone());
        Ok(InboxEffect::Moved {
            from: a.actor.clone(),
            to: target,
        })
    }

    fn handle_block(&mut self, a: &Activity) -> Result<InboxEffect> {
        let target = self.object_iri(a)?.to_string();
        self.blocks
            .entry(a.actor.clone())
            .or_default()
            .insert(target.clone());
        Ok(InboxEffect::Blocked {
            actor: a.actor.clone(),
            target,
        })
    }

    fn handle_flag(&mut self, a: &Activity) -> Result<InboxEffect> {
        self.flags.push(a.clone());
        Ok(InboxEffect::Flagged(a.id.clone()))
    }
}

/// Side-effect summary returned from [`InboxState::handle`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InboxEffect {
    /// `Create` — object stored under the given id.
    ObjectStored(String),
    /// `Update` — object id replaced.
    ObjectUpdated(String),
    /// `Delete` — object id tombstoned.
    ObjectDeleted(String),
    /// `Follow` — request stashed pending Accept/Reject.
    FollowPending(String),
    /// `Accept` of a Follow — follower added to target's set.
    FollowAccepted {
        /// IRI of the followed actor.
        target: String,
        /// IRI of the new follower.
        follower: String,
    },
    /// `Reject` of a Follow — pending entry removed.
    FollowRejected(String),
    /// `Announce` — boost recorded.
    Boosted {
        /// IRI of the boosted object.
        object: String,
        /// IRI of the Announce activity.
        activity: String,
    },
    /// `Like` — like recorded.
    Liked {
        /// IRI of the liked object.
        object: String,
        /// IRI of the liker.
        actor: String,
    },
    /// `Undo` — prior activity reversed.
    Undone,
    /// `Move` — actor migration recorded.
    Moved {
        /// IRI of the old account.
        from: String,
        /// IRI of the new account.
        to: String,
    },
    /// `Block` — block recorded.
    Blocked {
        /// IRI of the blocker.
        actor: String,
        /// IRI of the blocked party.
        target: String,
    },
    /// `Flag` — report recorded.
    Flagged(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_then_like() -> Result<()> {
        let mut state = InboxState::new();
        let note = Object::note(
            "https://a.test/notes/1",
            "https://a.test/users/alice",
            "hi",
        );
        let create = Activity::create(
            "https://a.test/activities/c",
            "https://a.test/users/alice",
            note,
        );
        let e = state.handle(&create)?;
        assert_eq!(e, InboxEffect::ObjectStored("https://a.test/notes/1".into()));
        let like = Activity::like(
            "https://b.test/activities/l",
            "https://b.test/users/bob",
            "https://a.test/notes/1",
        );
        let e2 = state.handle(&like)?;
        assert_eq!(
            e2,
            InboxEffect::Liked {
                object: "https://a.test/notes/1".into(),
                actor: "https://b.test/users/bob".into()
            }
        );
        assert!(state.likes["https://a.test/notes/1"].contains("https://b.test/users/bob"));
        Ok(())
    }

    #[test]
    fn create_rejects_actor_mismatch() {
        let mut state = InboxState::new();
        let note = Object::note(
            "https://a.test/notes/1",
            "https://a.test/users/alice",
            "hi",
        );
        let create = Activity::create(
            "https://a.test/activities/c",
            "https://b.test/users/mallory",
            note,
        );
        assert!(state.handle(&create).is_err());
    }
}
