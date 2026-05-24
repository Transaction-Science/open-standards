//! Outbox — local publication + delivery queue.
//!
//! When a local actor publishes a Create / Follow / Announce / Like /
//! etc, the activity is appended to the outbox and queued for delivery
//! to the audience computed from [`crate::addressing`].
//!
//! Delivery is intentionally in-memory and synchronous here. A
//! production server replaces [`DeliveryQueue::flush`] with HTTP POSTs
//! signed via [`crate::http_sig`]; the planning surface is identical.

use crate::activity::Activity;
use crate::addressing::{resolve_audience, strip_bcc, CollectionResolver, ResolvedAudience};
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A single queued delivery.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingDelivery {
    /// The fully-prepared Activity body (bcc stripped, signing pending).
    pub body: Activity,
    /// Inbox IRIs that still need a successful POST.
    pub inboxes: Vec<String>,
}

/// In-memory delivery queue + outbox log.
#[derive(Debug, Default, Clone)]
pub struct Outbox {
    /// Append-only log of activities the local actor has published.
    pub log: Vec<Activity>,
    /// Queue of pending deliveries.
    pub queue: VecDeque<PendingDelivery>,
}

impl Outbox {
    /// Fresh empty outbox.
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish an Activity locally — append to the log and enqueue
    /// deliveries for every resolved inbox.
    pub fn publish<R: CollectionResolver>(
        &mut self,
        activity: Activity,
        resolver: &R,
        actor_inboxes: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ResolvedAudience> {
        self.log.push(activity.clone());
        let audience = resolve_audience(&activity, resolver);
        let body = strip_bcc(&activity);
        let mut inboxes: Vec<String> = audience
            .actors
            .iter()
            .filter_map(|a| actor_inboxes(a))
            .collect();
        inboxes.sort();
        inboxes.dedup();
        if !inboxes.is_empty() {
            self.queue.push_back(PendingDelivery { body, inboxes });
        }
        Ok(audience)
    }

    /// Pop the next pending delivery, or `None` if the queue is empty.
    pub fn next(&mut self) -> Option<PendingDelivery> {
        self.queue.pop_front()
    }

    /// Drain the queue by invoking the supplied closure on each
    /// delivery. The closure returns the set of inboxes that *failed*;
    /// those are re-enqueued for retry.
    pub fn flush<F>(&mut self, mut deliver: F)
    where
        F: FnMut(&Activity, &str) -> bool,
    {
        let mut retries = VecDeque::new();
        while let Some(d) = self.queue.pop_front() {
            let mut failed = Vec::new();
            for inbox in &d.inboxes {
                if !deliver(&d.body, inbox) {
                    failed.push(inbox.clone());
                }
            }
            if !failed.is_empty() {
                retries.push_back(PendingDelivery {
                    body: d.body,
                    inboxes: failed,
                });
            }
        }
        self.queue = retries;
    }

    /// Number of activities the local actor has published.
    pub fn total_items(&self) -> u64 {
        self.log.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addressing::StaticResolver;
    use crate::object::Object;

    #[test]
    fn publish_enqueues_for_each_inbox() -> Result<()> {
        let mut ob = Outbox::new();
        let mut resolver = StaticResolver::new();
        resolver.insert(
            "https://a.test/users/alice/followers",
            vec![
                "https://b.test/users/bob".into(),
                "https://c.test/users/carol".into(),
            ],
        );
        let note = Object::note(
            "https://a.test/notes/1",
            "https://a.test/users/alice",
            "hi",
        )
        .cc("https://a.test/users/alice/followers");
        let create = Activity::create(
            "https://a.test/activities/c",
            "https://a.test/users/alice",
            note,
        );
        ob.publish(create, &resolver, &|actor| match actor {
            "https://b.test/users/bob" => Some("https://b.test/users/bob/inbox".into()),
            "https://c.test/users/carol" => Some("https://c.test/users/carol/inbox".into()),
            _ => None,
        })?;
        assert_eq!(ob.queue.len(), 1);
        let d = ob.next().expect("one delivery");
        assert_eq!(d.inboxes.len(), 2);
        Ok(())
    }

    #[test]
    fn flush_retries_failed_inboxes() -> Result<()> {
        let mut ob = Outbox::new();
        let resolver = StaticResolver::new();
        let note = Object::note(
            "https://a.test/notes/1",
            "https://a.test/users/alice",
            "hi",
        );
        let mut create = Activity::create(
            "https://a.test/activities/c",
            "https://a.test/users/alice",
            note,
        );
        create.to.push("https://b.test/users/bob".into());
        create.to.push("https://c.test/users/carol".into());
        ob.publish(create, &resolver, &|actor| match actor {
            "https://b.test/users/bob" => Some("https://b.test/users/bob/inbox".into()),
            "https://c.test/users/carol" => Some("https://c.test/users/carol/inbox".into()),
            _ => None,
        })?;
        ob.flush(|_a, inbox| inbox.contains("bob")); // only bob succeeds
        assert_eq!(ob.queue.len(), 1, "carol should be queued for retry");
        Ok(())
    }
}
