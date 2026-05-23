//! Pluggable subscription storage.
//!
//! Same shape as the other domain stores: a sync trait plus an
//! [`InMemorySubscriptionStore`] reference impl. A graph-backed
//! impl lives in `op-graph`.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::error::{Error, Result};
use crate::subscription::{Subscription, SubscriptionId};

/// Pluggable subscription storage interface.
pub trait SubscriptionStore: Send + Sync {
    /// Persist a fresh subscription. Idempotent on `external_id`.
    ///
    /// # Errors
    /// See [`Error`].
    fn create_subscription(&self, s: Subscription) -> Result<SubscriptionId>;

    /// Fetch by id.
    ///
    /// # Errors
    /// [`Error::NotFound`] if unknown.
    fn get_subscription(&self, id: SubscriptionId) -> Result<Subscription>;

    /// Find by external id.
    ///
    /// # Errors
    /// Backend-specific.
    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Subscription>>;

    /// All subscriptions for a customer. Order unspecified.
    ///
    /// # Errors
    /// Backend-specific.
    fn list_for_customer(&self, customer_ref: &str) -> Result<Vec<Subscription>>;

    /// All non-terminal subscriptions whose
    /// `current_period_end_unix_secs <= as_of`. Used by the
    /// billing tick to fan out due-charge events.
    ///
    /// # Errors
    /// Backend-specific.
    fn list_due_at(&self, as_of_unix_secs: u64) -> Result<Vec<Subscription>>;

    /// Apply a state transition via closure. Commits only on
    /// `Ok(())`.
    ///
    /// # Errors
    /// `Error::NotFound`, plus whatever the closure returns.
    fn update<F>(&self, id: SubscriptionId, f: F) -> Result<Subscription>
    where
        F: FnOnce(&mut Subscription) -> Result<()>;
}

// ============================================================
// In-memory ref impl
// ============================================================

#[derive(Default)]
struct Inner {
    subs: HashMap<SubscriptionId, Subscription>,
    by_external_id: HashMap<String, SubscriptionId>,
}

/// In-process subscription store. Not for multi-instance production.
#[derive(Default)]
pub struct InMemorySubscriptionStore {
    inner: Mutex<Inner>,
}

impl InMemorySubscriptionStore {
    /// Construct empty.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of subscriptions tracked.
    ///
    /// # Panics
    /// Panics only if the internal mutex was previously poisoned
    /// by a thread that panicked while holding it.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().expect("poisoned").subs.len()
    }

    /// True if no subscriptions stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn bodies_equivalent(a: &Subscription, b: &Subscription) -> bool {
    a.customer_ref == b.customer_ref
        && a.plan.id == b.plan.id
        && a.plan.amount == b.plan.amount
        && a.plan.interval == b.plan.interval
        && a.plan.interval_count == b.plan.interval_count
        && a.external_id == b.external_id
}

impl SubscriptionStore for InMemorySubscriptionStore {
    fn create_subscription(&self, s: Subscription) -> Result<SubscriptionId> {
        let mut inner = self.inner.lock().expect("poisoned");
        if let Some(ext) = &s.external_id
            && let Some(existing_id) = inner.by_external_id.get(ext).copied()
        {
            let existing = inner
                .subs
                .get(&existing_id)
                .expect("index/store invariant")
                .clone();
            if bodies_equivalent(&existing, &s) {
                return Ok(existing_id);
            }
            return Err(Error::IdempotencyMismatch(ext.clone()));
        }
        if let Some(ext) = &s.external_id {
            inner.by_external_id.insert(ext.clone(), s.id);
        }
        let id = s.id;
        inner.subs.insert(id, s);
        Ok(id)
    }

    fn get_subscription(&self, id: SubscriptionId) -> Result<Subscription> {
        self.inner
            .lock()
            .expect("poisoned")
            .subs
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::NotFound(id.to_string()))
    }

    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Subscription>> {
        let inner = self.inner.lock().expect("poisoned");
        let Some(id) = inner.by_external_id.get(external_id).copied() else {
            return Ok(None);
        };
        Ok(inner.subs.get(&id).cloned())
    }

    fn list_for_customer(&self, customer_ref: &str) -> Result<Vec<Subscription>> {
        Ok(self
            .inner
            .lock()
            .expect("poisoned")
            .subs
            .values()
            .filter(|s| s.customer_ref == customer_ref)
            .cloned()
            .collect())
    }

    fn list_due_at(&self, as_of_unix_secs: u64) -> Result<Vec<Subscription>> {
        Ok(self
            .inner
            .lock()
            .expect("poisoned")
            .subs
            .values()
            .filter(|s| {
                !s.status.is_terminal() && s.current_period_end_unix_secs <= as_of_unix_secs
            })
            .cloned()
            .collect())
    }

    fn update<F>(&self, id: SubscriptionId, f: F) -> Result<Subscription>
    where
        F: FnOnce(&mut Subscription) -> Result<()>,
    {
        let mut inner = self.inner.lock().expect("poisoned");
        let existing = inner
            .subs
            .get(&id)
            .ok_or_else(|| Error::NotFound(id.to_string()))?;
        let mut staged = existing.clone();
        f(&mut staged)?;
        let result = staged.clone();
        inner.subs.insert(id, staged);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Interval, Plan};
    use op_core::{Currency, Money, PaymentMethod, VaultRef};

    fn plan() -> Plan {
        Plan::new(
            "p",
            Money::from_minor(1000, Currency::USD),
            Interval::Month,
            1,
        )
        .unwrap()
    }

    fn sub(customer: &str, ext: Option<&str>) -> Subscription {
        let mut s = Subscription::new(
            customer,
            plan(),
            PaymentMethod::Vault(VaultRef::new("tok")),
            1_700_000_000,
        )
        .unwrap();
        if let Some(e) = ext {
            s = s.with_external_id(e);
        }
        s
    }

    #[test]
    fn round_trip() {
        let store = InMemorySubscriptionStore::new();
        let s = sub("c-1", None);
        let id = s.id;
        store.create_subscription(s.clone()).unwrap();
        let got = store.get_subscription(id).unwrap();
        assert_eq!(got.id, id);
        assert_eq!(got.customer_ref, s.customer_ref);
        assert_eq!(got.plan.id, s.plan.id);
        assert_eq!(got.status.code(), s.status.code());
    }

    #[test]
    fn idempotency_same_body() {
        let store = InMemorySubscriptionStore::new();
        let s = sub("c-1", Some("e1"));
        let id1 = store.create_subscription(s.clone()).unwrap();
        let id2 = store.create_subscription(s).unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn idempotency_mismatch_rejects() {
        let store = InMemorySubscriptionStore::new();
        store.create_subscription(sub("c-1", Some("e1"))).unwrap();
        assert!(matches!(
            store
                .create_subscription(sub("c-2", Some("e1")))
                .unwrap_err(),
            Error::IdempotencyMismatch(_)
        ));
    }

    #[test]
    fn list_for_customer_filters() {
        let store = InMemorySubscriptionStore::new();
        store.create_subscription(sub("c-1", Some("a"))).unwrap();
        store.create_subscription(sub("c-1", Some("b"))).unwrap();
        store.create_subscription(sub("c-2", Some("c"))).unwrap();
        assert_eq!(store.list_for_customer("c-1").unwrap().len(), 2);
        assert_eq!(store.list_for_customer("c-2").unwrap().len(), 1);
        assert_eq!(store.list_for_customer("nope").unwrap().len(), 0);
    }

    #[test]
    fn list_due_at_filters_by_period_end() {
        let store = InMemorySubscriptionStore::new();
        let s = sub("c-1", None);
        let period_end = s.current_period_end_unix_secs;
        store.create_subscription(s).unwrap();
        assert!(store.list_due_at(period_end - 1).unwrap().is_empty());
        assert_eq!(store.list_due_at(period_end + 1).unwrap().len(), 1);
    }

    #[test]
    fn update_commits_on_ok_only() {
        let store = InMemorySubscriptionStore::new();
        let s = sub("c-1", None);
        let id = s.id;
        store.create_subscription(s).unwrap();
        store.update(id, |s| s.pause(2_000)).unwrap();
        assert_eq!(store.get_subscription(id).unwrap().status.code(), "paused");
        let _ = store
            .update(id, |s| {
                s.resume()?;
                Err(Error::Invalid("rollback".into()))
            })
            .unwrap_err();
        // Closure failed; resume() side effect was discarded.
        assert_eq!(store.get_subscription(id).unwrap().status.code(), "paused");
    }
}
