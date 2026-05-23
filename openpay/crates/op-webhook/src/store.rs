//! Pluggable storage for webhook state.
//!
//! Mirrors the pattern used in `op-ledger` and `op-orchestrator`:
//! a trait surface that admits Postgres / Redis / etc. backends,
//! with an in-memory reference impl for tests.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::endpoint::{Endpoint, EndpointId, EndpointStatus};
use crate::error::{Error, Result};
use crate::event::{
    DeliveryAttempt, DeliveryAttemptId, DeliveryStatus, WebhookEvent, WebhookEventId,
};

/// Pluggable storage trait. Sync.
pub trait WebhookStore: Send + Sync {
    /// Register or update an endpoint.
    fn put_endpoint(&self, endpoint: Endpoint) -> Result<EndpointId>;

    /// Look up an endpoint.
    fn get_endpoint(&self, id: EndpointId) -> Result<Endpoint>;

    /// List all endpoints that subscribe to the given event type.
    /// Filters to `Active` endpoints only.
    fn list_active_endpoints_for(&self, event_type: &str) -> Result<Vec<Endpoint>>;

    /// Update endpoint status (used on auto-disable and operator
    /// re-enable flows).
    fn set_endpoint_status(&self, id: EndpointId, status: EndpointStatus) -> Result<()>;

    /// Update an endpoint's `consecutive_failures` counter.
    fn set_endpoint_consecutive_failures(&self, id: EndpointId, failures: u32) -> Result<()>;

    /// Save an event for replay and audit.
    fn put_event(&self, event: WebhookEvent) -> Result<WebhookEventId>;

    /// Look up an event.
    fn get_event(&self, id: WebhookEventId) -> Result<WebhookEvent>;

    /// Save a delivery attempt.
    fn put_attempt(&self, attempt: DeliveryAttempt) -> Result<DeliveryAttemptId>;

    /// Look up a delivery attempt.
    fn get_attempt(&self, id: DeliveryAttemptId) -> Result<DeliveryAttempt>;

    /// List attempts for an (event, endpoint) pair in attempt-order.
    fn list_attempts(
        &self,
        event_id: WebhookEventId,
        endpoint_id: EndpointId,
    ) -> Result<Vec<DeliveryAttempt>>;

    /// List attempts in [`DeliveryStatus::RetryScheduled`] whose
    /// `next_attempt_at_unix_secs <= now`. Used by the
    /// dispatcher's `process_due_retries` worker.
    fn list_due_retries(&self, now_unix_secs: u64) -> Result<Vec<DeliveryAttempt>>;
}

// ============================================================
// InMemoryWebhookStore
// ============================================================

#[derive(Default)]
struct Inner {
    endpoints: HashMap<EndpointId, Endpoint>,
    events: HashMap<WebhookEventId, WebhookEvent>,
    attempts: HashMap<DeliveryAttemptId, DeliveryAttempt>,
}

/// In-process store. NOT for multi-instance production.
#[derive(Default)]
pub struct InMemoryWebhookStore {
    inner: Mutex<Inner>,
}

impl InMemoryWebhookStore {
    /// Construct empty.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Diagnostic: number of endpoints.
    #[must_use]
    pub fn endpoint_count(&self) -> usize {
        self.inner.lock().expect("poisoned").endpoints.len()
    }

    /// Diagnostic: number of events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.inner.lock().expect("poisoned").events.len()
    }

    /// Diagnostic: number of delivery attempts.
    #[must_use]
    pub fn attempt_count(&self) -> usize {
        self.inner.lock().expect("poisoned").attempts.len()
    }
}

impl WebhookStore for InMemoryWebhookStore {
    fn put_endpoint(&self, endpoint: Endpoint) -> Result<EndpointId> {
        let mut g = self.inner.lock().expect("poisoned");
        let id = endpoint.id;
        g.endpoints.insert(id, endpoint);
        Ok(id)
    }

    fn get_endpoint(&self, id: EndpointId) -> Result<Endpoint> {
        let g = self.inner.lock().expect("poisoned");
        g.endpoints
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::EndpointNotFound(id.to_string()))
    }

    fn list_active_endpoints_for(&self, event_type: &str) -> Result<Vec<Endpoint>> {
        let g = self.inner.lock().expect("poisoned");
        Ok(g.endpoints
            .values()
            .filter(|e| e.status == EndpointStatus::Active && e.matches(event_type))
            .cloned()
            .collect())
    }

    fn set_endpoint_status(&self, id: EndpointId, status: EndpointStatus) -> Result<()> {
        let mut g = self.inner.lock().expect("poisoned");
        let e = g
            .endpoints
            .get_mut(&id)
            .ok_or_else(|| Error::EndpointNotFound(id.to_string()))?;
        e.status = status;
        Ok(())
    }

    fn set_endpoint_consecutive_failures(&self, id: EndpointId, failures: u32) -> Result<()> {
        let mut g = self.inner.lock().expect("poisoned");
        let e = g
            .endpoints
            .get_mut(&id)
            .ok_or_else(|| Error::EndpointNotFound(id.to_string()))?;
        e.consecutive_failures = failures;
        Ok(())
    }

    fn put_event(&self, event: WebhookEvent) -> Result<WebhookEventId> {
        let mut g = self.inner.lock().expect("poisoned");
        let id = event.id;
        g.events.insert(id, event);
        Ok(id)
    }

    fn get_event(&self, id: WebhookEventId) -> Result<WebhookEvent> {
        let g = self.inner.lock().expect("poisoned");
        g.events
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::EventNotFound(id.to_string()))
    }

    fn put_attempt(&self, attempt: DeliveryAttempt) -> Result<DeliveryAttemptId> {
        let mut g = self.inner.lock().expect("poisoned");
        let id = attempt.id;
        g.attempts.insert(id, attempt);
        Ok(id)
    }

    fn get_attempt(&self, id: DeliveryAttemptId) -> Result<DeliveryAttempt> {
        let g = self.inner.lock().expect("poisoned");
        g.attempts
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::AttemptNotFound(id.to_string()))
    }

    fn list_attempts(
        &self,
        event_id: WebhookEventId,
        endpoint_id: EndpointId,
    ) -> Result<Vec<DeliveryAttempt>> {
        let g = self.inner.lock().expect("poisoned");
        let mut v: Vec<DeliveryAttempt> = g
            .attempts
            .values()
            .filter(|a| a.event_id == event_id && a.endpoint_id == endpoint_id)
            .cloned()
            .collect();
        v.sort_by_key(|a| a.attempt_number);
        Ok(v)
    }

    fn list_due_retries(&self, now_unix_secs: u64) -> Result<Vec<DeliveryAttempt>> {
        let g = self.inner.lock().expect("poisoned");
        Ok(g.attempts
            .values()
            .filter(|a| {
                a.status == DeliveryStatus::RetryScheduled
                    && a.next_attempt_at_unix_secs
                        .map(|t| t <= now_unix_secs)
                        .unwrap_or(false)
            })
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint() -> Endpoint {
        Endpoint::new(
            "https://merchant.example.com/h",
            b"secret".to_vec(),
            vec!["*".to_string(), "ledger.txn.posted".to_string()],
        )
        .unwrap()
    }

    #[test]
    fn put_and_get_endpoint() {
        let store = InMemoryWebhookStore::new();
        let e = endpoint();
        let id = e.id;
        store.put_endpoint(e.clone()).unwrap();
        let recovered = store.get_endpoint(id).unwrap();
        assert_eq!(recovered, e);
    }

    #[test]
    fn get_unknown_endpoint_errors() {
        let store = InMemoryWebhookStore::new();
        let r = store.get_endpoint(EndpointId::new());
        assert!(matches!(r, Err(Error::EndpointNotFound(_))));
    }

    #[test]
    fn list_active_filters_by_status_and_match() {
        let store = InMemoryWebhookStore::new();

        // Endpoint #1: active, subscribes to ledger.txn.posted only.
        let e1 = Endpoint::new(
            "https://x.example",
            b"s".to_vec(),
            vec!["ledger.txn.posted".to_string()],
        )
        .unwrap();
        let e1_id = e1.id;
        store.put_endpoint(e1).unwrap();

        // Endpoint #2: wildcard.
        let e2 = Endpoint::new("https://y.example", b"s".to_vec(), vec!["*".to_string()]).unwrap();
        let e2_id = e2.id;
        store.put_endpoint(e2).unwrap();

        // Endpoint #3: wildcard but disabled.
        let mut e3 =
            Endpoint::new("https://z.example", b"s".to_vec(), vec!["*".to_string()]).unwrap();
        e3.status = EndpointStatus::Disabled;
        let e3_id = e3.id;
        store.put_endpoint(e3).unwrap();

        // For "ledger.txn.posted": e1 + e2 match; e3 disabled.
        let matched = store
            .list_active_endpoints_for("ledger.txn.posted")
            .unwrap();
        let ids: Vec<_> = matched.iter().map(|e| e.id).collect();
        assert!(ids.contains(&e1_id));
        assert!(ids.contains(&e2_id));
        assert!(!ids.contains(&e3_id));

        // For "anything_else": only e2 (wildcard).
        let matched = store.list_active_endpoints_for("anything_else").unwrap();
        let ids: Vec<_> = matched.iter().map(|e| e.id).collect();
        assert!(!ids.contains(&e1_id));
        assert!(ids.contains(&e2_id));
    }

    #[test]
    fn set_endpoint_status() {
        let store = InMemoryWebhookStore::new();
        let e = endpoint();
        let id = e.id;
        store.put_endpoint(e).unwrap();
        store
            .set_endpoint_status(id, EndpointStatus::AutoDisabled)
            .unwrap();
        assert_eq!(
            store.get_endpoint(id).unwrap().status,
            EndpointStatus::AutoDisabled
        );
    }

    #[test]
    fn set_endpoint_status_unknown_id_errors() {
        let store = InMemoryWebhookStore::new();
        let r = store.set_endpoint_status(EndpointId::new(), EndpointStatus::Disabled);
        assert!(matches!(r, Err(Error::EndpointNotFound(_))));
    }

    #[test]
    fn set_endpoint_consecutive_failures() {
        let store = InMemoryWebhookStore::new();
        let e = endpoint();
        let id = e.id;
        store.put_endpoint(e).unwrap();
        store.set_endpoint_consecutive_failures(id, 5).unwrap();
        assert_eq!(store.get_endpoint(id).unwrap().consecutive_failures, 5);
    }

    #[test]
    fn put_and_get_event() {
        let store = InMemoryWebhookStore::new();
        let e = WebhookEvent::new("payment.authorized", b"{}".to_vec(), 0);
        let id = e.id;
        store.put_event(e.clone()).unwrap();
        assert_eq!(store.get_event(id).unwrap(), e);
    }

    #[test]
    fn list_attempts_sorted_by_number() {
        let store = InMemoryWebhookStore::new();
        let eid = WebhookEventId::new();
        let epid = EndpointId::new();
        // Insert attempts out of order.
        let a2 = DeliveryAttempt::new_pending(eid, epid, 2, 200);
        let a0 = DeliveryAttempt::new_pending(eid, epid, 0, 0);
        let a1 = DeliveryAttempt::new_pending(eid, epid, 1, 100);
        store.put_attempt(a2).unwrap();
        store.put_attempt(a0).unwrap();
        store.put_attempt(a1).unwrap();
        let list = store.list_attempts(eid, epid).unwrap();
        let numbers: Vec<u32> = list.iter().map(|a| a.attempt_number).collect();
        assert_eq!(numbers, vec![0, 1, 2]);
    }

    #[test]
    fn list_due_retries_filters_by_status_and_time() {
        let store = InMemoryWebhookStore::new();
        let eid = WebhookEventId::new();
        let epid = EndpointId::new();
        // Due now.
        let mut a1 = DeliveryAttempt::new_pending(eid, epid, 0, 0);
        a1.status = DeliveryStatus::RetryScheduled;
        a1.next_attempt_at_unix_secs = Some(100);
        // Due in the future.
        let mut a2 = DeliveryAttempt::new_pending(eid, epid, 1, 0);
        a2.status = DeliveryStatus::RetryScheduled;
        a2.next_attempt_at_unix_secs = Some(500);
        // Already failed (terminal) — not due even if scheduled time
        // passed.
        let mut a3 = DeliveryAttempt::new_pending(eid, epid, 2, 0);
        a3.status = DeliveryStatus::Failed;
        a3.next_attempt_at_unix_secs = Some(50);
        let a1_id = a1.id;
        store.put_attempt(a1).unwrap();
        store.put_attempt(a2).unwrap();
        store.put_attempt(a3).unwrap();

        let due = store.list_due_retries(200).unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].id, a1_id);
    }

    #[test]
    fn diagnostic_counters() {
        let store = InMemoryWebhookStore::new();
        store.put_endpoint(endpoint()).unwrap();
        let e = WebhookEvent::new("t", vec![], 0);
        let eid = e.id;
        store.put_event(e).unwrap();
        store
            .put_attempt(DeliveryAttempt::new_pending(eid, EndpointId::new(), 0, 0))
            .unwrap();
        assert_eq!(store.endpoint_count(), 1);
        assert_eq!(store.event_count(), 1);
        assert_eq!(store.attempt_count(), 1);
    }
}
