//! Idempotency: prevent duplicate execution under retry.
//!
//! Industry consensus (Stripe / Adyen / Carbon Canyon postmortem):
//!
//! - Clients generate idempotency keys (UUID v4) per logical request.
//! - Servers store key → response payload with a TTL of 24h-7d.
//! - Duplicate keys return the cached response without re-executing.
//! - Mismatched body for the same key → 409 / `IdempotencyMismatch`.
//! - Concurrent requests with the same key → only one runs; others
//!   wait or get a transient-error retry signal.
//!
//! This module ships an in-process [`InMemoryIdempotencyStore`]
//! suitable for tests and short-lived processes (e.g. an
//! unattended-checkout kiosk on a single Linux box). Production
//! deployments plug in their own [`IdempotencyStore`] backed by
//! Redis / Postgres / Spanner.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::outcome::OrchestrationOutcome;

/// Caller-supplied unique key for a logical request.
///
/// Clone-cheap (just a wrapped String). UUID v4 is the canonical
/// choice but any unique-per-request string works.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct IdempotencyKey(String);

impl IdempotencyKey {
    /// Construct from any string.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<T: Into<String>> From<T> for IdempotencyKey {
    fn from(s: T) -> Self {
        Self::new(s)
    }
}

/// A stored idempotency record.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IdempotencyRecord {
    /// The body signature that must match on duplicate-key requests.
    /// See [`PaymentIntent::body_signature`](crate::PaymentIntent::body_signature).
    pub body_signature: String,

    /// The cached terminal outcome. `None` if the record is reserved
    /// but the outcome hasn't been written yet (in-flight slot).
    pub outcome: Option<OrchestrationOutcome>,
}

/// Pluggable idempotency backend.
///
/// All three methods must be **atomic** with respect to concurrent
/// callers, or duplicate execution becomes possible. In particular:
///
/// - [`Self::reserve`] must `CAS`-style insert iff the key is absent.
/// - [`Self::commit`] must overwrite the reservation atomically.
///
/// The trait is sync — callers wrap it in `tokio::task::spawn_blocking`
/// or similar if they need to run in an async context.
pub trait IdempotencyStore: Send + Sync {
    /// Atomically reserve a slot for `key` with the given body
    /// signature. Returns:
    ///
    /// - `Ok(None)` if the slot was newly reserved (caller proceeds).
    /// - `Ok(Some(record))` if the slot already existed; the caller
    ///   must check `record.body_signature` for match and return
    ///   `record.outcome` if present.
    ///
    /// The caller must subsequently call [`Self::commit`] with the
    /// terminal outcome, OR call [`Self::release`] if the orchestrator
    /// crashed mid-flight (so the next retry doesn't see a stuck
    /// "in-flight" record).
    fn reserve(&self, key: &IdempotencyKey, body_signature: &str) -> Option<IdempotencyRecord>;

    /// Atomically write the terminal outcome for `key`. Overwrites
    /// any in-flight reservation.
    fn commit(&self, key: &IdempotencyKey, outcome: &OrchestrationOutcome);

    /// Release an in-flight reservation (orchestrator crashed before
    /// reaching a terminal state). The slot is removed; a subsequent
    /// retry can reserve a fresh one.
    ///
    /// Production stores typically don't remove the slot but instead
    /// mark it as "expired-in-flight" so analytics can spot leaked
    /// reservations.
    fn release(&self, key: &IdempotencyKey);
}

/// In-process store. NOT for multi-instance production.
///
/// Uses a `Mutex<HashMap>` for atomic compare-and-swap semantics.
/// All operations are O(1).
#[derive(Default)]
pub struct InMemoryIdempotencyStore {
    inner: Mutex<HashMap<IdempotencyKey, IdempotencyRecord>>,
}

impl InMemoryIdempotencyStore {
    /// Construct an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of records (for diagnostics).
    pub fn len(&self) -> usize {
        self.inner.lock().expect("idempotency store poisoned").len()
    }

    /// Is the store empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl IdempotencyStore for InMemoryIdempotencyStore {
    fn reserve(&self, key: &IdempotencyKey, body_signature: &str) -> Option<IdempotencyRecord> {
        let mut map = self.inner.lock().expect("idempotency store poisoned");
        if let Some(existing) = map.get(key) {
            return Some(existing.clone());
        }
        map.insert(
            key.clone(),
            IdempotencyRecord {
                body_signature: body_signature.to_owned(),
                outcome: None,
            },
        );
        None
    }

    fn commit(&self, key: &IdempotencyKey, outcome: &OrchestrationOutcome) {
        let mut map = self.inner.lock().expect("idempotency store poisoned");
        if let Some(rec) = map.get_mut(key) {
            rec.outcome = Some(outcome.clone());
        } else {
            // Commit without prior reservation — shouldn't happen in
            // normal flow, but be defensive. Insert a fresh record
            // with empty signature; later `reserve` calls will
            // collide as "in-flight done".
            map.insert(
                key.clone(),
                IdempotencyRecord {
                    body_signature: String::new(),
                    outcome: Some(outcome.clone()),
                },
            );
        }
    }

    fn release(&self, key: &IdempotencyKey) {
        let mut map = self.inner.lock().expect("idempotency store poisoned");
        // Only release if the record is in-flight (outcome=None).
        // Removing a committed outcome would be a correctness bug.
        if let Some(rec) = map.get(key)
            && rec.outcome.is_none()
        {
            map.remove(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::{OrchestrationOutcome, TerminalStatus};

    fn fake_outcome() -> OrchestrationOutcome {
        OrchestrationOutcome {
            terminal_status: TerminalStatus::Approved,
            attempts: Vec::new(),
            rail_used: None,
            psp_payment_id: Some("psp_test_1".into()),
            uetr: None,
        }
    }

    #[test]
    fn reserve_returns_none_on_first_call() {
        let s = InMemoryIdempotencyStore::new();
        let r = s.reserve(&IdempotencyKey::new("k1"), "sig-a");
        assert!(r.is_none());
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn reserve_returns_existing_on_duplicate() {
        let s = InMemoryIdempotencyStore::new();
        s.reserve(&IdempotencyKey::new("k1"), "sig-a");
        let r2 = s.reserve(&IdempotencyKey::new("k1"), "sig-a");
        assert!(r2.is_some());
        assert!(r2.unwrap().outcome.is_none()); // still in-flight
    }

    #[test]
    fn commit_then_reserve_returns_cached_outcome() {
        let s = InMemoryIdempotencyStore::new();
        let k = IdempotencyKey::new("k1");
        s.reserve(&k, "sig-a");
        s.commit(&k, &fake_outcome());
        let r = s.reserve(&k, "sig-a");
        assert!(r.is_some());
        let rec = r.unwrap();
        assert!(rec.outcome.is_some());
        assert_eq!(
            rec.outcome.unwrap().terminal_status,
            TerminalStatus::Approved
        );
    }

    #[test]
    fn body_signature_round_trips_via_reservation() {
        let s = InMemoryIdempotencyStore::new();
        let k = IdempotencyKey::new("k1");
        s.reserve(&k, "sig-a");
        let r = s.reserve(&k, "sig-DIFFERENT");
        assert_eq!(r.unwrap().body_signature, "sig-a");
        // Caller is the one who detects the mismatch; the store just
        // surfaces the original signature.
    }

    #[test]
    fn release_removes_inflight_record() {
        let s = InMemoryIdempotencyStore::new();
        let k = IdempotencyKey::new("k1");
        s.reserve(&k, "sig-a");
        assert_eq!(s.len(), 1);
        s.release(&k);
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn release_preserves_committed_record() {
        // Critical: release MUST NOT remove a committed outcome.
        // Otherwise a slow retry would re-execute the payment.
        let s = InMemoryIdempotencyStore::new();
        let k = IdempotencyKey::new("k1");
        s.reserve(&k, "sig-a");
        s.commit(&k, &fake_outcome());
        s.release(&k);
        assert_eq!(s.len(), 1, "committed record must NOT be released");
        let r = s.reserve(&k, "sig-a");
        assert!(r.unwrap().outcome.is_some());
    }

    #[test]
    fn commit_without_reserve_still_works() {
        // Defensive — shouldn't happen in normal flow but must not
        // panic.
        let s = InMemoryIdempotencyStore::new();
        let k = IdempotencyKey::new("k1");
        s.commit(&k, &fake_outcome());
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn idempotency_key_from_str_and_string() {
        let _k1 = IdempotencyKey::from("foo");
        let _k2 = IdempotencyKey::from(String::from("bar"));
        let _k3 = IdempotencyKey::new("baz");
    }

    #[test]
    fn keys_are_distinct_by_string() {
        let a = IdempotencyKey::new("a");
        let b = IdempotencyKey::new("b");
        let s = InMemoryIdempotencyStore::new();
        assert!(s.reserve(&a, "x").is_none());
        assert!(s.reserve(&b, "y").is_none());
        assert_eq!(s.len(), 2);
    }
}
