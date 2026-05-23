//! [`GraphIdempotencyStore`] — Minigraf-backed
//! [`IdempotencyStore`].
//!
//! Each idempotency key becomes an `idempotency_record` vertex
//! whose id is the deterministic UUIDv5 over the key string. That
//! gives us O(1)-equivalent lookup (no scan needed) and natural
//! collision handling: re-`reserve`-ing the same key always lands
//! on the same vertex.
//!
//! Properties:
//! - `key` — the raw idempotency key string.
//! - `body_signature` — for mismatch detection.
//! - `outcome` — JSON-encoded [`OrchestrationOutcome`], present only
//!   after commit.

use std::sync::Mutex;

use op_orchestrator::{IdempotencyKey, IdempotencyRecord, IdempotencyStore, OrchestrationOutcome};
use serde_json::Value as Json;
use uuid::Uuid;

use crate::graph::{GraphHandle, vtypes};

/// Graph-backed idempotency store. Persists key → record across
/// process restarts when the underlying handle is file-backed.
pub struct GraphIdempotencyStore {
    handle: GraphHandle,
    // Serialize concurrent reserve / commit so the
    // check-then-insert pattern stays atomic from the trait's
    // caller perspective. Minigraf transactions are themselves
    // serialized, but the *trait* contract is reserve-as-CAS.
    reserve_lock: Mutex<()>,
}

impl GraphIdempotencyStore {
    /// Construct on a fresh in-memory graph.
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_handle(GraphHandle::new_in_memory())
    }

    /// Construct on a shared handle.
    #[must_use]
    pub fn with_handle(handle: GraphHandle) -> Self {
        Self {
            handle,
            reserve_lock: Mutex::new(()),
        }
    }

    /// Borrow the underlying handle.
    #[must_use]
    pub fn handle(&self) -> &GraphHandle {
        &self.handle
    }

    /// Deterministic vertex id for a key. UUIDv5 over the key in
    /// the URL namespace — stable across processes / restarts.
    fn vertex_id(key: &IdempotencyKey) -> Uuid {
        Uuid::new_v5(&Uuid::NAMESPACE_URL, key.as_str().as_bytes())
    }

    fn load(&self, key: &IdempotencyKey) -> Option<IdempotencyRecord> {
        let id = Self::vertex_id(key);
        let exists = self.handle.vertex_exists(id).ok()?;
        if !exists {
            return None;
        }
        let props = self.handle.get_vertex_properties(id).ok()?;
        let body_signature = match props.get("body_signature") {
            Some(Json::String(s)) => s.clone(),
            _ => String::new(),
        };
        let outcome = props
            .get("outcome")
            .and_then(|v| serde_json::from_value::<OrchestrationOutcome>(v.clone()).ok());
        Some(IdempotencyRecord {
            body_signature,
            outcome,
        })
    }

    /// Diagnostic: number of records stored.
    pub fn len(&self) -> usize {
        self.handle
            .vertices_of_type(vtypes::IDEMPOTENCY_RECORD)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// True if no records have been stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl IdempotencyStore for GraphIdempotencyStore {
    fn reserve(&self, key: &IdempotencyKey, body_signature: &str) -> Option<IdempotencyRecord> {
        // CAS: under the reserve_lock, check-then-insert.
        let _guard = self.reserve_lock.lock().expect("idempotency lock poisoned");
        if let Some(existing) = self.load(key) {
            return Some(existing);
        }
        let id = Self::vertex_id(key);
        // Create vertex + set indexed properties. Failures
        // bubble up as a "not reserved" condition — the
        // upstream caller will see `None` and proceed,
        // worst-case admitting a duplicate execution on a graph
        // I/O fault.
        if self
            .handle
            .create_vertex(vtypes::IDEMPOTENCY_RECORD, id)
            .is_err()
        {
            return None;
        }
        let _ = self
            .handle
            .set_vertex_property(id, "key", Json::String(key.as_str().to_owned()));
        let _ = self.handle.set_vertex_property(
            id,
            "body_signature",
            Json::String(body_signature.to_owned()),
        );
        None
    }

    fn commit(&self, key: &IdempotencyKey, outcome: &OrchestrationOutcome) {
        let id = Self::vertex_id(key);
        // If the reservation isn't there yet, create the
        // vertex defensively — mirrors the in-memory store's
        // late-commit behavior.
        if !self.handle.vertex_exists(id).unwrap_or(false) {
            let _ = self.handle.create_vertex(vtypes::IDEMPOTENCY_RECORD, id);
            let _ =
                self.handle
                    .set_vertex_property(id, "key", Json::String(key.as_str().to_owned()));
            let _ =
                self.handle
                    .set_vertex_property(id, "body_signature", Json::String(String::new()));
        }
        if let Ok(v) = serde_json::to_value(outcome) {
            let _ = self.handle.set_vertex_property(id, "outcome", v);
        }
    }

    fn release(&self, key: &IdempotencyKey) {
        // Minigraf doesn't expose vertex delete (bi-temporal:
        // retracts move the "current" view but the history is
        // preserved). To mark a slot as released we clear its
        // body_signature so a fresh reserve sees an empty record
        // and treats it as in-flight-expired.
        //
        // This matches the production-store guidance in the trait
        // docs: "mark as expired-in-flight so analytics can spot
        // leaked reservations."
        let id = Self::vertex_id(key);
        if self.handle.vertex_exists(id).unwrap_or(false) {
            let _ = self.handle.set_vertex_property(
                id,
                "body_signature",
                Json::String("__released__".to_owned()),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::RailKind;
    use op_orchestrator::{AttemptOutcome, OrchestrationOutcome, TerminalStatus};

    fn fake_outcome() -> OrchestrationOutcome {
        OrchestrationOutcome {
            terminal_status: TerminalStatus::Approved,
            attempts: vec![op_orchestrator::Attempt {
                rail: RailKind::Card,
                driver: "test".into(),
                outcome: AttemptOutcome::Success,
            }],
            rail_used: Some(RailKind::Card),
            psp_payment_id: Some("psp_test".into()),
            uetr: None,
        }
    }

    #[test]
    fn reserve_then_load_returns_record() {
        let s = GraphIdempotencyStore::new_in_memory();
        let key = IdempotencyKey::new("k-1");
        assert!(s.reserve(&key, "sig-1").is_none());
        let rec = s.load(&key).unwrap();
        assert_eq!(rec.body_signature, "sig-1");
        assert!(rec.outcome.is_none());
    }

    #[test]
    fn reserve_twice_returns_existing() {
        let s = GraphIdempotencyStore::new_in_memory();
        let key = IdempotencyKey::new("k-2");
        let _ = s.reserve(&key, "sig");
        let second = s.reserve(&key, "sig").unwrap();
        assert_eq!(second.body_signature, "sig");
    }

    #[test]
    fn commit_writes_outcome() {
        let s = GraphIdempotencyStore::new_in_memory();
        let key = IdempotencyKey::new("k-3");
        s.reserve(&key, "sig");
        s.commit(&key, &fake_outcome());
        let rec = s.reserve(&key, "sig").unwrap();
        assert!(rec.outcome.is_some());
        assert_eq!(
            rec.outcome.unwrap().terminal_status,
            TerminalStatus::Approved
        );
    }

    #[test]
    fn release_marks_slot_released() {
        let s = GraphIdempotencyStore::new_in_memory();
        let key = IdempotencyKey::new("k-4");
        s.reserve(&key, "sig");
        s.release(&key);
        let rec = s.reserve(&key, "sig").unwrap();
        assert_eq!(rec.body_signature, "__released__");
    }

    #[test]
    fn vertex_id_is_deterministic() {
        let k = IdempotencyKey::new("stable-key");
        assert_eq!(
            GraphIdempotencyStore::vertex_id(&k),
            GraphIdempotencyStore::vertex_id(&k)
        );
    }
}
