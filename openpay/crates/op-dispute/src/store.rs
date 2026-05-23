//! Pluggable dispute storage. Same shape as `op_refund::store`.

use std::collections::HashMap;
use std::sync::Mutex;

use op_ledger::TransactionId;

use crate::dispute::{Dispute, DisputeId};
use crate::error::{Error, Result};

/// The pluggable storage interface for disputes.
pub trait DisputeStore: Send + Sync {
    /// Persist a freshly-created dispute. Idempotent on
    /// `external_id` (same body returns the existing id; different
    /// body returns [`Error::IdempotencyMismatch`]).
    ///
    /// # Errors
    /// See [`Error`].
    fn create_dispute(&self, dispute: Dispute) -> Result<DisputeId>;

    /// Fetch a dispute by id.
    ///
    /// # Errors
    /// [`Error::NotFound`] if unknown.
    fn get_dispute(&self, id: DisputeId) -> Result<Dispute>;

    /// Find by external id. `None` is success-with-no-match.
    ///
    /// # Errors
    /// Backend-specific.
    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Dispute>>;

    /// All disputes against the given ledger transaction.
    ///
    /// # Errors
    /// Backend-specific.
    fn list_for_tx(&self, tx_id: TransactionId) -> Result<Vec<Dispute>>;

    /// Apply a state transition / evidence attachment / status
    /// change via the closure. The store commits only if the
    /// closure returns `Ok(())`.
    ///
    /// # Errors
    /// `Error::NotFound`, plus whatever the closure returns.
    fn update<F>(&self, id: DisputeId, f: F) -> Result<Dispute>
    where
        F: FnOnce(&mut Dispute) -> Result<()>;
}

// ============================================================
// InMemoryDisputeStore
// ============================================================

#[derive(Default)]
struct Inner {
    disputes: HashMap<DisputeId, Dispute>,
    by_external_id: HashMap<String, DisputeId>,
}

/// In-process dispute store.
#[derive(Default)]
pub struct InMemoryDisputeStore {
    inner: Mutex<Inner>,
}

impl InMemoryDisputeStore {
    /// Construct empty.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of disputes tracked.
    ///
    /// # Panics
    /// Panics only if the internal mutex was previously poisoned by
    /// a thread that panicked while holding it — a condition that
    /// indicates broken state worth surfacing loudly anyway.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().expect("poisoned").disputes.len()
    }

    /// True when no disputes have been stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl DisputeStore for InMemoryDisputeStore {
    fn create_dispute(&self, dispute: Dispute) -> Result<DisputeId> {
        let mut inner = self.inner.lock().expect("poisoned");
        if let Some(ext) = &dispute.external_id {
            if let Some(existing_id) = inner.by_external_id.get(ext).copied() {
                let existing = inner
                    .disputes
                    .get(&existing_id)
                    .expect("index/store invariant")
                    .clone();
                if bodies_equivalent(&existing, &dispute) {
                    return Ok(existing_id);
                }
                return Err(Error::IdempotencyMismatch(ext.clone()));
            }
            inner.by_external_id.insert(ext.clone(), dispute.id);
        }
        let id = dispute.id;
        inner.disputes.insert(id, dispute);
        Ok(id)
    }

    fn get_dispute(&self, id: DisputeId) -> Result<Dispute> {
        self.inner
            .lock()
            .expect("poisoned")
            .disputes
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::NotFound(id.to_string()))
    }

    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Dispute>> {
        let inner = self.inner.lock().expect("poisoned");
        let Some(id) = inner.by_external_id.get(external_id).copied() else {
            return Ok(None);
        };
        Ok(inner.disputes.get(&id).cloned())
    }

    fn list_for_tx(&self, tx_id: TransactionId) -> Result<Vec<Dispute>> {
        Ok(self
            .inner
            .lock()
            .expect("poisoned")
            .disputes
            .values()
            .filter(|d| d.original_tx_id == tx_id)
            .cloned()
            .collect())
    }

    fn update<F>(&self, id: DisputeId, f: F) -> Result<Dispute>
    where
        F: FnOnce(&mut Dispute) -> Result<()>,
    {
        let mut inner = self.inner.lock().expect("poisoned");
        let existing = inner
            .disputes
            .get(&id)
            .ok_or_else(|| Error::NotFound(id.to_string()))?;
        let mut staged = existing.clone();
        f(&mut staged)?;
        let result = staged.clone();
        inner.disputes.insert(id, staged);
        Ok(result)
    }
}

fn bodies_equivalent(a: &Dispute, b: &Dispute) -> bool {
    a.original_tx_id == b.original_tx_id
        && a.amount == b.amount
        && a.reason == b.reason
        && a.network_reason_code == b.network_reason_code
        && a.external_id == b.external_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispute::Status;
    use crate::evidence::EvidenceRef;
    use crate::reason::DisputeReason;
    use op_core::{Currency, Money};
    use op_ledger::TransactionId;

    fn sample(ext: Option<&str>) -> Dispute {
        let mut d = Dispute::new(
            TransactionId::new(),
            Money::from_minor(2_500, Currency::USD),
            DisputeReason::Fraudulent,
            1_000,
        )
        .unwrap();
        if let Some(e) = ext {
            d = d.with_external_id(e);
        }
        d
    }

    #[test]
    fn create_and_get_round_trip() {
        let store = InMemoryDisputeStore::new();
        let d = sample(None);
        let id = d.id;
        store.create_dispute(d.clone()).unwrap();
        assert_eq!(store.get_dispute(id).unwrap(), d);
    }

    #[test]
    fn idempotency_returns_same_on_match() {
        let store = InMemoryDisputeStore::new();
        let d = sample(Some("disp-1"));
        let id1 = store.create_dispute(d.clone()).unwrap();
        let id2 = store.create_dispute(d).unwrap();
        assert_eq!(id1, id2);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn idempotency_rejects_on_body_mismatch() {
        let store = InMemoryDisputeStore::new();
        let d1 = sample(Some("disp-1"));
        store.create_dispute(d1).unwrap();
        let mut d2 = sample(Some("disp-1"));
        d2.amount = Money::from_minor(9999, Currency::USD);
        assert!(matches!(
            store.create_dispute(d2),
            Err(Error::IdempotencyMismatch(_))
        ));
    }

    #[test]
    fn update_state_transition_persists() {
        let store = InMemoryDisputeStore::new();
        let d = sample(None);
        let id = d.id;
        store.create_dispute(d).unwrap();
        store
            .update(id, |d| {
                d.attach_evidence(EvidenceRef::new("receipt", "s3://b/r", 1_500))?;
                d.represent()
            })
            .unwrap();
        let after = store.get_dispute(id).unwrap();
        assert!(matches!(after.status, Status::Representment));
        assert_eq!(after.evidence.len(), 1);
    }

    #[test]
    fn update_rollback_on_error() {
        let store = InMemoryDisputeStore::new();
        let d = sample(None);
        let id = d.id;
        store.create_dispute(d).unwrap();
        let res: Result<Dispute> = store.update(id, |d| {
            d.represent()?;
            Err(Error::Invalid("simulated".into()))
        });
        assert!(res.is_err());
        // State should NOT have changed.
        assert!(matches!(
            store.get_dispute(id).unwrap().status,
            Status::Chargeback
        ));
    }

    #[test]
    fn list_for_tx_filters() {
        let store = InMemoryDisputeStore::new();
        let tx = TransactionId::new();
        for i in 0..2 {
            let mut d = sample(Some(&format!("disp-{i}")));
            d.original_tx_id = tx;
            store.create_dispute(d).unwrap();
        }
        store.create_dispute(sample(Some("orphan"))).unwrap();
        assert_eq!(store.list_for_tx(tx).unwrap().len(), 2);
    }
}
