//! Pluggable settlement storage.
//!
//! Same shape as the ledger / webhook / refund / dispute stores: a
//! sync trait + an in-memory ref impl. Production deployments plug
//! in their own backend.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::batch::{Batch, BatchId};
use crate::error::{Error, Result};

/// Pluggable storage interface for settlement batches.
pub trait SettlementStore: Send + Sync {
    /// Persist a freshly-opened batch. Idempotent on `external_id`.
    ///
    /// # Errors
    /// See [`Error`].
    fn create_batch(&self, batch: Batch) -> Result<BatchId>;

    /// Fetch a batch by id.
    ///
    /// # Errors
    /// [`Error::NotFound`] if unknown.
    fn get_batch(&self, id: BatchId) -> Result<Batch>;

    /// Find by external id.
    ///
    /// # Errors
    /// Backend-specific.
    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Batch>>;

    /// All currently-open batches. Order is unspecified.
    ///
    /// # Errors
    /// Backend-specific.
    fn list_open(&self) -> Result<Vec<Batch>>;

    /// Apply a state transition through a closure. The store
    /// commits only on `Ok(())`.
    ///
    /// # Errors
    /// `Error::NotFound`, plus whatever the closure returns.
    fn update<F>(&self, id: BatchId, f: F) -> Result<Batch>
    where
        F: FnOnce(&mut Batch) -> Result<()>;
}

// ============================================================
// InMemorySettlementStore
// ============================================================

#[derive(Default)]
struct Inner {
    batches: HashMap<BatchId, Batch>,
    by_external_id: HashMap<String, BatchId>,
}

/// In-process settlement store. Not for multi-instance production.
#[derive(Default)]
pub struct InMemorySettlementStore {
    inner: Mutex<Inner>,
}

impl InMemorySettlementStore {
    /// Construct empty.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of batches tracked.
    ///
    /// # Panics
    /// Panics only if the internal mutex was previously poisoned by
    /// a thread that panicked while holding it.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().expect("poisoned").batches.len()
    }

    /// True when no batches have been stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl SettlementStore for InMemorySettlementStore {
    fn create_batch(&self, batch: Batch) -> Result<BatchId> {
        let mut inner = self.inner.lock().expect("poisoned");
        if let Some(ext) = &batch.external_id {
            if let Some(existing_id) = inner.by_external_id.get(ext).copied() {
                let existing = inner
                    .batches
                    .get(&existing_id)
                    .expect("index/store invariant: id in index must be in store")
                    .clone();
                if bodies_equivalent(&existing, &batch) {
                    return Ok(existing_id);
                }
                return Err(Error::IdempotencyMismatch(ext.clone()));
            }
            inner.by_external_id.insert(ext.clone(), batch.id);
        }
        let id = batch.id;
        inner.batches.insert(id, batch);
        Ok(id)
    }

    fn get_batch(&self, id: BatchId) -> Result<Batch> {
        self.inner
            .lock()
            .expect("poisoned")
            .batches
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::NotFound(id.to_string()))
    }

    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Batch>> {
        let inner = self.inner.lock().expect("poisoned");
        let Some(id) = inner.by_external_id.get(external_id).copied() else {
            return Ok(None);
        };
        Ok(inner.batches.get(&id).cloned())
    }

    fn list_open(&self) -> Result<Vec<Batch>> {
        Ok(self
            .inner
            .lock()
            .expect("poisoned")
            .batches
            .values()
            .filter(|b| matches!(b.status, crate::batch::Status::Open))
            .cloned()
            .collect())
    }

    fn update<F>(&self, id: BatchId, f: F) -> Result<Batch>
    where
        F: FnOnce(&mut Batch) -> Result<()>,
    {
        let mut inner = self.inner.lock().expect("poisoned");
        let existing = inner
            .batches
            .get(&id)
            .ok_or_else(|| Error::NotFound(id.to_string()))?;
        let mut staged = existing.clone();
        f(&mut staged)?;
        let result = staged.clone();
        inner.batches.insert(id, staged);
        Ok(result)
    }
}

/// Body-equivalence for batch idempotency: currency + rail +
/// `external_id` + initial entry list must match exactly. Status /
/// holdback / metadata may differ (the live batch advances).
fn bodies_equivalent(a: &Batch, b: &Batch) -> bool {
    a.currency == b.currency
        && a.rail == b.rail
        && a.external_id == b.external_id
        && a.entries == b.entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::Batch;
    use crate::payout::PayoutRail;
    use op_core::{Currency, Money};

    #[test]
    fn create_and_get_round_trip() {
        let store = InMemorySettlementStore::new();
        let b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000);
        let bid = b.id;
        store.create_batch(b.clone()).unwrap();
        assert_eq!(store.get_batch(bid).unwrap(), b);
    }

    #[test]
    fn idempotency_same_body_returns_same_id() {
        let store = InMemorySettlementStore::new();
        let b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000).with_external_id("pay-1");
        let id1 = store.create_batch(b.clone()).unwrap();
        let id2 = store.create_batch(b).unwrap();
        assert_eq!(id1, id2);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn idempotency_different_body_rejects() {
        let store = InMemorySettlementStore::new();
        let b1 = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000).with_external_id("pay-1");
        store.create_batch(b1).unwrap();
        let b2 = Batch::open(Currency::USD, PayoutRail::SepaCt, 1_000).with_external_id("pay-1");
        let err = store.create_batch(b2).unwrap_err();
        assert!(matches!(err, Error::IdempotencyMismatch(_)));
    }

    #[test]
    fn list_open_excludes_closed() {
        let store = InMemorySettlementStore::new();
        let b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000);
        let bid = b.id;
        store.create_batch(b.clone()).unwrap();
        // Close it via update.
        store
            .update(bid, |bb| {
                bb.add_entry(
                    op_ledger::TransactionId::new(),
                    Money::from_minor(100, Currency::USD),
                    None,
                )?;
                let gross = bb.gross()?;
                let hb = crate::holdback::HoldbackPolicy::none().compute(gross, 0)?;
                bb.close(hb, 2_000)?;
                let _ = &b;
                Ok(())
            })
            .unwrap();
        assert!(store.list_open().unwrap().is_empty());
    }
}
