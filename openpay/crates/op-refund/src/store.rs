//! Pluggable refund storage.
//!
//! Same shape as the ledger and webhook stores: a sync trait and an
//! [`InMemoryRefundStore`] reference impl. Production deployments
//! plug in a Postgres / graph / `TigerBeetle` backend.

use std::collections::HashMap;
use std::sync::Mutex;

use op_ledger::TransactionId;

use crate::error::{Error, Result};
use crate::refund::{Refund, RefundId};

/// The pluggable storage interface for refunds.
pub trait RefundStore: Send + Sync {
    /// Persist a freshly-created refund. Returns its id.
    ///
    /// **Idempotent on `external_id`.** A second call with the same
    /// `external_id` and the same body returns the existing id; a
    /// matching id but mismatched body returns
    /// [`Error::IdempotencyMismatch`].
    ///
    /// # Errors
    /// See [`Error`].
    fn create_refund(&self, refund: Refund) -> Result<RefundId>;

    /// Fetch a refund by id.
    ///
    /// # Errors
    /// [`Error::NotFound`] if unknown.
    fn get_refund(&self, id: RefundId) -> Result<Refund>;

    /// Find a refund by its operator-supplied external id.
    ///
    /// # Errors
    /// Backend-specific. `None` is success-with-no-match.
    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Refund>>;

    /// All refunds against the given original ledger transaction.
    /// Order is unspecified; callers typically sort by
    /// `requested_at_unix_secs` themselves.
    ///
    /// # Errors
    /// Backend-specific.
    fn list_for_tx(&self, tx_id: TransactionId) -> Result<Vec<Refund>>;

    /// Apply a state transition. The closure receives the current
    /// refund and either mutates it via [`Refund::submit`] /
    /// `approve` / `settle` / `decline` / `fail_after_approval` and
    /// returns `Ok(())`, or returns an `Err` to abort the
    /// transition.
    ///
    /// The store commits the mutation only if the closure returns
    /// `Ok(())`. This is the trait's atomicity contract: a partial
    /// mutation must never be persisted.
    ///
    /// # Errors
    /// `Error::NotFound`, plus whatever the closure returns.
    fn update<F>(&self, id: RefundId, f: F) -> Result<Refund>
    where
        F: FnOnce(&mut Refund) -> Result<()>;
}

// ============================================================
// InMemoryRefundStore
// ============================================================

#[derive(Default)]
struct Inner {
    refunds: HashMap<RefundId, Refund>,
    by_external_id: HashMap<String, RefundId>,
}

/// In-process refund store. Not for multi-instance production.
#[derive(Default)]
pub struct InMemoryRefundStore {
    inner: Mutex<Inner>,
}

impl InMemoryRefundStore {
    /// Construct empty.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of refunds tracked.
    ///
    /// # Panics
    /// Panics only if the internal mutex was previously poisoned by
    /// a thread that panicked while holding it — a condition that
    /// indicates broken state worth surfacing loudly anyway.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().expect("poisoned").refunds.len()
    }

    /// True when no refunds have been stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl RefundStore for InMemoryRefundStore {
    fn create_refund(&self, refund: Refund) -> Result<RefundId> {
        let mut inner = self.inner.lock().expect("poisoned");
        if let Some(ext) = &refund.external_id {
            if let Some(existing_id) = inner.by_external_id.get(ext).copied() {
                let existing = inner
                    .refunds
                    .get(&existing_id)
                    .expect("index/store invariant: id in index must be in store")
                    .clone();
                if bodies_equivalent(&existing, &refund) {
                    return Ok(existing_id);
                }
                return Err(Error::IdempotencyMismatch(ext.clone()));
            }
            inner.by_external_id.insert(ext.clone(), refund.id);
        }
        let id = refund.id;
        inner.refunds.insert(id, refund);
        Ok(id)
    }

    fn get_refund(&self, id: RefundId) -> Result<Refund> {
        self.inner
            .lock()
            .expect("poisoned")
            .refunds
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::NotFound(id.to_string()))
    }

    fn find_by_external_id(&self, external_id: &str) -> Result<Option<Refund>> {
        let inner = self.inner.lock().expect("poisoned");
        let Some(id) = inner.by_external_id.get(external_id).copied() else {
            return Ok(None);
        };
        Ok(inner.refunds.get(&id).cloned())
    }

    fn list_for_tx(&self, tx_id: TransactionId) -> Result<Vec<Refund>> {
        Ok(self
            .inner
            .lock()
            .expect("poisoned")
            .refunds
            .values()
            .filter(|r| r.original_tx_id == tx_id)
            .cloned()
            .collect())
    }

    fn update<F>(&self, id: RefundId, f: F) -> Result<Refund>
    where
        F: FnOnce(&mut Refund) -> Result<()>,
    {
        let mut inner = self.inner.lock().expect("poisoned");
        let existing = inner
            .refunds
            .get(&id)
            .ok_or_else(|| Error::NotFound(id.to_string()))?;
        let mut staged = existing.clone();
        f(&mut staged)?;
        // Commit only after the closure succeeded — the store's
        // atomicity contract.
        let result = staged.clone();
        inner.refunds.insert(id, staged);
        Ok(result)
    }
}

/// Two refund bodies are "equivalent" for idempotency if every
/// field outside the lifecycle state matches. We compare by
/// projecting both into a canonical tuple — order-insensitive on
/// metadata.
fn bodies_equivalent(a: &Refund, b: &Refund) -> bool {
    if a.original_tx_id != b.original_tx_id
        || a.amount != b.amount
        || a.reason != b.reason
        || a.external_id != b.external_id
    {
        return false;
    }
    if a.metadata.len() != b.metadata.len() {
        return false;
    }
    // Tally bag of (k, v) pairs — operator may have written them in
    // a different order on retry.
    let mut bag_a: HashMap<&(String, String), i32> = HashMap::new();
    let mut bag_b: HashMap<&(String, String), i32> = HashMap::new();
    for pair in &a.metadata {
        *bag_a.entry(pair).or_insert(0) += 1;
    }
    for pair in &b.metadata {
        *bag_b.entry(pair).or_insert(0) += 1;
    }
    bag_a == bag_b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reason::RefundReason;
    use op_core::{Currency, Money};
    use op_ledger::TransactionId;

    fn sample(ext: Option<&str>) -> Refund {
        let mut r = Refund::new(
            TransactionId::new(),
            Money::from_minor(500, Currency::USD),
            RefundReason::CustomerRequest,
            1_000,
        )
        .unwrap();
        if let Some(e) = ext {
            r = r.with_external_id(e);
        }
        r
    }

    #[test]
    fn create_and_get_round_trip() {
        let store = InMemoryRefundStore::new();
        let r = sample(None);
        let rid = r.id;
        store.create_refund(r.clone()).unwrap();
        let got = store.get_refund(rid).unwrap();
        assert_eq!(got, r);
    }

    #[test]
    fn external_id_idempotency_returns_same_for_same_body() {
        let store = InMemoryRefundStore::new();
        let r = sample(Some("ref-1"));
        let id1 = store.create_refund(r.clone()).unwrap();
        let id2 = store.create_refund(r).unwrap();
        assert_eq!(id1, id2);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn external_id_idempotency_rejects_different_body() {
        let store = InMemoryRefundStore::new();
        let r1 = sample(Some("ref-1"));
        store.create_refund(r1).unwrap();
        let mut r2 = sample(Some("ref-1"));
        r2.amount = Money::from_minor(999, Currency::USD);
        let err = store.create_refund(r2).unwrap_err();
        assert!(matches!(err, Error::IdempotencyMismatch(_)));
    }

    #[test]
    fn list_for_tx_groups_by_original() {
        let store = InMemoryRefundStore::new();
        let tx = TransactionId::new();
        for i in 0..3 {
            let mut r = sample(Some(&format!("ref-{i}")));
            r.original_tx_id = tx;
            store.create_refund(r).unwrap();
        }
        assert_eq!(store.list_for_tx(tx).unwrap().len(), 3);
        assert_eq!(store.list_for_tx(TransactionId::new()).unwrap().len(), 0);
    }

    #[test]
    fn update_commits_only_on_ok() {
        let store = InMemoryRefundStore::new();
        let r = sample(None);
        let rid = r.id;
        store.create_refund(r).unwrap();
        // Successful transition: commit.
        let after = store.update(rid, |r| r.submit("psp-1")).unwrap();
        assert_eq!(after.status.code(), "submitted");
        assert_eq!(store.get_refund(rid).unwrap().status.code(), "submitted");
        // Closure returning Err: discard.
        let _ = store
            .update(rid, |r| {
                r.approve()?;
                Err::<(), _>(Error::Invalid("simulated rollback".into()))
            })
            .unwrap_err();
        // Still in `submitted`, not `approved`.
        assert_eq!(store.get_refund(rid).unwrap().status.code(), "submitted");
    }

    #[test]
    fn get_refund_returns_not_found_for_unknown() {
        let store = InMemoryRefundStore::new();
        let id = RefundId::new();
        let err = store.get_refund(id).unwrap_err();
        assert!(matches!(err, Error::NotFound(_)));
    }
}
