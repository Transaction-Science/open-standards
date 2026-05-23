//! [`GraphRefundStore`] — Minigraf-backed [`RefundStore`].
//!
//! Each [`Refund`] becomes a `refund` vertex keyed by its
//! [`RefundId`]. We persist the full `Refund` JSON in a `state`
//! property and additionally expose three indexed properties:
//!
//! - `external_id` — operator idempotency token (when set).
//! - `original_tx_id` — for `list_for_tx` lookup.
//! - `status_code` — for filtering by lifecycle state.
//!
//! When the original ledger transaction lives on the same
//! [`GraphHandle`], a `refund --refunds--> ledger_tx` edge is drawn
//! so the audit report can walk from a posted transaction to every
//! refund issued against it in one hop.

use op_ledger::TransactionId;
use op_refund::{Error as RefundError, Refund, RefundId, RefundStore, Result as RefundResult};
use serde_json::Value as Json;

use crate::graph::{GraphHandle, etypes, vtypes};

/// Graph-backed refund store.
pub struct GraphRefundStore {
    handle: GraphHandle,
}

impl GraphRefundStore {
    /// Construct on a fresh in-memory graph.
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_handle(GraphHandle::new_in_memory())
    }

    /// Construct on a shared handle.
    #[must_use]
    pub fn with_handle(handle: GraphHandle) -> Self {
        Self { handle }
    }

    /// Borrow the underlying handle.
    #[must_use]
    pub fn handle(&self) -> &GraphHandle {
        &self.handle
    }

    /// Diagnostic: number of refund vertices in the graph.
    pub fn len(&self) -> usize {
        self.handle
            .vertices_of_type(vtypes::REFUND)
            .map_or(0, |v| v.len())
    }

    /// True iff no refunds stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Load a `Refund` from its vertex `state` property.
    fn load_refund(&self, id: RefundId) -> RefundResult<Option<Refund>> {
        if !self
            .handle
            .vertex_exists(id.as_uuid())
            .map_err(graph_to_refund_err)?
        {
            return Ok(None);
        }
        let props = self
            .handle
            .get_vertex_properties(id.as_uuid())
            .map_err(graph_to_refund_err)?;
        let state = props
            .get("state")
            .ok_or_else(|| RefundError::Invalid("refund vertex missing `state`".into()))?;
        let refund: Refund = serde_json::from_value(state.clone())
            .map_err(|e| RefundError::Invalid(format!("refund state decode: {e}")))?;
        Ok(Some(refund))
    }

    /// Persist a `Refund`'s state + indexed properties. The vertex
    /// must already exist.
    fn persist_refund(&self, refund: &Refund) -> RefundResult<()> {
        let state = serde_json::to_value(refund)
            .map_err(|e| RefundError::Invalid(format!("refund encode: {e}")))?;
        let id = refund.id.as_uuid();
        self.handle
            .set_vertex_property(id, "state", state)
            .map_err(graph_to_refund_err)?;
        self.handle
            .set_vertex_property(
                id,
                "status_code",
                Json::String(refund.status.code().to_owned()),
            )
            .map_err(graph_to_refund_err)?;
        self.handle
            .set_vertex_property(
                id,
                "original_tx_id",
                Json::String(refund.original_tx_id.to_string()),
            )
            .map_err(graph_to_refund_err)?;
        if let Some(ext) = &refund.external_id {
            self.handle
                .set_vertex_property(id, "external_id", Json::String(ext.clone()))
                .map_err(graph_to_refund_err)?;
        }
        Ok(())
    }

    /// Find the existing refund vertex id whose `external_id`
    /// property matches `external_id`. Linear scan — Minigraf
    /// rebuilds with an index later; for now O(n_refunds) is fine.
    fn lookup_by_external_id(&self, external_id: &str) -> RefundResult<Option<RefundId>> {
        let vertices = self
            .handle
            .vertices_of_type(vtypes::REFUND)
            .map_err(graph_to_refund_err)?;
        for v in vertices {
            let props = self
                .handle
                .get_vertex_properties(v.id)
                .map_err(graph_to_refund_err)?;
            if let Some(Json::String(ext)) = props.get("external_id")
                && ext == external_id
            {
                return Ok(Some(RefundId::from_uuid(v.id)));
            }
        }
        Ok(None)
    }
}

fn graph_to_refund_err(e: crate::Error) -> RefundError {
    RefundError::Invalid(format!("graph backend: {e}"))
}

/// Body-equivalence for refund idempotency. Mirrors the in-memory
/// store's check: amount, reason, original tx, external id, and an
/// order-insensitive metadata bag.
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
    let mut bag_a: std::collections::HashMap<&(String, String), i32> = Default::default();
    let mut bag_b: std::collections::HashMap<&(String, String), i32> = Default::default();
    for pair in &a.metadata {
        *bag_a.entry(pair).or_insert(0) += 1;
    }
    for pair in &b.metadata {
        *bag_b.entry(pair).or_insert(0) += 1;
    }
    bag_a == bag_b
}

impl RefundStore for GraphRefundStore {
    fn create_refund(&self, refund: Refund) -> RefundResult<RefundId> {
        // Idempotency: look up by external_id first.
        if let Some(ext) = &refund.external_id
            && let Some(existing_id) = self.lookup_by_external_id(ext)?
        {
            let existing = self
                .load_refund(existing_id)?
                .ok_or_else(|| RefundError::Invalid("indexed refund vanished".into()))?;
            if bodies_equivalent(&existing, &refund) {
                return Ok(existing_id);
            }
            return Err(RefundError::IdempotencyMismatch(ext.clone()));
        }

        // Mint vertex.
        let id = refund.id;
        self.handle
            .create_vertex(vtypes::REFUND, id.as_uuid())
            .map_err(graph_to_refund_err)?;
        self.persist_refund(&refund)?;

        // Cross-store edge: refund --refunds--> ledger_tx, only if
        // the tx vertex is in the same graph.
        let tx_uuid = refund.original_tx_id.as_uuid();
        if self
            .handle
            .vertex_exists(tx_uuid)
            .map_err(graph_to_refund_err)?
        {
            self.handle
                .create_edge(id.as_uuid(), etypes::REFUND_REFUNDS, tx_uuid)
                .map_err(graph_to_refund_err)?;
        }
        Ok(id)
    }

    fn get_refund(&self, id: RefundId) -> RefundResult<Refund> {
        self.load_refund(id)?
            .ok_or_else(|| RefundError::NotFound(id.to_string()))
    }

    fn find_by_external_id(&self, external_id: &str) -> RefundResult<Option<Refund>> {
        let Some(id) = self.lookup_by_external_id(external_id)? else {
            return Ok(None);
        };
        self.load_refund(id)
    }

    fn list_for_tx(&self, tx_id: TransactionId) -> RefundResult<Vec<Refund>> {
        let target = tx_id.to_string();
        let vertices = self
            .handle
            .vertices_of_type(vtypes::REFUND)
            .map_err(graph_to_refund_err)?;
        let mut out = Vec::new();
        for v in vertices {
            let props = self
                .handle
                .get_vertex_properties(v.id)
                .map_err(graph_to_refund_err)?;
            if let Some(Json::String(t)) = props.get("original_tx_id")
                && t == &target
                && let Some(state) = props.get("state")
            {
                let r: Refund = serde_json::from_value(state.clone())
                    .map_err(|e| RefundError::Invalid(format!("refund decode: {e}")))?;
                out.push(r);
            }
        }
        Ok(out)
    }

    fn update<F>(&self, id: RefundId, f: F) -> RefundResult<Refund>
    where
        F: FnOnce(&mut Refund) -> RefundResult<()>,
    {
        // Load → stage → run closure → only persist on Ok. The
        // atomicity contract is "no partial mutation persists" —
        // we run the mutation against a stacked clone and only
        // touch the graph if the closure approves.
        let mut staged = self.get_refund(id)?;
        f(&mut staged)?;
        self.persist_refund(&staged)?;
        Ok(staged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money};
    use op_refund::reason::RefundReason;

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
        let store = GraphRefundStore::new_in_memory();
        let r = sample(None);
        let rid = r.id;
        store.create_refund(r.clone()).unwrap();
        let got = store.get_refund(rid).unwrap();
        assert_eq!(got, r);
    }

    #[test]
    fn external_id_idempotency_returns_same_for_same_body() {
        let store = GraphRefundStore::new_in_memory();
        let r = sample(Some("ext-1"));
        let id1 = store.create_refund(r.clone()).unwrap();
        let id2 = store.create_refund(r).unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn external_id_mismatch_rejects() {
        let store = GraphRefundStore::new_in_memory();
        let r1 = sample(Some("ext-1"));
        store.create_refund(r1).unwrap();
        let mut r2 = sample(Some("ext-1"));
        r2.amount = Money::from_minor(999, Currency::USD);
        let err = store.create_refund(r2).unwrap_err();
        assert!(matches!(err, RefundError::IdempotencyMismatch(_)));
    }

    #[test]
    fn list_for_tx_groups_by_original() {
        let store = GraphRefundStore::new_in_memory();
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
        let store = GraphRefundStore::new_in_memory();
        let r = sample(None);
        let rid = r.id;
        store.create_refund(r).unwrap();
        store.update(rid, |r| r.submit("psp-1")).unwrap();
        assert_eq!(store.get_refund(rid).unwrap().status.code(), "submitted");
        let _ = store
            .update(rid, |r| {
                r.approve()?;
                Err(RefundError::Invalid("rollback".into()))
            })
            .unwrap_err();
        // Still submitted; the failed update did not persist.
        assert_eq!(store.get_refund(rid).unwrap().status.code(), "submitted");
    }

    #[test]
    fn get_unknown_returns_not_found() {
        let store = GraphRefundStore::new_in_memory();
        let err = store.get_refund(RefundId::new()).unwrap_err();
        assert!(matches!(err, RefundError::NotFound(_)));
    }

    #[test]
    fn find_by_external_id_returns_none_when_absent() {
        let store = GraphRefundStore::new_in_memory();
        assert!(store.find_by_external_id("nope").unwrap().is_none());
    }
}
