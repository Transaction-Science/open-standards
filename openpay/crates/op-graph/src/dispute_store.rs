//! [`GraphDisputeStore`] — Minigraf-backed [`DisputeStore`].
//!
//! Mirrors [`crate::GraphRefundStore`]: one `dispute` vertex per
//! dispute, indexed `external_id` / `original_tx_id` /
//! `status_code` properties, full state JSON in `state`, and a
//! `dispute --disputes--> ledger_tx` edge when the original
//! transaction lives on the same handle.

use op_dispute::{
    Dispute, DisputeId, DisputeStore, Error as DisputeError, Result as DisputeResult,
};
use op_ledger::TransactionId;
use serde_json::Value as Json;

use crate::graph::{GraphHandle, etypes, vtypes};

/// Graph-backed dispute store.
pub struct GraphDisputeStore {
    handle: GraphHandle,
}

impl GraphDisputeStore {
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

    /// Diagnostic: number of dispute vertices in the graph.
    pub fn len(&self) -> usize {
        self.handle
            .vertices_of_type(vtypes::DISPUTE)
            .map_or(0, |v| v.len())
    }

    /// True iff no disputes stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn load_dispute(&self, id: DisputeId) -> DisputeResult<Option<Dispute>> {
        if !self.handle.vertex_exists(id.as_uuid()).map_err(g2d)? {
            return Ok(None);
        }
        let props = self
            .handle
            .get_vertex_properties(id.as_uuid())
            .map_err(g2d)?;
        let state = props
            .get("state")
            .ok_or_else(|| DisputeError::Invalid("dispute vertex missing `state`".into()))?;
        let d: Dispute = serde_json::from_value(state.clone())
            .map_err(|e| DisputeError::Invalid(format!("dispute decode: {e}")))?;
        Ok(Some(d))
    }

    fn persist_dispute(&self, d: &Dispute) -> DisputeResult<()> {
        let state = serde_json::to_value(d)
            .map_err(|e| DisputeError::Invalid(format!("dispute encode: {e}")))?;
        let id = d.id.as_uuid();
        self.handle
            .set_vertex_property(id, "state", state)
            .map_err(g2d)?;
        self.handle
            .set_vertex_property(id, "status_code", Json::String(d.status.code().to_owned()))
            .map_err(g2d)?;
        self.handle
            .set_vertex_property(
                id,
                "original_tx_id",
                Json::String(d.original_tx_id.to_string()),
            )
            .map_err(g2d)?;
        if let Some(ext) = &d.external_id {
            self.handle
                .set_vertex_property(id, "external_id", Json::String(ext.clone()))
                .map_err(g2d)?;
        }
        Ok(())
    }

    fn lookup_by_external_id(&self, external_id: &str) -> DisputeResult<Option<DisputeId>> {
        let vertices = self.handle.vertices_of_type(vtypes::DISPUTE).map_err(g2d)?;
        for v in vertices {
            let props = self.handle.get_vertex_properties(v.id).map_err(g2d)?;
            if let Some(Json::String(ext)) = props.get("external_id")
                && ext == external_id
            {
                return Ok(Some(DisputeId::from_uuid(v.id)));
            }
        }
        Ok(None)
    }
}

fn g2d(e: crate::Error) -> DisputeError {
    DisputeError::Invalid(format!("graph backend: {e}"))
}

fn bodies_equivalent(a: &Dispute, b: &Dispute) -> bool {
    a.original_tx_id == b.original_tx_id
        && a.amount == b.amount
        && a.reason == b.reason
        && a.network_reason_code == b.network_reason_code
        && a.external_id == b.external_id
}

impl DisputeStore for GraphDisputeStore {
    fn create_dispute(&self, dispute: Dispute) -> DisputeResult<DisputeId> {
        if let Some(ext) = &dispute.external_id
            && let Some(existing_id) = self.lookup_by_external_id(ext)?
        {
            let existing = self
                .load_dispute(existing_id)?
                .ok_or_else(|| DisputeError::Invalid("indexed dispute vanished".into()))?;
            if bodies_equivalent(&existing, &dispute) {
                return Ok(existing_id);
            }
            return Err(DisputeError::IdempotencyMismatch(ext.clone()));
        }

        let id = dispute.id;
        self.handle
            .create_vertex(vtypes::DISPUTE, id.as_uuid())
            .map_err(g2d)?;
        self.persist_dispute(&dispute)?;

        let tx_uuid = dispute.original_tx_id.as_uuid();
        if self.handle.vertex_exists(tx_uuid).map_err(g2d)? {
            self.handle
                .create_edge(id.as_uuid(), etypes::DISPUTE_DISPUTES, tx_uuid)
                .map_err(g2d)?;
        }
        Ok(id)
    }

    fn get_dispute(&self, id: DisputeId) -> DisputeResult<Dispute> {
        self.load_dispute(id)?
            .ok_or_else(|| DisputeError::NotFound(id.to_string()))
    }

    fn find_by_external_id(&self, external_id: &str) -> DisputeResult<Option<Dispute>> {
        let Some(id) = self.lookup_by_external_id(external_id)? else {
            return Ok(None);
        };
        self.load_dispute(id)
    }

    fn list_for_tx(&self, tx_id: TransactionId) -> DisputeResult<Vec<Dispute>> {
        let target = tx_id.to_string();
        let vertices = self.handle.vertices_of_type(vtypes::DISPUTE).map_err(g2d)?;
        let mut out = Vec::new();
        for v in vertices {
            let props = self.handle.get_vertex_properties(v.id).map_err(g2d)?;
            if let Some(Json::String(t)) = props.get("original_tx_id")
                && t == &target
                && let Some(state) = props.get("state")
            {
                let d: Dispute = serde_json::from_value(state.clone())
                    .map_err(|e| DisputeError::Invalid(format!("dispute decode: {e}")))?;
                out.push(d);
            }
        }
        Ok(out)
    }

    fn update<F>(&self, id: DisputeId, f: F) -> DisputeResult<Dispute>
    where
        F: FnOnce(&mut Dispute) -> DisputeResult<()>,
    {
        let mut staged = self.get_dispute(id)?;
        f(&mut staged)?;
        self.persist_dispute(&staged)?;
        Ok(staged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money};
    use op_dispute::{evidence::EvidenceRef, reason::DisputeReason};

    fn sample(ext: Option<&str>) -> Dispute {
        let mut d = Dispute::new(
            TransactionId::new(),
            Money::from_minor(500, Currency::USD),
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
    fn round_trip() {
        let store = GraphDisputeStore::new_in_memory();
        let d = sample(None);
        let did = d.id;
        store.create_dispute(d.clone()).unwrap();
        assert_eq!(store.get_dispute(did).unwrap(), d);
    }

    #[test]
    fn idempotent_on_external_id() {
        let store = GraphDisputeStore::new_in_memory();
        let d = sample(Some("ext-1"));
        let id1 = store.create_dispute(d.clone()).unwrap();
        let id2 = store.create_dispute(d).unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn rejects_mismatched_external_id_body() {
        let store = GraphDisputeStore::new_in_memory();
        store.create_dispute(sample(Some("e"))).unwrap();
        let mut d2 = sample(Some("e"));
        d2.amount = Money::from_minor(999, Currency::USD);
        assert!(matches!(
            store.create_dispute(d2).unwrap_err(),
            DisputeError::IdempotencyMismatch(_)
        ));
    }

    #[test]
    fn update_persists_evidence() {
        let store = GraphDisputeStore::new_in_memory();
        let d = sample(None);
        let did = d.id;
        store.create_dispute(d).unwrap();
        store
            .update(did, |d| {
                d.attach_evidence(EvidenceRef {
                    kind: "receipt".into(),
                    url: "https://x/a.pdf".into(),
                    note: None,
                    attached_at_unix_secs: 2_000,
                })
            })
            .unwrap();
        let after = store.get_dispute(did).unwrap();
        assert_eq!(after.evidence.len(), 1);
    }
}
