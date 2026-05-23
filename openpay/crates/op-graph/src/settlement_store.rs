//! [`GraphSettlementStore`] — Minigraf-backed [`SettlementStore`].
//!
//! Each batch is a `settlement_batch` vertex with the full batch
//! JSON in `state`. Each `BatchEntry` becomes a
//! `batch --includes--> ledger_tx` edge when the referenced
//! transaction exists in the same graph — this is what lets the
//! audit report walk from a posted tx to the batch that paid it
//! out.

use op_settlement::{
    Batch, BatchId, Error as SettleError, Result as SettleResult, SettlementStore,
};
use serde_json::Value as Json;

use crate::graph::{GraphHandle, etypes, vtypes};

/// Graph-backed settlement store.
pub struct GraphSettlementStore {
    handle: GraphHandle,
}

impl GraphSettlementStore {
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

    /// Diagnostic: number of batch vertices in the graph.
    pub fn len(&self) -> usize {
        self.handle
            .vertices_of_type(vtypes::SETTLEMENT_BATCH)
            .map_or(0, |v| v.len())
    }

    /// True iff no batches stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn load_batch(&self, id: BatchId) -> SettleResult<Option<Batch>> {
        if !self.handle.vertex_exists(id.as_uuid()).map_err(g2s)? {
            return Ok(None);
        }
        let props = self
            .handle
            .get_vertex_properties(id.as_uuid())
            .map_err(g2s)?;
        let state = props
            .get("state")
            .ok_or_else(|| SettleError::Invalid("batch vertex missing `state`".into()))?;
        let b: Batch = serde_json::from_value(state.clone())
            .map_err(|e| SettleError::Invalid(format!("batch decode: {e}")))?;
        Ok(Some(b))
    }

    fn persist_batch(&self, b: &Batch) -> SettleResult<()> {
        let state = serde_json::to_value(b)
            .map_err(|e| SettleError::Invalid(format!("batch encode: {e}")))?;
        let id = b.id.as_uuid();
        self.handle
            .set_vertex_property(id, "state", state)
            .map_err(g2s)?;
        self.handle
            .set_vertex_property(id, "status_code", Json::String(b.status.code().to_owned()))
            .map_err(g2s)?;
        self.handle
            .set_vertex_property(
                id,
                "currency_code",
                Json::String(b.currency.code().to_owned()),
            )
            .map_err(g2s)?;
        if let Some(ext) = &b.external_id {
            self.handle
                .set_vertex_property(id, "external_id", Json::String(ext.clone()))
                .map_err(g2s)?;
        }
        Ok(())
    }

    fn lookup_by_external_id(&self, external_id: &str) -> SettleResult<Option<BatchId>> {
        let vertices = self
            .handle
            .vertices_of_type(vtypes::SETTLEMENT_BATCH)
            .map_err(g2s)?;
        for v in vertices {
            let props = self.handle.get_vertex_properties(v.id).map_err(g2s)?;
            if let Some(Json::String(ext)) = props.get("external_id")
                && ext == external_id
            {
                return Ok(Some(BatchId::from_uuid(v.id)));
            }
        }
        Ok(None)
    }

    /// Add a `batch --includes--> tx` edge for each entry whose
    /// referenced transaction vertex exists in the same graph.
    /// Idempotent on the existence of edges of that type by skipping
    /// already-linked tx ids (we re-emit on every persist, but
    /// `set_vertex_property` semantics elsewhere are retract-then-
    /// assert; edges are append-only so we de-dupe explicitly).
    fn sync_entry_edges(&self, b: &Batch) -> SettleResult<()> {
        let batch_uuid = b.id.as_uuid();
        let existing: std::collections::HashSet<uuid::Uuid> = self
            .handle
            .out_edges(batch_uuid, etypes::BATCH_INCLUDES)
            .map_err(g2s)?
            .into_iter()
            .map(|e| e.to)
            .collect();
        for entry in &b.entries {
            let tx_uuid = entry.tx_id.as_uuid();
            if existing.contains(&tx_uuid) {
                continue;
            }
            if !self.handle.vertex_exists(tx_uuid).map_err(g2s)? {
                continue;
            }
            self.handle
                .create_edge(batch_uuid, etypes::BATCH_INCLUDES, tx_uuid)
                .map_err(g2s)?;
        }
        Ok(())
    }
}

fn g2s(e: crate::Error) -> SettleError {
    SettleError::Invalid(format!("graph backend: {e}"))
}

fn bodies_equivalent(a: &Batch, b: &Batch) -> bool {
    a.currency == b.currency
        && a.rail == b.rail
        && a.external_id == b.external_id
        && a.entries == b.entries
}

impl SettlementStore for GraphSettlementStore {
    fn create_batch(&self, batch: Batch) -> SettleResult<BatchId> {
        if let Some(ext) = &batch.external_id
            && let Some(existing_id) = self.lookup_by_external_id(ext)?
        {
            let existing = self
                .load_batch(existing_id)?
                .ok_or_else(|| SettleError::Invalid("indexed batch vanished".into()))?;
            if bodies_equivalent(&existing, &batch) {
                return Ok(existing_id);
            }
            return Err(SettleError::IdempotencyMismatch(ext.clone()));
        }

        let id = batch.id;
        self.handle
            .create_vertex(vtypes::SETTLEMENT_BATCH, id.as_uuid())
            .map_err(g2s)?;
        self.persist_batch(&batch)?;
        self.sync_entry_edges(&batch)?;
        Ok(id)
    }

    fn get_batch(&self, id: BatchId) -> SettleResult<Batch> {
        self.load_batch(id)?
            .ok_or_else(|| SettleError::NotFound(id.to_string()))
    }

    fn find_by_external_id(&self, external_id: &str) -> SettleResult<Option<Batch>> {
        let Some(id) = self.lookup_by_external_id(external_id)? else {
            return Ok(None);
        };
        self.load_batch(id)
    }

    fn list_open(&self) -> SettleResult<Vec<Batch>> {
        let vertices = self
            .handle
            .vertices_of_type(vtypes::SETTLEMENT_BATCH)
            .map_err(g2s)?;
        let mut out = Vec::new();
        for v in vertices {
            let props = self.handle.get_vertex_properties(v.id).map_err(g2s)?;
            if matches!(props.get("status_code"), Some(Json::String(s)) if s == "open")
                && let Some(state) = props.get("state")
            {
                let b: Batch = serde_json::from_value(state.clone())
                    .map_err(|e| SettleError::Invalid(format!("batch decode: {e}")))?;
                out.push(b);
            }
        }
        Ok(out)
    }

    fn update<F>(&self, id: BatchId, f: F) -> SettleResult<Batch>
    where
        F: FnOnce(&mut Batch) -> SettleResult<()>,
    {
        let mut staged = self.get_batch(id)?;
        f(&mut staged)?;
        self.persist_batch(&staged)?;
        self.sync_entry_edges(&staged)?;
        Ok(staged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{Currency, Money};
    use op_ledger::TransactionId;
    use op_settlement::{HoldbackPolicy, PayoutRail};

    #[test]
    fn create_and_get_round_trip() {
        let store = GraphSettlementStore::new_in_memory();
        let b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000);
        let bid = b.id;
        store.create_batch(b.clone()).unwrap();
        assert_eq!(store.get_batch(bid).unwrap(), b);
    }

    #[test]
    fn idempotency() {
        let store = GraphSettlementStore::new_in_memory();
        let b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000).with_external_id("e1");
        let id1 = store.create_batch(b.clone()).unwrap();
        let id2 = store.create_batch(b).unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn list_open_excludes_closed() {
        let store = GraphSettlementStore::new_in_memory();
        let b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000);
        let bid = b.id;
        store.create_batch(b).unwrap();
        store
            .update(bid, |bb| {
                bb.add_entry(
                    TransactionId::new(),
                    Money::from_minor(100, Currency::USD),
                    None,
                )?;
                let gross = bb.gross()?;
                let hb = HoldbackPolicy::none().compute(gross, 0)?;
                bb.close(hb, 2_000)
            })
            .unwrap();
        assert!(store.list_open().unwrap().is_empty());
    }
}
