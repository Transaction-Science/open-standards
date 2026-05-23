//! [`GraphReconciliationStore`] — IndraDB-backed
//! [`ReconciliationStore`].
//!
//! Each discrepancy becomes a `reconciliation_task` vertex. When the
//! discrepancy names a ledger transaction that already lives in the
//! **same** `GraphHandle` (e.g. written by [`crate::GraphLedgerStore`]),
//! a `task_about` edge is drawn task → `ledger_tx`. When it names a
//! statement line, a `statement_line` vertex is created (if absent)
//! and a `task_about` edge drawn task → `statement_line`.
//!
//! That shared-handle wiring is the payoff of the single-graph design:
//! an operator can traverse from a ledger account, to its
//! transactions, to the open reconciliation tasks that touch them, in
//! one graph.
//!
//! ## Idempotency
//!
//! Tasks key on the deterministic
//! [`task_id`](op_reconciliation::ReconciliationTask::task_id). Before
//! creating, `record_report` indexes existing `reconciliation_task`
//! vertices by their `task_id` property and reuses a match. So a
//! nightly job re-recording the same unresolved discrepancies is a
//! no-op rather than a pile of duplicates — matching the
//! cache-mirrors-the-graph discipline of the other graph stores.

use std::collections::HashMap;

use op_reconciliation::{
    Error as ReconError, ReconciliationReport, ReconciliationStore, ReconciliationTask,
    Result as ReconResult,
};
use serde_json::Value as Json;
use uuid::Uuid;

use crate::graph::{GraphHandle, etypes, vtypes};

/// IndraDB-backed reconciliation-task store.
pub struct GraphReconciliationStore {
    handle: GraphHandle,
}

impl GraphReconciliationStore {
    /// Construct on a fresh in-memory graph.
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_handle(GraphHandle::new_in_memory())
    }

    /// Construct on an existing handle — share it with a
    /// [`crate::GraphLedgerStore`] so `task_about` edges can reach
    /// the ledger transactions the tasks are about.
    #[must_use]
    pub fn with_handle(handle: GraphHandle) -> Self {
        Self { handle }
    }

    /// Borrow the underlying handle (for cross-store wiring / tests).
    #[must_use]
    pub fn handle(&self) -> &GraphHandle {
        &self.handle
    }

    /// Index existing task vertices by their `task_id` property.
    fn existing_task_ids(&self) -> ReconResult<HashMap<String, Uuid>> {
        let vertices = self
            .handle
            .vertices_of_type(vtypes::RECONCILIATION_TASK)
            .map_err(|e| ReconError::Backend(e.to_string()))?;
        let mut idx = HashMap::new();
        for v in vertices {
            let props = self
                .handle
                .get_vertex_properties(v.id)
                .map_err(|e| ReconError::Backend(e.to_string()))?;
            if let Some(Json::String(tid)) = props.get("task_id") {
                idx.insert(tid.clone(), v.id);
            }
        }
        Ok(idx)
    }

    /// Deterministic vertex id for a statement line (so re-recording
    /// the same line reuses its vertex). UUIDv5 over the source id in
    /// the URL namespace — stable, no clock, no RNG.
    fn statement_line_uuid(source_id: &str) -> Uuid {
        Uuid::new_v5(&Uuid::NAMESPACE_URL, source_id.as_bytes())
    }
}

impl ReconciliationStore for GraphReconciliationStore {
    fn record_report(&self, report: &ReconciliationReport) -> ReconResult<Vec<String>> {
        let mut existing = self.existing_task_ids()?;
        let mut touched = Vec::with_capacity(report.discrepancies.len());

        for d in &report.discrepancies {
            let desc = d.task_descriptor();
            touched.push(desc.task_id.clone());

            // Reuse an existing task vertex, or mint one.
            let task_uuid = if let Some(u) = existing.get(&desc.task_id) {
                *u
            } else {
                let u = Uuid::new_v4();
                self.handle
                    .create_vertex(vtypes::RECONCILIATION_TASK, u)
                    .map_err(|e| ReconError::Backend(e.to_string()))?;
                self.handle
                    .set_vertex_property(u, "task_id", Json::String(desc.task_id.clone()))
                    .map_err(|e| ReconError::Backend(e.to_string()))?;
                self.handle
                    .set_vertex_property(u, "kind", Json::String(desc.kind.to_owned()))
                    .map_err(|e| ReconError::Backend(e.to_string()))?;
                self.handle
                    .set_vertex_property(u, "detail", Json::String(desc.detail.clone()))
                    .map_err(|e| ReconError::Backend(e.to_string()))?;
                existing.insert(desc.task_id.clone(), u);
                u
            };

            // task --about--> ledger_tx, only if that tx vertex is in
            // this shared graph.
            if let Some(tx_id) = desc.ledger_tx_id {
                let tx_uuid = tx_id.0;
                let present = self
                    .handle
                    .vertex_exists(tx_uuid)
                    .map_err(|e| ReconError::Backend(e.to_string()))?;
                if present {
                    self.handle
                        .create_edge(task_uuid, etypes::TASK_ABOUT, tx_uuid)
                        .map_err(|e| ReconError::Backend(e.to_string()))?;
                }
            }

            // task --about--> statement_line (create the line vertex
            // if it doesn't exist yet).
            if let Some(src) = &desc.statement_source_id {
                let line_uuid = Self::statement_line_uuid(src);
                let exists = self
                    .handle
                    .vertex_exists(line_uuid)
                    .map_err(|e| ReconError::Backend(e.to_string()))?;
                if !exists {
                    self.handle
                        .create_vertex(vtypes::STATEMENT_LINE, line_uuid)
                        .map_err(|e| ReconError::Backend(e.to_string()))?;
                    self.handle
                        .set_vertex_property(line_uuid, "source_id", Json::String(src.clone()))
                        .map_err(|e| ReconError::Backend(e.to_string()))?;
                }
                self.handle
                    .create_edge(task_uuid, etypes::TASK_ABOUT, line_uuid)
                    .map_err(|e| ReconError::Backend(e.to_string()))?;
            }
        }

        // Matched pairs: emit `statement_line --reconciles--> ledger_tx`
        // edges so an operator can traverse from a tx to the bank
        // line that settled it. Skip when the tx vertex isn't in
        // this shared graph (caller owns a different ledger store).
        for pair in &report.matched_pairs {
            let line_uuid = Self::statement_line_uuid(&pair.statement_source_id);
            let tx_uuid = pair.tx_id.0;
            let tx_present = self
                .handle
                .vertex_exists(tx_uuid)
                .map_err(|e| ReconError::Backend(e.to_string()))?;
            if !tx_present {
                continue;
            }
            // Ensure the statement_line vertex exists (matched lines
            // didn't necessarily produce a task vertex, so the
            // earlier-loop creation may not have hit them).
            let line_present = self
                .handle
                .vertex_exists(line_uuid)
                .map_err(|e| ReconError::Backend(e.to_string()))?;
            if !line_present {
                self.handle
                    .create_vertex(vtypes::STATEMENT_LINE, line_uuid)
                    .map_err(|e| ReconError::Backend(e.to_string()))?;
                self.handle
                    .set_vertex_property(
                        line_uuid,
                        "source_id",
                        Json::String(pair.statement_source_id.clone()),
                    )
                    .map_err(|e| ReconError::Backend(e.to_string()))?;
            }
            self.handle
                .create_edge(line_uuid, etypes::RECONCILES, tx_uuid)
                .map_err(|e| ReconError::Backend(e.to_string()))?;
            if pair.fuzzy {
                // Tag the edge so a downstream consumer can tell
                // exact vs heuristic matches apart in audits.
                let edges = self
                    .handle
                    .out_edges(line_uuid, etypes::RECONCILES)
                    .map_err(|e| ReconError::Backend(e.to_string()))?;
                if let Some(edge) = edges.into_iter().find(|e| e.to == tx_uuid) {
                    let _ = self
                        .handle
                        .set_edge_property(&edge, "fuzzy", Json::Bool(true));
                }
            }
        }

        Ok(touched)
    }

    fn list_tasks(&self) -> ReconResult<Vec<ReconciliationTask>> {
        let vertices = self
            .handle
            .vertices_of_type(vtypes::RECONCILIATION_TASK)
            .map_err(|e| ReconError::Backend(e.to_string()))?;
        let mut out = Vec::new();
        for v in vertices {
            let props = self
                .handle
                .get_vertex_properties(v.id)
                .map_err(|e| ReconError::Backend(e.to_string()))?;
            let s = |k: &str| match props.get(k) {
                Some(Json::String(s)) => s.clone(),
                _ => String::new(),
            };
            out.push(ReconciliationTask {
                task_id: s("task_id"),
                kind: s("kind"),
                detail: s("detail"),
            });
        }
        Ok(out)
    }
}
