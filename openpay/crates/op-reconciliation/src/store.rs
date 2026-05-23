//! The pluggable reconciliation-task store trait.
//!
//! Defined here (not in `op-graph`) so the graph-backed
//! implementation can depend on this crate without this crate ever
//! depending on `indradb`. Same license-hygiene seam used by
//! `LedgerStore` / `WebhookStore`: an Apache-2.0-only operator can
//! supply their own store and never link the MPL-2.0 graph backend.

use crate::discrepancy::ReconciliationReport;
use crate::error::Result;

/// A persisted reconciliation task — one open discrepancy an operator
/// still needs to resolve.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReconciliationTask {
    /// Deterministic id (see
    /// [`Discrepancy::task_descriptor`](crate::Discrepancy::task_descriptor)).
    pub task_id: String,
    /// Discrepancy class.
    pub kind: String,
    /// Operator-facing one-line summary.
    pub detail: String,
}

/// Pluggable storage for reconciliation tasks.
pub trait ReconciliationStore {
    /// Persist every discrepancy in `report` as a task.
    ///
    /// **Idempotent**: tasks key on the deterministic
    /// [`task_id`](ReconciliationTask::task_id), so re-recording a
    /// report whose discrepancies are unchanged creates no
    /// duplicates. Returns the task ids touched (created or already
    /// present), in report order.
    ///
    /// # Errors
    /// Backend-specific persistence failure.
    fn record_report(&self, report: &ReconciliationReport) -> Result<Vec<String>>;

    /// Every reconciliation task currently recorded.
    ///
    /// # Errors
    /// Backend-specific read failure.
    fn list_tasks(&self) -> Result<Vec<ReconciliationTask>>;
}
