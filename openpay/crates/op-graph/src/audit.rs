//! Audit report: the single-shot join across the whole graph.
//!
//! Operators (and auditors) want to ask: *"for every ledger
//! transaction in this window, show me the rail that processed
//! it, the reconciliation tasks that touched it, and the status."*
//! That join already exists implicitly across the typed stores;
//! Phase 21 surfaces it as a structured report.
//!
//! Built from a single [`GraphHandle`] — same substrate every other
//! store sits on, so the join is consistent: all the vertices are
//! in one graph; the report walks them in one pass.

use std::collections::HashMap;

use op_ledger::TransactionId;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::error::Result;
use crate::graph::{GraphHandle, etypes, vtypes};

/// One row of the audit report — everything we can learn about a
/// single ledger transaction by traversing the shared graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditEntry {
    /// The ledger transaction.
    pub tx_id: TransactionId,
    /// The operator-supplied idempotency key, when present. This is
    /// the join key against `rail_attempt.external_id_hint`.
    pub external_id: Option<String>,
    /// Lifecycle state at report time (`"pending"` / `"posted"` /
    /// `"archived"`). Use the time-travel reads on
    /// `LedgerHistory` for a historical view.
    pub status: String,
    /// The bi-temporal counter just after this tx's writes settled
    /// — operators pair this with `LedgerHistory::balance_as_of` to
    /// reconstruct the world the moment this tx posted.
    pub posted_at_tx_count: u64,
    /// Caller-supplied effective time, unix epoch seconds.
    pub effective_at_unix_secs: u64,
    /// Sum of debit-side entries, in minor units. Single-currency
    /// only (multi-currency txs return `None`).
    pub settled_amount_minor: Option<i64>,
    /// Currency code if [`Self::settled_amount_minor`] is `Some`.
    pub currency_code: Option<String>,
    /// Rail (`"Card"`, `"A2a"`, ...) that processed this tx, via
    /// the `rail_attempt` vertex whose `external_id_hint` matches.
    /// `None` if no telemetry was recorded or external_id is absent.
    pub rail: Option<String>,
    /// Driver name (operator-registered) of the processing adapter.
    pub driver: Option<String>,
    /// Reconciliation tasks that point at this tx via `task_about`
    /// edges. Each entry is the task's stable id (the deterministic
    /// `task_id` string assigned at record-report time).
    pub reconciliation_task_ids: Vec<String>,
}

/// The structured report for an audit window.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditReport {
    /// `[start_tx_count, end_tx_count]` window the caller asked
    /// about, inclusive both sides.
    pub window: (u64, u64),
    /// One entry per ledger transaction whose `posted_at_tx_count`
    /// falls in the window, ordered ascending by that counter.
    pub entries: Vec<AuditEntry>,
    /// When the report was built (unix epoch seconds — operator-
    /// supplied so the report is deterministic for tests / replay).
    pub generated_at_unix_secs: u64,
}

impl AuditReport {
    /// Build a report for the `[start_tx, end_tx]` window.
    ///
    /// Walks all `ledger_tx` vertices, filters by
    /// `posted_at_tx_count`, then joins each one against
    /// `rail_attempt` (by `external_id`) and inbound `task_about`
    /// edges (the reconciliation tasks). One graph, one pass.
    ///
    /// # Errors
    /// Backend-specific propagation from the graph reads. Returns
    /// an empty report when `end_tx < start_tx`.
    pub fn for_window(
        handle: &GraphHandle,
        start_tx: u64,
        end_tx: u64,
        generated_at_unix_secs: u64,
    ) -> Result<Self> {
        if end_tx < start_tx {
            return Ok(Self {
                window: (start_tx, end_tx),
                entries: Vec::new(),
                generated_at_unix_secs,
            });
        }

        // Build an external_id → (rail, driver) index up front so
        // we don't re-scan rail_attempt vertices once per tx.
        let rail_index = build_rail_index(handle)?;

        let txs = handle.vertices_of_type(vtypes::LEDGER_TX)?;
        let mut entries: Vec<AuditEntry> = Vec::new();
        for v in txs {
            let props = handle.get_vertex_properties(v.id)?;
            let Some(posted_at) = props.get("posted_at_tx_count").and_then(json_u64) else {
                continue;
            };
            if posted_at < start_tx || posted_at > end_tx {
                continue;
            }

            let tx_id = TransactionId::from_uuid(v.id);
            let external_id = props.get("external_id").and_then(json_string);
            let status = props
                .get("status")
                .and_then(json_string)
                .unwrap_or_default();
            let effective_at_unix_secs = props
                .get("effective_at_unix_secs")
                .and_then(json_u64)
                .unwrap_or(0);

            let (settled_amount_minor, currency_code) = settled_amount(handle, v.id)?;

            let (rail, driver) = match &external_id {
                Some(eid) => rail_index
                    .get(eid)
                    .cloned()
                    .map_or((None, None), |(r, d)| (Some(r), Some(d))),
                None => (None, None),
            };

            let reconciliation_task_ids = task_ids_about(handle, v.id)?;

            entries.push(AuditEntry {
                tx_id,
                external_id,
                status,
                posted_at_tx_count: posted_at,
                effective_at_unix_secs,
                settled_amount_minor,
                currency_code,
                rail,
                driver,
                reconciliation_task_ids,
            });
        }

        // Stable ascending order by posted_at_tx_count so the
        // report reads like an audit log.
        entries.sort_by_key(|e| e.posted_at_tx_count);

        Ok(Self {
            window: (start_tx, end_tx),
            entries,
            generated_at_unix_secs,
        })
    }
}

// ============================================================
// Helpers
// ============================================================

/// `external_id_hint → (rail, driver)` lookup table from every
/// `rail_attempt` vertex. Last write wins on duplicates — typical
/// when the orchestrator retried, the latest attempt is what posted
/// the tx. (Operators who want the *whole* attempt list use
/// [`crate::GraphRailTelemetry::list_attempts`] directly.)
fn build_rail_index(handle: &GraphHandle) -> Result<HashMap<String, (String, String)>> {
    let attempts = handle.vertices_of_type(vtypes::RAIL_ATTEMPT)?;
    let mut idx: HashMap<String, (String, String)> = HashMap::new();
    for v in attempts {
        let props = handle.get_vertex_properties(v.id)?;
        let Some(ext) = props.get("external_id_hint").and_then(json_string) else {
            continue;
        };
        let rail = props.get("rail").and_then(json_string).unwrap_or_default();
        let driver = props
            .get("driver")
            .and_then(json_string)
            .unwrap_or_default();
        idx.insert(ext, (rail, driver));
    }
    Ok(idx)
}

/// Sum the debit-side entry amounts for a tx vertex. Returns
/// `(None, None)` for multi-currency or entry-less txs.
fn settled_amount(
    handle: &GraphHandle,
    tx_uuid: uuid::Uuid,
) -> Result<(Option<i64>, Option<String>)> {
    let debits = handle.out_edges(tx_uuid, etypes::LEDGER_DEBIT)?;
    let mut total: Option<i64> = None;
    let mut currency: Option<String> = None;
    for edge in debits {
        let props = handle.get_edge_properties(&edge)?;
        let Some(amount) = props.get("amount_minor").and_then(json_i64) else {
            continue;
        };
        let Some(ccode) = props.get("currency_code").and_then(json_string) else {
            continue;
        };
        match (&total, &currency) {
            (None, None) => {
                total = Some(amount.abs());
                currency = Some(ccode);
            }
            (Some(_), Some(c)) if *c == ccode => {
                total = total.map(|t| t.saturating_add(amount.abs()));
            }
            _ => {
                // Mixed-currency: surface as unknown.
                return Ok((None, None));
            }
        }
    }
    Ok((total, currency))
}

/// Task ids referencing this tx via inbound `task_about` edges.
fn task_ids_about(handle: &GraphHandle, tx_uuid: uuid::Uuid) -> Result<Vec<String>> {
    let edges = handle.in_edges(tx_uuid, etypes::TASK_ABOUT)?;
    let mut out = Vec::with_capacity(edges.len());
    for e in edges {
        // Edge source is the reconciliation_task vertex.
        let props = handle.get_vertex_properties(e.from)?;
        if let Some(tid) = props.get("task_id").and_then(json_string) {
            out.push(tid);
        }
    }
    Ok(out)
}

fn json_u64(v: &Json) -> Option<u64> {
    match v {
        Json::Number(n) => n.as_u64(),
        _ => None,
    }
}

fn json_i64(v: &Json) -> Option<i64> {
    match v {
        Json::Number(n) => n.as_i64(),
        _ => None,
    }
}

fn json_string(v: &Json) -> Option<String> {
    match v {
        Json::String(s) => Some(s.clone()),
        _ => None,
    }
}
