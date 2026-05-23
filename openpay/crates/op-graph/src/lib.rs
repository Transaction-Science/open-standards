//! # `op-graph` — Graph-backed stores for OpenPay
//!
//! OpenPay's data is graph-shaped:
//!
//! - Every **ledger transaction** is a hyperedge that connects 2+
//!   accounts. The "give me every account touched by transaction T"
//!   question is a one-hop traversal on a graph; in a relational
//!   model it's a join.
//! - Every **webhook delivery attempt** is an edge from an event
//!   vertex to an endpoint vertex. "Show me every attempt for
//!   event E across all endpoints" is also one hop.
//! - **Reversal chains** are a directed sub-graph: tx → reverses →
//!   tx → reverses → tx. Walking the chain is graph traversal,
//!   not recursive CTE.
//! - **Fraud queries** ("which endpoints share a webhook secret
//!   prefix," "which accounts are linked by reversal patterns")
//!   are graph queries by nature.
//!
//! This crate provides:
//!
//! 1. A thin facade over IndraDB ([`GraphHandle`]) that does the
//!    typing and JSON-property bookkeeping IndraDB doesn't enforce.
//! 2. [`GraphLedgerStore`] — an implementation of
//!    [`op_ledger::LedgerStore`] that persists to a graph.
//! 3. [`GraphWebhookStore`] — an implementation of
//!    [`op_webhook::WebhookStore`] that persists to a graph.
//! 4. [`queries`] — opinionated read-side helpers for the
//!    OpenPay-relevant graph traversals: "trace this reversal
//!    chain," "list every attempt for this event," "list every
//!    transaction that touched this account."
//!
//! ## Vertex / edge schema
//!
//! ```text
//!  ledger:account ─┐
//!                  │
//!                  │ ledger:debit │ ledger:credit
//!                  ▼
//!  ledger:tx ──reverses──► ledger:tx
//!                  │
//!                  ▼
//!  webhook:event ──delivers──► webhook:attempt ──to──► webhook:endpoint
//! ```
//!
//! Vertex types (all created via `Identifier::new`):
//!
//! - `ledger_account` — `currency`, `name`, `class`, `normal_balance`,
//!   `external_id` (optional)
//! - `ledger_tx` — `status`, `external_id` (optional),
//!   `description` (optional), `effective_at_unix_secs`,
//!   `ledger_id`
//! - `ledger_ledger` — `name`, `description` (optional)
//! - `webhook_event` — `event_type`, `payload_b64`,
//!   `created_at_unix_secs`
//! - `webhook_endpoint` — `url`, `secret_b64`, `event_filters_csv`,
//!   `status`, `consecutive_failures`
//! - `webhook_attempt` — `attempt_number`, `status`, `http_status`
//!   (optional), `started_at_unix_secs`, `completed_at_unix_secs`
//!   (optional), `next_attempt_at_unix_secs` (optional),
//!   `response_body_excerpt` (optional), `error` (optional)
//!
//! Edge types:
//!
//! - `ledger_debit`: `ledger_tx` → `ledger_account`. Property:
//!   `amount_minor`, `currency`.
//! - `ledger_credit`: `ledger_tx` → `ledger_account`. Property:
//!   `amount_minor`, `currency`.
//! - `ledger_in_ledger`: `ledger_tx` → `ledger_ledger`,
//!   `ledger_account` → `ledger_ledger`.
//! - `ledger_reverses`: `ledger_tx` → `ledger_tx` (the original).
//! - `webhook_delivers`: `webhook_event` → `webhook_attempt`.
//! - `webhook_to`: `webhook_attempt` → `webhook_endpoint`.
//!
//! ## License compatibility note
//!
//! IndraDB is **MPL-2.0**. OpenPay is Apache-2.0. The MPL-2.0
//! license has file-level copyleft — modifications to IndraDB
//! source files must remain MPL-2.0 — but **linking** against
//! IndraDB from an Apache-2.0 crate is permitted without
//! relicensing the consuming crate. This is the same model as
//! many Firefox-embedded applications. Operators who prefer a
//! single-license deployment can substitute a different
//! `LedgerStore` / `WebhookStore` backend (the trait surfaces
//! exist precisely for this).
//!
//! ## What this crate does NOT do
//!
//! - **No graph query language exposed to operators.** We expose
//!   typed Rust helpers, not Cypher / GQL / Gremlin. IndraDB
//!   itself doesn't have those either.
//! - **No persistence by default.** Constructed with an in-memory
//!   IndraDB datastore. Operators wanting durability call
//!   `MemoryDatastore::new_db_with_path` and use IndraDB's
//!   `Sync()` (see IndraDB docs).
//! - **No transactionality across LedgerStore + WebhookStore.**
//!   IndraDB's transaction guarantees are per-operation; cross-
//!   store atomicity is the operator's concern.
//! - **No streaming subscriptions.** Operators poll or run
//!   periodic queries.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod audit;
pub mod dispute_store;
pub mod error;
pub mod graph;
pub mod idempotency_store;
pub mod ledger_store;
pub mod queries;
pub mod rail_telemetry;
pub mod reconciliation_store;
pub mod refund_store;
pub mod settlement_store;
pub mod subscription_store;
pub mod webhook_store;

pub use audit::{AuditEntry, AuditReport};
pub use dispute_store::GraphDisputeStore;
pub use error::{Error, Result};
pub use graph::GraphHandle;
pub use idempotency_store::GraphIdempotencyStore;
pub use ledger_store::GraphLedgerStore;
pub use queries::{
    accounts_linked_via_chargeback, accounts_touched_by_transaction, attempts_for_event,
    attempts_with_shared_ip, endpoints_sharing_secret_prefix, reversal_chain,
    transactions_touching_account,
};
pub use rail_telemetry::{
    DEFAULT_LATENCY_SLO_MS, DEFAULT_WINDOW_SECS, DiscrepancyWeights, GraphRailTelemetry,
    RailAttemptRecord,
};
pub use reconciliation_store::GraphReconciliationStore;
pub use refund_store::GraphRefundStore;
pub use settlement_store::GraphSettlementStore;
pub use subscription_store::GraphSubscriptionStore;
pub use webhook_store::GraphWebhookStore;
