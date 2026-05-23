//! # `op-reconciliation` — ledger vs. statement diffing
//!
//! Reconciliation answers one question: **does what the bank/PSP says
//! happened agree with what our ledger says happened?** Every payment
//! stack needs this because the two records drift:
//!
//! - A rail confirms settlement but the webhook never arrives, so the
//!   ledger never posts.
//! - A chargeback debits the merchant account at the bank but no
//!   reversal is booked.
//! - A PSP fee is netted out of a payout and the gross/fee split was
//!   recorded wrong.
//!
//! Without reconciliation these are invisible until an auditor or an
//! angry customer finds them. With it, each one becomes a typed
//! [`Discrepancy`](discrepancy::Discrepancy).
//!
//! ## Inputs are pluggable
//!
//! A [`ReconciliationSource`](source::ReconciliationSource) yields a
//! stream of normalized [`StatementLine`](statement::StatementLine)s.
//! Three reference sources ship:
//!
//! - [`Camt053Source`](sources::Camt053Source) — end-of-day bank
//!   statement (ISO 20022 `camt.053`). The authoritative artifact.
//! - [`Camt054Source`](sources::Camt054Source) — intra-day
//!   debit/credit notifications (`camt.054`).
//! - [`WebhookEventSource`](sources::WebhookEventSource) — the
//!   settlement webhooks already flowing through `op-webhook`.
//!
//! Operators with a proprietary PSP CSV implement the trait
//! themselves; the rest of the engine doesn't change. This mirrors
//! the `LedgerStore` / `WebhookStore` pluggable-driver idiom used
//! elsewhere in `OpenPay`.
//!
//! ## The ledger side is caller-selected
//!
//! [`op_ledger::LedgerStore`] deliberately exposes no "list all
//! transactions" method (a real store has millions of rows; an
//! unbounded scan is a footgun). So the [`Reconciler`](engine::Reconciler)
//! takes the ledger window as an explicit `&[Transaction]` slice the
//! caller assembled — by date range, by ledger, however their store
//! indexes. We never impose an iteration API on `op-ledger`.
//!
//! ## Outputs: both a report and (optionally) graph vertices
//!
//! [`Reconciler::reconcile`](engine::Reconciler::reconcile) returns a
//! serializable [`ReconciliationReport`](discrepancy::ReconciliationReport)
//! — matched counts plus a `Vec<Discrepancy>` an operator can feed
//! into any ticketing system. Materializing discrepancies as graph
//! vertices (so an operator can traverse from a ledger account to its
//! open reconciliation tasks) lives in `op-graph` behind a trait, so
//! this crate stays free of the MPL-2.0 `indradb` dependency.
//!
//! ## What this crate does NOT do
//!
//! - **No currency conversion.** A line in EUR against a USD tx is an
//!   `AmountMismatch`, not an FX calculation.
//! - **No bipartite optimal matching.** v1 is a deterministic two-tier
//!   join (external-id, then amount+window heuristic). Optimal
//!   assignment is future work.
//! - **No clock.** Reconciliation windows are caller-supplied so
//!   replay is deterministic.
//! - **No auto-resolution.** We detect and classify; booking the
//!   correcting entry is an operator decision.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod discrepancy;
pub mod engine;
pub mod error;
pub mod matcher;
pub mod source;
pub mod sources;
pub mod statement;
pub mod store;

pub use discrepancy::{Discrepancy, ReconciliationReport, TaskDescriptor};
pub use engine::Reconciler;
pub use error::{Error, Result};
pub use matcher::MatchedPair;
pub use source::ReconciliationSource;
pub use statement::{LineDirection, StatementLine};
pub use store::{ReconciliationStore, ReconciliationTask};
