//! # `op-statements` — Merchant statements, reconciliation feeds, fee accounting
//!
//! Every payment system eventually owes the merchant a piece of paper
//! (or its electronic equivalent) that says: here is the gross volume
//! you processed, here is what we kept in fees, here is what we paid
//! out, and here is what survived as your ending balance. That piece
//! of paper is the **statement**.
//!
//! `op-statements` is the lowest common denominator implementation of
//! statement generation and reconciliation feed production, sitting
//! above [`op-ledger`](../op_ledger) (the source of truth for money
//! movement) and consuming the same normalized vocabulary as
//! [`op-reconciliation`](../op_reconciliation) (statement lines).
//!
//! ## Architectural place in the stack
//!
//! ```text
//!   op-ledger ─► op-statements ─► RenderTarget {Pdf,Csv,Json,FixedWidth}
//!                     │
//!                     ├─► iso20022::camt.053 builder
//!                     ├─► bai2 writer
//!                     └─► mt940 writer
//! ```
//!
//! ## What ships in v1
//!
//! - [`Statement`] — the merchant-facing aggregate: period, currency,
//!   gross volume, refunds, chargebacks, fees, payouts, balances.
//! - [`render::RenderTarget`] — pluggable serializer for the four
//!   canonical output shapes ([`render::Pdf`] template-driven,
//!   [`render::Csv`], [`render::Json`], [`render::FixedWidth`]
//!   NACHA-style).
//! - [`iso20022::Camt053Builder`] — emits an ISO 20022 `camt.053`
//!   end-of-day statement XML from the same [`Statement`] aggregate.
//! - [`bai2::Bai2Writer`] — Bank Administration Institute BAI2
//!   transmission file.
//! - [`mt940::Mt940Writer`] — SWIFT MT940 customer statement message
//!   (and MT942 intra-day report).
//! - [`fees::FeeBucket`] / [`fees::FeeLine`] — typed fee accrual
//!   (interchange, scheme, acquirer, FX, settlement-network, BNPL).
//! - [`reconcile::Reconciler`] — pairs statement lines against
//!   [`reconcile::LedgerEntry`] rows (a structural copy of the op-ledger
//!   contract; we don't depend on op-ledger directly to keep this
//!   crate's dependency graph minimal).
//! - [`cadence::Cadence`] — daily/weekly/monthly/custom schedules with
//!   deterministic period enumeration.
//!
//! ## What this crate does NOT do
//!
//! - **No real PDF binary emission.** [`render::Pdf`] produces a
//!   template-driven plain-text rendering of the statement
//!   ("template-driven PDF" in the OpenPay sense means: a structured
//!   text artifact a downstream PDF library can consume). Embedding a
//!   PDF stack would bloat the dependency graph and tie us to one
//!   licence model.
//! - **No clock.** All cadence enumeration is caller-supplied unix
//!   epoch seconds. Tests and replay stay deterministic.
//! - **No persistence.** Statements are values; the operator picks the
//!   store.
//! - **No FX rate sourcing.** Multi-currency aggregation carries
//!   per-currency totals and an FX-adjusted view only when the caller
//!   supplies the rate.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod bai2;
pub mod cadence;
pub mod error;
pub mod fees;
pub mod iso20022;
pub mod mt940;
pub mod reconcile;
pub mod render;
pub mod statement;

pub use bai2::Bai2Writer;
pub use cadence::{Cadence, Period};
pub use error::{Error, Result};
pub use fees::{FeeBucket, FeeLine, FeeRule, FeeSchedule};
pub use iso20022::Camt053Builder;
pub use mt940::Mt940Writer;
pub use reconcile::{LedgerEntry, ReconRecord, Reconciler};
pub use render::{Csv, FixedWidth, Json, Pdf, RenderTarget};
pub use statement::{
    BalanceSnapshot, CurrencyAggregate, FxRate, Statement, StatementLine, StatementLineKind,
};
