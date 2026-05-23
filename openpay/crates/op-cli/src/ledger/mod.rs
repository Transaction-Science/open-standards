//! `op ledger ...` — bi-temporal time-travel queries against the
//! `OpenPay` ledger.
//!
//! The ledger is append-only and stored on a bi-temporal substrate
//! ([`op_graph::GraphLedgerStore`] over Minigraf's EAV fact log).
//! Every account balance and transaction status carries two clocks:
//!
//! - **Valid time** — when the event happened in the real world
//!   (`effective_at_unix_secs` on every [`op_ledger::Transaction`]).
//! - **Transaction time** — when the system learned about it
//!   (the monotonic `tx_count` advanced by the underlying store on
//!   every write).
//!
//! [`as_of`] exposes both axes to operators. The CLI accepts ISO 8601
//! timestamps on each axis and returns the ledger state as it stood
//! at those coordinates. This is the "lens" half of the issue: the
//! substrate already retains the facts, the lens just makes them
//! inspectable from a terminal.
//!
//! ## Local-only by design
//!
//! Unlike the rest of [`crate`] (which speaks HTTP to a running
//! [`op_server`](https://docs.rs/op-server) deployment), the `ledger
//! as-of` subcommand runs the query in-process against a
//! [`GraphLedgerStore`](op_graph::GraphLedgerStore). v1 ships a
//! deterministic demo seed so the command works out of the box; a
//! follow-up issue will wire it to a server endpoint backed by the
//! operator's production graph.

pub mod as_of;

pub use as_of::{AsOfArgs, Output, run as run_as_of};
