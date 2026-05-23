//! # `op-ledger` — Double-entry append-only ledger
//!
//! The source of truth for money movement in OpenPay.
//!
//! ## Architectural place in the stack
//!
//! Earlier crates produce events ("a card auth approved",
//! "a FedNow transfer settled"). [`op-orchestrator`](../op_orchestrator)
//! coordinates them. **`op-ledger` records them**, in a form that
//! survives software bugs, race conditions, and adversarial replays.
//!
//! Without a ledger:
//! - balance state is a [`UPDATE users SET balance = ...`] race
//!   condition waiting to happen,
//! - rebuilding history requires log-grepping the payment service,
//! - reconciling with bank statements is manual archaeology,
//! - regulatory audits are impossible.
//!
//! With a ledger:
//! - balances are derived (`sum of credits − sum of debits`, signed
//!   by [`NormalBalance`]) from immutable entries,
//! - every state transition has a deterministic record,
//! - bank-statement reconciliation is a set difference,
//! - audits are a SQL query.
//!
//! ## Core invariants
//!
//! 1. **Double-entry per currency.** For every [`Transaction`], the
//!    sum of debit amounts equals the sum of credit amounts **per
//!    currency**. A USD-only transaction has at least one debit and
//!    one credit, both USD, equal amounts. A multi-currency
//!    transaction (e.g. FX) has the invariant holding per currency.
//!
//! 2. **Append-only.** Once a transaction is [`Status::Posted`] it
//!    cannot be modified. To correct a posted transaction you post a
//!    **reversal** transaction that explicitly negates it.
//!
//! 3. **Accounts have a fixed currency.** Set at creation, never
//!    changed. Entries against an account must use that currency.
//!
//! 4. **Accounts have a normal balance** (debit-normal or
//!    credit-normal). Asset and Expense accounts are debit-normal —
//!    debits increase them. Liability, Equity, and Revenue accounts
//!    are credit-normal — credits increase them. The
//!    [`Account::balance`] derivation respects this so that "the
//!    balance" is always the natural sign for the account class.
//!
//! 5. **Pending vs posted.** A transaction starts [`Status::Pending`]
//!    (e.g. a card auth has happened but the rail hasn't settled).
//!    Pending transactions can be transitioned to [`Status::Posted`]
//!    (settlement confirmed) or [`Status::Archived`] (transaction
//!    cancelled, e.g. void before capture). Pending balances
//!    include both pending and posted; posted balances include only
//!    posted.
//!
//! 6. **Idempotency by `external_id`.** Posting a transaction with
//!    an `external_id` that has been seen returns the existing
//!    transaction (or a mismatch error if the body differs).
//!
//! ## Verified design choices
//!
//! - **Modern Treasury** ([`docs.moderntreasury.com`](https://docs.moderntreasury.com))
//!   pioneered this exact API model in 2020. We adopt the same
//!   `Ledger / Account / Transaction / Entry / Status` vocabulary
//!   so operators familiar with that surface have a smooth landing.
//! - **`pgledger`** (Paul Gross, 2025) is a pure-PostgreSQL
//!   reference implementation following the same invariants.
//! - **TigerBeetle** is the high-throughput in-cluster reference;
//!   not modeled here because it has its own VSR/consensus story.
//!   The trait surface we expose admits a TigerBeetle-backed
//!   [`LedgerStore`] adapter as future work.
//!
//! ## What this crate does NOT do
//!
//! - **No persistence.** [`InMemoryLedgerStore`] is for tests and
//!   single-process kiosks. Production deployments plug in their
//!   own [`LedgerStore`] backed by Postgres, TigerBeetle, etc.
//! - **No FX conversion.** Multi-currency transactions record both
//!   sides as separate entries with their own currency; the FX rate
//!   is the caller's metadata.
//! - **No GAAP / IFRS classification.** Account classes
//!   ([`AccountClass`]) are informational; we don't generate
//!   trial balances or financial statements (downstream concern).
//! - **No clock.** Transaction `effective_at` is caller-supplied so
//!   tests and replay are deterministic. Production callers pass
//!   `SystemTime::now()`.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod account;
pub mod balance;
pub mod entry;
pub mod error;
pub mod history;
pub mod ledger;
pub mod store;
pub mod transaction;

pub use account::{Account, AccountClass, AccountId, NormalBalance};
pub use balance::Balance;
pub use entry::{Direction, Entry};
pub use error::{Error, Result};
pub use history::LedgerHistory;
pub use ledger::{Ledger, LedgerId};
pub use store::{InMemoryLedgerStore, LedgerStore};
pub use transaction::{Status, Transaction, TransactionId};
