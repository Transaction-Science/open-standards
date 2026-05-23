//! # `op-settlement` — When does the vendor actually see the money?
//!
//! Posted ledger transactions are an accounting fact, not a bank
//! deposit. Between "we recorded a payment" and "funds landed in the
//! vendor's bank account" sits the **settlement layer**:
//!
//! 1. Batch posted transactions by **cutoff window** (daily,
//!    multi-daily, manual).
//! 2. Subtract **holdbacks** — operator-configured reserve plus an
//!    optional risk-driven adjustment for open disputes.
//! 3. Generate the **payout file** for the chosen rail (NACHA for
//!    US ACH, `pacs.008` for SEPA / RTP / `FedNow` via
//!    [`op_iso20022`]).
//! 4. Track the batch lifecycle: `Open → Closed → Paying → Paid`
//!    (or `Failed`), with retries on the rail side.
//!
//! ## Architectural place
//!
//! `op-settlement` sits *downstream* of [`op_ledger`] (it consumes
//! posted transactions) and *upstream* of the rail payout mechanism
//! (NACHA file, ISO 20022 customer credit transfer). It does NOT
//! itself talk to a bank — that's the operator's payout adapter,
//! which the trait surface admits.
//!
//! ```text
//!   op-ledger  ──posted tx──►  op-settlement  ──batch──►  op-iso20022
//!                                    │                    (pacs.008)
//!                                    └─────────────►  NACHA writer
//! ```
//!
//! ## What this crate does NOT do
//!
//! - **No bank connectivity.** We emit payout *files*; an operator
//!   adapter SFTPs the NACHA to their ODFI or POSTs the pacs.008.
//! - **No FX.** A batch is single-currency. Multi-currency
//!   operators run one batch per currency.
//! - **No persistence by default.** [`InMemorySettlementStore`] for
//!   tests / single-process kiosks; production deployments plug in
//!   their own [`SettlementStore`].
//! - **No clock.** All timestamps are caller-supplied unix epoch
//!   seconds so tests and replay are deterministic.

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod batch;
pub mod cutoff;
pub mod engine;
pub mod error;
pub mod holdback;
pub mod nacha;
pub mod payout;
pub mod store;

pub use batch::{Batch, BatchId, Status};
pub use cutoff::Cutoff;
pub use engine::SettlementEngine;
pub use error::{Error, Result};
pub use holdback::{Holdback, HoldbackPolicy};
pub use nacha::{NachaCredit, nacha_file};
pub use payout::PayoutRail;
pub use store::{InMemorySettlementStore, SettlementStore};
