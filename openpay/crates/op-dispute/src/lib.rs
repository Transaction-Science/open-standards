//! # `op-dispute` — chargeback / dispute domain
//!
//! Where [`op_refund`](op_refund) is operator-initiated reversal,
//! this crate is **bank-initiated** reversal: a chargeback. A
//! customer's issuing bank files a dispute, funds get clawed back
//! from the merchant, and a status workflow follows that ends in
//! the merchant either accepting the loss or representing evidence
//! to win it back.
//!
//! ## What this crate owns
//!
//! - The [`Dispute`] domain type and its [`Status`] workflow.
//! - [`DisputeReason`] (the card-network reason-code taxonomy plus
//!   A2A-rail equivalents).
//! - The [`DisputeStore`] trait and an in-memory reference impl.
//! - Validation invariants (terminal-state guard, non-negative
//!   amount, idempotency by external id).
//!
//! ## What this crate does NOT own
//!
//! - **The rail-side workflow.** Submitting evidence to a PSP,
//!   filing pre-arbitration, etc. — all out of scope; operators
//!   wire that through their own adapter.
//! - **The ledger reversal.** A chargeback that the merchant
//!   loses produces a reversing ledger transaction (same way a
//!   settled refund does). The operator posts that via their
//!   `LedgerStore`. We track the dispute side of the linkage; we
//!   don't double-book it.
//!
//! ## Why a separate crate from `op-refund`
//!
//! Refunds and disputes share the "reversal of a settled payment"
//! shape but the operator semantics differ enough to keep them
//! apart:
//!
//! - **Refund**: operator-initiated, fast, no evidence flow.
//! - **Dispute**: bank-initiated, multi-week, evidence-driven,
//!   chargeback fees attached.
//!
//! Conflating them into one type forces every consumer to thread
//! awkward null-defaults through the unused fields. Same reasoning
//! that drove `op_ledger::Direction` to be distinct from
//! `op_reconciliation::LineDirection`.

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod dispute;
pub mod error;
pub mod evidence;
pub mod reason;
pub mod store;

pub use dispute::{Dispute, DisputeId, Status};
pub use error::{Error, Result};
pub use evidence::EvidenceRef;
pub use reason::DisputeReason;
pub use store::{DisputeStore, InMemoryDisputeStore};
