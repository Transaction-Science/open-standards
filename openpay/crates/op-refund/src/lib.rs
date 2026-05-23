//! # `op-refund` — refund domain
//!
//! Every payment-acceptance stack needs refunds: the operator
//! initiates a partial or full return of value to a customer after a
//! payment has settled. This crate is the **state machine and
//! durability surface** for that flow — the bookkeeping that says
//! "this refund was requested, sent to the rail, approved, settled."
//!
//! ## What this crate owns
//!
//! - The [`Refund`] domain type and its [`Status`] state machine.
//! - The [`RefundStore`] trait for persistence and the in-memory
//!   reference implementation.
//! - Validation invariants (non-negative amount, terminal-state
//!   guard, idempotency by external id).
//!
//! ## What this crate does NOT own
//!
//! - **The rail call.** Sending the refund instruction to the PSP /
//!   acquirer is the operator's `RailAdapter`'s job, same shape as
//!   the original payment. This crate doesn't know HTTP from gRPC.
//! - **The ledger reversal.** When a refund settles, the operator
//!   posts a reversing [`op_ledger::Transaction`] via their
//!   `LedgerStore`. That coupling — refund id ↔ ledger tx — is the
//!   operator's choice; we don't impose it. Most deployments use
//!   the refund's `external_id` as the reversal tx's `external_id`,
//!   the same convention Phase 19 uses for routing-signal joins.
//! - **The customer notification.** Out of scope. Operator's app.
//!
//! ## Why a separate crate, not part of `op-orchestrator`
//!
//! Refunds are conceptually parallel to payment intents — both are
//! coordinated PSP operations with their own state machines — but
//! they're triggered on a different cadence (days or weeks after
//! the original payment), often through a different operator UI,
//! and they don't need fraud scoring or rail-routing. Keeping them
//! in their own crate matches the codebase's "one concern per
//! crate" pattern (the same reason `op-dispute` lives next door).
//!
//! ## Core invariants
//!
//! 1. **Amount non-negative.** A refund of 0 minor units is legal
//!    (representing "no money moved, status workflow only") because
//!    some PSPs use that for void; negative is rejected.
//! 2. **Terminal states are terminal.** Once a refund is `Settled`,
//!    `Declined`, or `Failed`, no further transitions.
//! 3. **Idempotency by `external_id`.** Posting a refund with an
//!    `external_id` that was already used returns the existing
//!    refund iff the body matches; otherwise
//!    [`Error::IdempotencyMismatch`].
//! 4. **Refund total ≤ original transaction amount.** Enforcement
//!    happens at the store level by summing prior refunds against
//!    the same `original_tx_id`. Out-of-scope for an
//!    `InMemoryRefundStore` test build; documented for production.

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod reason;
pub mod refund;
pub mod store;

pub use error::{Error, Result};
pub use reason::RefundReason;
pub use refund::{Refund, RefundId, Status};
pub use store::{InMemoryRefundStore, RefundStore};
