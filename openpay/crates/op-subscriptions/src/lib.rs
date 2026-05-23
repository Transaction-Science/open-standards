//! # `op-subscriptions` — recurring billing
//!
//! Plans, billing cycles, dunning, proration. The single most-
//! requested feature for an operator-deployable payments stack
//! that we couldn't model until now.
//!
//! ## Architecture
//!
//! - [`Plan`] — the price template: amount, currency, interval,
//!   trial. Snapshotted into each [`Subscription`] at creation so
//!   later plan edits don't silently re-price existing customers.
//! - [`Subscription`] — one customer's recurring billing contract.
//!   Carries the plan, payment method, current period, status.
//! - [`Status`] — `Trialing → Active → PastDue/Canceled/Paused`,
//!   `Active → Canceled` on cancel, etc.
//! - [`BillingScheduler`] — pure-function period math: given a
//!   subscription and "now", returns whether the period has
//!   closed and what the next period bounds are.
//! - [`DunningPolicy`] — backoff schedule for failed charges
//!   before the subscription tips into `Canceled`.
//! - [`proration::credit_remaining`] — exact-integer credit when a
//!   plan changes mid-cycle.
//! - [`SubscriptionStore`] — pluggable storage; [`InMemorySubscriptionStore`]
//!   is the ref impl. A graph-backed impl lives in `op-graph`.
//!
//! ## What this crate does NOT do
//!
//! - **No clock.** All `_unix_secs` fields are caller-supplied;
//!   tests and replay are deterministic.
//! - **No actual charging.** The scheduler emits "due now"
//!   signals; the operator wires the actual charge through the
//!   orchestrator (or whatever rail-attempt layer they prefer).
//! - **No payment-method storage.** Subscriptions reference a
//!   `PaymentMethod` (vault token, etc.); the vault is `op-vault`'s
//!   job.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]

pub mod dunning;
pub mod error;
pub mod plan;
pub mod proration;
pub mod scheduler;
pub mod store;
pub mod subscription;

pub use dunning::{DunningOutcome, DunningPolicy};
pub use error::{Error, Result};
pub use plan::{Interval, Plan, PlanId};
pub use scheduler::{BillingScheduler, DueState};
pub use store::{InMemorySubscriptionStore, SubscriptionStore};
pub use subscription::{Status, Subscription, SubscriptionId};
