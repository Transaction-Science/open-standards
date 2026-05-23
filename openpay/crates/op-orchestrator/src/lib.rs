//! # `op-orchestrator` — Cross-rail payment orchestration
//!
//! Coordinates the OpenPay stack into an end-to-end flow:
//!
//! ```text
//!  PaymentIntent
//!       │
//!       ▼
//!  ┌──────────────────────────────────────────┐
//!  │ 1. Idempotency check (in-process cache)  │
//!  ├──────────────────────────────────────────┤
//!  │ 2. Fraud scoring (op-fraud)              │
//!  ├──────────────────────────────────────────┤
//!  │ 3. Routing (Router trait)                │
//!  │      decide rail order and retry budget  │
//!  ├──────────────────────────────────────────┤
//!  │ 4. Attempt loop                          │
//!  │      • call rail (op-rails-card / a2a)   │
//!  │      • classify outcome (Final/Retry/    │
//!  │        Fallback)                         │
//!  │      • on Fallback: next rail w/ SAME    │
//!  │        idempotency key                   │
//!  │      • on Retry: backoff + same rail     │
//!  │      • circuit breaker per (rail, psp)   │
//!  ├──────────────────────────────────────────┤
//!  │ 5. Persist terminal state                │
//!  │      future-replay of the same intent    │
//!  │      key returns the same outcome        │
//!  └──────────────────────────────────────────┘
//!       │
//!       ▼
//!  OrchestrationOutcome
//! ```
//!
//! ## Why this lives outside the rail crates
//!
//! Each rail crate (`op-rails-card`, `op-rails-a2a`) knows how to
//! talk to its own backends. They do **not** know about cross-rail
//! fallback, fraud gating, or idempotency. Mixing those concerns
//! into the rail crates would create a circular dependency (fraud
//! needs to score a generic payment, not a card-specific one) and
//! would prevent operators from swapping in their own rail drivers
//! while reusing the orchestrator.
//!
//! ## Design rationale verified against industry practice
//!
//! - **Idempotency keys (UUIDv4/v7)** — Adyen / Stripe convention.
//!   Same key carries across retries AND rail fallbacks. 7-day
//!   minimum TTL per Adyen docs.
//! - **State machine** — payment is a workflow, not a record.
//!   Stripe's interview prompt; valid transitions enforced.
//! - **Exponential backoff with jitter** — industry standard for
//!   non-idempotent operations after timeouts.
//! - **Circuit breaker** — 5 consecutive failures → open for 60s
//!   cooldown → half-open probe → close on success. Standard
//!   resilience pattern.
//! - **No floats for money** — already enforced by `op-core::Money`
//!   (i64 minor units).
//!
//! ## What this crate does NOT do
//!
//! - **No async runtime.** The orchestrator is sync, like the
//!   acquirer traits it sits on top of. Async is an
//!   integration concern; wrap with `tokio::task::spawn_blocking`
//!   if needed.
//! - **No persistent idempotency store.** The default
//!   [`InMemoryIdempotencyStore`] is for tests and short-lived
//!   processes. Production deployments plug in their own
//!   [`IdempotencyStore`] backed by Redis / Postgres / Spanner.
//! - **No webhook fanout.** Terminal outcomes are returned
//!   synchronously; emitting webhooks is the caller's job.
//! - **No ledger.** The orchestrator returns outcomes; persisting
//!   them into a double-entry ledger is downstream.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod adapters;
pub mod circuit_breaker;
pub mod engine;
pub mod error;
pub mod idempotency;
pub mod intent;
pub mod outcome;
pub mod router;
pub mod signals;
pub mod telemetry;

pub use adapters::{A2aAdapter, CardAdapter, CryptoAdapter, MerchantBankProfile};
pub use circuit_breaker::{CircuitBreaker, CircuitState, InMemoryCircuitBreaker};
pub use engine::{BackoffPolicy, Orchestrator, OrchestratorConfig};
pub use error::{Error, Result};
pub use idempotency::{
    IdempotencyKey, IdempotencyRecord, IdempotencyStore, InMemoryIdempotencyStore,
};
pub use intent::{PaymentIntent, RoutingHints};
pub use outcome::{Attempt, AttemptOutcome, OrchestrationOutcome, TerminalStatus};
pub use router::{PolicyRouter, RailChoice, Router, RoutingDecision};
pub use signals::{
    AttemptResultClass, NoOpRailTelemetry, NoOpRoutingSignals, RailTelemetry, RoutingSignals,
    SignalCombiner, noop_signals, noop_telemetry,
};
