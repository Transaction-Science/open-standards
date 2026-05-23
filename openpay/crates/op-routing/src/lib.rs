//! # `op-routing`
//!
//! Advanced routing primitives for `OpenPay`. The orchestrator's
//! [`PolicyRouter`](https://docs.rs/op-orchestrator) handles static
//! priority-list routing. To be a complete payment stack, `OpenPay`
//! also needs the three capabilities every commercial competitor
//! ships:
//!
//! 1. **Least-cost routing (LCR)** — given an intent, estimate the
//!    landed cost on each candidate route (interchange + scheme
//!    fees + PSP markup + fixed-per-tx) and prefer the cheapest.
//!    See [`lcr`].
//!
//! 2. **MCC-aware routing** — different PSPs have markedly different
//!    auth rates and acceptance profiles by ISO 18245 Merchant
//!    Category Code. Grocery (`5411`) routes well through
//!    different processors than digital goods (`6051`) or gambling
//!    (`7995`). See [`mcc`].
//!
//! 3. **Intelligent retry / soft-decline recovery** — ISO 8583 soft
//!    declines (`05`, `51`, `91`, `96`, ...) are recoverable on an
//!    alternate rail within seconds; hard declines (`04`, `43`,
//!    `41`, `62`, `59`, ...) must never retry. See [`retry`].
//!
//! These compose through [`ComposedRouter`](composer::ComposedRouter):
//! MCC narrows the route pool → LCR sorts the pool by estimated
//! cost → retry policy decides whether we're on first attempt or in
//! recovery.
//!
//! ## Route shape
//!
//! `op-orchestrator` has its own `RailChoice` (`{ rail, driver }`).
//! This crate works at a slightly richer level — a [`Route`] also
//! carries the destination country (relevant for interchange tier
//! tables) and an optional MCC-acceptance tag. Operators bridge the
//! two by mapping `Route` → `RailChoice` after `ComposedRouter`
//! returns its decision. The gap is intentional: cost / MCC / retry
//! reasoning is a strict superset of "which (rail, driver) tuple do
//! I try next" so we don't force orchestrator-level changes on
//! anyone who only needs the static router.
//!
//! ## Determinism
//!
//! Pure compute. Same input → same output. The retry module's
//! exponential-with-jitter backoff exposes the jitter seed so test
//! and production traces are reproducible. No I/O, no clock reads
//! beyond what the caller passes in.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
// Pedantic lints we intentionally tolerate for this crate:
// - `must_use_candidate`: ubiquitous for builder-style APIs; the
//   crate explicitly marks the load-bearing builders.
// - `duration_suboptimal_units`: cargo-1.95 nags about
//   `from_secs(60)` vs `from_mins(1)`; the former is the conventional
//   choice for retry-window literals.
// - `should_implement_trait`: `Mcc::from_str` / `DeclineCode::from_str`
//   intentionally return `Option`, not `Result<_, FromStrErr>`, so
//   the `FromStr` trait shape is wrong here.
// - `elidable_lifetime_names`: a few signatures are clearer with
//   named lifetimes for documentation.
#![allow(clippy::must_use_candidate)]
#![allow(clippy::duration_suboptimal_units)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::too_long_first_doc_paragraph)]

pub mod composer;
pub mod cost;
pub mod lcr;
pub mod mcc;
pub mod retry;
pub mod route;

pub use composer::{ComposedRouter, RouteDecision};
pub use cost::{
    BlendedRateEstimator, Bps, CostEstimator, CostModel, InterchangePlusEstimator,
    TieredFixedEstimator,
};
pub use lcr::LeastCostRouter;
pub use mcc::{Mcc, McuPreferences, MccPolicy, mcc_catalogue};
pub use retry::{
    Attempt, BackoffPolicy, DeclineCategory, DeclineCode, IntelligentRetry, default_hard_declines,
    default_soft_declines,
};
pub use route::{DriverId, Route};
