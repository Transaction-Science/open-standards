//! # `op-server` — HTTP API surface for `OpenPay`
//!
//! An [`axum`]-based HTTP server that exposes the `OpenPay`
//! workspace as a deployable REST API. Operators run this binary;
//! merchants POST JSON. The crate also exports the [`Router`]
//! and [`AppState`] types so embedders can mount the routes on a
//! larger application or run them in-process for testing.
//!
//! ## Endpoint surface
//!
//! | Path | Verb | What it does |
//! |------|------|-------------|
//! | `/health` | GET | Liveness — always 200 |
//! | `/readiness` | GET | Readiness — verifies the in-memory stores answer |
//! | `/v1/intents` | POST | Create + run a payment intent through the orchestrator |
//! | `/v1/refunds` | POST | Create a refund (idempotent on `external_id`) |
//! | `/v1/refunds/{id}` | GET | Fetch a refund |
//! | `/v1/refunds/{id}:submit` | POST | Transition `Requested → Submitted` |
//! | `/v1/refunds/{id}:settle` | POST | Transition `Approved → Settled` |
//! | `/v1/disputes` | POST | Create a dispute |
//! | `/v1/disputes/{id}` | GET | Fetch a dispute |
//! | `/v1/disputes/{id}/evidence` | POST | Attach evidence |
//! | `/v1/settlement/batches` | POST | Open a new batch |
//! | `/v1/settlement/batches/{id}` | GET | Fetch a batch |
//! | `/v1/settlement/batches/{id}:close` | POST | Close + apply holdback |
//! | `/v1/audit/report` | GET | Multi-store audit report for a tx-count window |
//!
//! ## What this crate does NOT do
//!
//! - **No auth.** Apply auth via a `tower::Layer`. The binary
//!   accepts an optional `--api-key` env arg and rejects requests
//!   missing it, but that is a hint, not real authentication.
//! - **No persistence.** State holds in-memory stores by default.
//!   Operators replace the [`AppState::new_in_memory`] constructor
//!   with their backend-specific equivalent.
//! - **No TLS.** Terminate TLS in front (Caddy / nginx / a load
//!   balancer). We bind plain HTTP.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

pub mod auth;
pub mod config;
pub mod error;
pub mod events;
pub mod handlers;
pub mod rate_limit;
pub mod routes;
pub mod state;

pub use auth::{ApiKeyAuthLayer, ApiKeyAuthService};
pub use config::{EnvMiddleware, build_middleware_from_env, build_state_from_env};
pub use error::{ApiError, ApiResult};
pub use rate_limit::{RateLimitLayer, RateLimitService};
pub use routes::{router, router_with_middleware};
pub use state::AppState;
