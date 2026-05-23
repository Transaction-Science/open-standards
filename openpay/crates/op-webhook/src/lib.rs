//! # `op-webhook` — Outbound webhook delivery
//!
//! The crate that turns ledger events (and any other internal
//! event-shaped data) into reliably-delivered HTTP POSTs to
//! merchant-configured endpoints. This is the **fanout** edge of
//! the OpenPay stack: everything upstream produces facts; this
//! crate signs them, retries them, and surfaces dead letters.
//!
//! ## Architectural place in the stack
//!
//! ```text
//!  op-orchestrator  ─►  op-ledger  ─►  op-webhook  ─►  merchant ERP
//!   (run intent)        (record)        (notify)        (act)
//! ```
//!
//! The crate is intentionally **decoupled** from `op-ledger` and
//! `op-orchestrator`: it takes opaque `Vec<u8>` payloads. Operators
//! choose the wire format (JSON, CBOR, MessagePack — anything). The
//! signature scheme is Stripe-compatible
//! (`t={ts},v1={hmac_sha256_hex(secret, "{ts}.{body}")}`) so any
//! merchant who has already integrated with Stripe receives webhooks
//! they can verify with the same code.
//!
//! ## Core invariants
//!
//! 1. **At-least-once delivery.** A subscribed endpoint is
//!    guaranteed to be **attempted** at least once. Network failures
//!    queue retries; consumers must dedupe via `event_id`.
//! 2. **Exponential backoff with full jitter.** Failed deliveries
//!    are rescheduled at `min(base * 2^n, max_delay)` modulated by
//!    full jitter (`rand_uniform(0, computed_delay)`), capped at a
//!    configurable max attempts and max age (Stripe's 72-hour
//!    window by default).
//! 3. **Stripe-compatible signing.** Header
//!    `OpenPay-Signature: t={unix_secs},v1={hex_hmac}` carries the
//!    signature; the signed payload is `"{ts}.{body}"`; the HMAC
//!    key is the endpoint's secret. Constant-time comparison
//!    (`subtle::ConstantTimeEq`) is used for verification.
//! 4. **Auto-disable on chronic failure.** After
//!    `disable_after_consecutive_failures` consecutive delivery
//!    failures (Stripe: signal-via-email, Razorpay: 24h then
//!    disable; we default to 10 failures), the endpoint's status
//!    flips to [`EndpointStatus::Disabled`] and no further attempts
//!    are made until an operator manually re-enables it.
//! 5. **Replay is explicit.** Operators trigger replay via
//!    [`WebhookDispatcher::replay`]; the system never silently
//!    resurrects events.
//! 6. **Pluggable HTTP transport.** The crate ships with no real
//!    HTTP client; operators wire `reqwest`, `hyper`, `ureq`, or
//!    anything else that implements [`HttpTransport`].
//! 7. **Pluggable storage.** [`WebhookStore`] trait;
//!    [`InMemoryWebhookStore`] reference impl for tests and
//!    single-process kiosks.
//!
//! ## What this crate does NOT do
//!
//! - **No HTTP client.** No `reqwest`, no `hyper`. The
//!   [`HttpTransport`] trait is the seam.
//! - **No async runtime.** All operations are sync. Async wrappers
//!   are operator-level (`tokio::task::spawn_blocking` or a
//!   long-running thread).
//! - **No durable queue.** [`InMemoryWebhookStore`] loses state on
//!   process exit. Production deployments plug in their own.
//! - **No CloudEvents formatting.** The payload is opaque; encode
//!   however you like.
//! - **No key rotation flow.** A single secret per endpoint. Adding
//!   a "previous secret" field for rotation is documented as a
//!   future extension.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod dispatcher;
pub mod emitter;
pub mod endpoint;
pub mod error;
pub mod event;
pub mod hexutil;
#[cfg(feature = "reqwest-transport")]
pub mod reqwest_transport;
pub mod retry;
pub mod signing;
pub mod store;
pub mod transport;

pub use dispatcher::{DispatchOutcome, WebhookDispatcher};
pub use emitter::{EventEmitter, NoOpEmitter, WebhookEmitter};
pub use endpoint::{Endpoint, EndpointId, EndpointStatus};
pub use error::{Error, Result};
pub use event::{DeliveryAttempt, DeliveryAttemptId, DeliveryStatus, WebhookEvent, WebhookEventId};
#[cfg(feature = "reqwest-transport")]
pub use reqwest_transport::ReqwestTransport;
pub use retry::{ExponentialBackoffPolicy, RetryPolicy, jitter_full};
pub use signing::{SIGNATURE_HEADER, SignedPayload, compute_signature, verify_signature};
pub use store::{InMemoryWebhookStore, WebhookStore};
pub use transport::{HttpRequest, HttpResponse, HttpTransport, MockTransport};
