//! # `op-driver-sdk` — author your own `OpenPay` rail driver
//!
//! `OpenPay`'s `RailAdapter` / `CardAcquirer` / `A2aAcquirer` traits
//! are deliberately the *only* extensibility surface operators
//! need: write one impl, register it with the orchestrator, and
//! the rest of the stack (routing, telemetry, audit, settlement)
//! treats your driver identically to the reference ones.
//!
//! This crate gives driver authors two things:
//!
//! 1. **Deterministic mocks.** [`DeterministicCardAcquirer`] and
//!    [`DeterministicA2aGateway`] are programmable, side-effect-
//!    free implementations of the rail traits. Operators use them
//!    in their own test suites to build up a working `OpenPay`
//!    deployment without live PSP credentials, and driver authors
//!    use them as reference behaviors during development.
//!
//! 2. **A conformance harness.** [`conformance::run_card`] /
//!    [`conformance::run_a2a`] drive any acquirer impl through a
//!    battery of contract checks: idempotency-key propagation,
//!    no-panic on transport errors, status taxonomy coverage,
//!    `supports()` honesty, attempt-number determinism. A failing
//!    check returns a [`ConformanceFailure`] enumerating exactly
//!    what the driver got wrong.
//!
//! ## The driver-author flow
//!
//! ```text
//!   1. impl CardAcquirer for MyPspClient { ... }
//!   2. Write at least one happy-path test against your real PSP
//!      (sandbox mode is fine).
//!   3. Run op_driver_sdk::conformance::run_card(&my_client)?
//!      to catch the structural bugs.
//!   4. Wrap your acquirer in op_orchestrator::CardAdapter and
//!      register with the orchestrator.
//! ```
//!
//! ## What the conformance harness covers
//!
//! | Check | Why |
//! |---|---|
//! | Auth response carries a non-empty `psp_payment_id` | Capture / refund / void all key off it |
//! | The driver returns *some* status for every (sandbox) input | Drivers must never panic on unknown PSP responses |
//! | `supports()` is consistent with what `authorize()` accepts | Misclassification breaks routing |
//! | Idempotency key in the request equals the value the PSP saw | Required for safe retries |
//! | `attempt_number` parameter is honored (no caching of attempt 0 for attempt 1) | The orchestrator relies on attempt-aware behavior for soft-retry / 3DS-on-retry flows |
//!
//! ## What it does NOT cover
//!
//! - **Live PSP behavior.** That's the driver author's
//!   integration-test responsibility.
//! - **Performance.** Conformance is correctness, not throughput.
//! - **PCI compliance.** Drivers handling raw PAN have their own
//!   regulatory burden; we test the data-flow contract, not the
//!   secure-storage contract.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_panics_doc)]

pub mod a2a;
pub mod card;
pub mod conformance;
pub mod crypto;

pub use a2a::DeterministicA2aGateway;
pub use card::DeterministicCardAcquirer;
pub use conformance::{ConformanceFailure, ConformanceReport};
pub use crypto::DeterministicCryptoGateway;
