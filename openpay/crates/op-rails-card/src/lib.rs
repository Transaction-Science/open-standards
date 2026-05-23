//! # `op-rails-card`
//!
//! Pluggable card-rail adapters. Every PSP — Hyperswitch, Stripe, Adyen,
//! Finix, Moov — implements the [`CardAcquirer`] trait. The orchestrator
//! holds a `Box<dyn CardAcquirer>` and never knows which one it has.
//!
//! ## Lifecycle mapping
//!
//! The PSP's payment-intent lifecycle and our [`op_core::Payment<S>`]
//! typestate machine line up cleanly:
//!
//! | `OpenPay` state | PSP lifecycle step                       | Driver call         |
//! |---------------|------------------------------------------|---------------------|
//! | `Created`     | Create payment intent                    | [`CardAcquirer::create`]   |
//! | `Authorized`  | Confirm (manual capture)                 | [`CardAcquirer::authorize`] |
//! | `Captured`    | Capture (or auto-capture on confirm)     | [`CardAcquirer::capture`]   |
//! | `Voided`      | Cancel a pre-capture authorization       | [`CardAcquirer::void`]      |
//! | `Refunded`    | Refund a captured payment                | [`CardAcquirer::refund`]    |
//! | `Failed`      | Any of the above returned a hard decline | — (read in response)   |
//!
//! ## Tap-to-Pay flow
//!
//! When the merchant device produces an EMV TLV blob from `ProximityReader`
//! (iOS) or `IsoDep` (Android), the driver wraps it in the PSP's
//! card-present payload format. Hyperswitch's V1 API doesn't yet
//! standardize EMV TLV; we pass it through `connector_metadata` until the
//! V2 surface lands.
//!
//! ## No raw PAN
//!
//! Drivers never see raw card numbers. The only `op_core::PaymentMethod`
//! variants they accept are `Vault(VaultRef)`, `Wallet(Token)`, and
//! `Emv(Token)` — all opaque references. This keeps `op-rails-card`
//! itself out of PCI DSS scope.
//!
//! ## Module layout
//!
//! - [`error`] — sealed `Error` enum, one variant per failure class.
//! - [`acquirer`] — the `CardAcquirer` trait and `AuthDecision` types.
//! - [`hyperswitch`] — driver against `https://sandbox.hyperswitch.io`
//!   and self-hosted Hyperswitch deployments.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)] // Errors documented in error.rs

pub mod acquirer;
pub mod error;
pub mod network_token;

#[cfg(feature = "hyperswitch")]
pub mod hyperswitch;

pub use acquirer::{
    AuthDecision, AuthRequest, CaptureRequest, CardAcquirer, RefundRequest, VoidRequest,
};
pub use error::{Error, Result};
pub use network_token::{
    Cryptogram, LifecycleEvent, LifecycleStream, MdesProvider, NetworkTokenProvider,
    ProviderConfig, TokenOnlyRail, VtsProvider,
};
