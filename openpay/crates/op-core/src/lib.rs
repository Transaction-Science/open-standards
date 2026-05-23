//! # `op-core`
//!
//! Safety-critical core for `OpenPay`. Pure domain logic: no I/O, no network,
//! no platform-specific code.
//!
//! ## Invariants enforced by the type system
//!
//! 1. Money values cannot be constructed from `f64`. All amounts are exact
//!    minor-unit integers paired with an ISO 4217 currency.
//! 2. A `Payment<S>` can only transition through legal states. Illegal
//!    transitions (e.g. capturing an unauthorized payment) do not compile.
//! 3. Raw PAN (Primary Account Number) is unreachable without the
//!    `pci-scope` feature flag. The default surface exposes only opaque
//!    `Token` and `EmvBlob` payment-method variants.
//! 4. No `unsafe` in this crate.
//!
//! ## Module layout
//!
//! - [`money`] — `Money`, `Currency`, ISO 4217 minor-unit arithmetic.
//! - [`method`] — `PaymentMethod` enum (token/EMV/A2A/wallet).
//! - [`payment`] — `Payment<S>` typestate machine.
//! - [`error`] — sealed error enum, one per crate convention.
//! - [`rails`] — rail-agnostic `Acquirer` trait that adapter crates implement.
//! - [`network_token`] — `Tokenized<Card>` / `Vaulted<Card>` typestates
//!   and network-token primitives (VTS / MDES surrogate credentials).

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod method;
pub mod money;
pub mod network_token;
pub mod payment;
pub mod rails;

pub use error::Error;
pub use method::{A2aKey, CryptoAddress, PaymentMethod, Token, VaultRef};
pub use money::{Currency, Money};
pub use network_token::{
    Card, CardNetwork, NetworkToken, NetworkTokenLifecycleEvent, PaymentMethodKind, Tokenized,
    Vaulted,
};
pub use payment::{Authorized, Captured, Created, Failed, Payment, Refunded, Voided};
pub use rails::{Acquirer, AcquirerResponse, RailKind};

/// Raw PAN. Re-exported here only when the `pci-scope` feature is on.
/// Available exclusively to PCI-certified code paths (the vault).
#[cfg(feature = "pci-scope")]
pub use method::pci::RawPan;
