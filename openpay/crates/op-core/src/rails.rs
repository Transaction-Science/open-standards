//! Rail abstraction.
//!
//! `op-core` is rail-agnostic. Concrete drivers (Hyperswitch, `FedNow`, PIX,
//! UPI, Apple Pay, ...) live in `op-rails-*` crates and implement the
//! [`Acquirer`] trait. The orchestrator routes a `Payment<Created>` to a
//! driver by inspecting the payment's [`RailKind`].
//!
//! ## Async
//!
//! The trait is intentionally NOT `async fn` in a trait (yet — Rust 1.95
//! supports it but with object-safety footguns). Drivers return a
//! `Pin<Box<dyn Future>>` via the `async_trait` crate, or callers use a
//! runtime-specific concrete type. We will refine this when the
//! `return_type_notation` RFC stabilizes.

use serde::{Deserialize, Serialize};

use crate::method::PaymentMethod;
use crate::money::Money;

/// Top-level family of payment rails.
///
/// A specific driver crate maps each variant to one or more underlying
/// schemes (e.g. `A2a` covers `FedNow`, RTP, PIX, UPI, SEPA Instant; the
/// driver picks based on currency and routing).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RailKind {
    /// Card networks (Visa, Mastercard, Amex, Discover, `RuPay`, Elo, etc.)
    /// via a PSP. Requires `PaymentMethod::Vault | Wallet | Emv`.
    Card,
    /// Account-to-account / instant rails: `FedNow`, RTP, PIX, UPI, SEPA Inst.
    /// Requires `PaymentMethod::A2a` or `PaymentMethod::Qr`.
    A2a,
    /// Mobile wallet rails using device cryptograms (Apple Pay, Google Pay).
    /// Requires `PaymentMethod::Wallet`.
    Wallet,
    /// QR-presented payment (`EMVCo` merchant-presented or
    /// consumer-presented). Often resolves to A2a underneath.
    Qr,
    /// Crypto / stablecoin settlement (USDC on Solana / Base / Ethereum,
    /// EURC, PYUSD, etc.). Requires `PaymentMethod::Crypto`. The
    /// fee profile that motivates the whole rail family: sub-cent
    /// settlement vs. 100–300 bp on card.
    Crypto,
}

/// What an acquirer returns after attempting authorization or capture.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcquirerResponse {
    /// Opaque id from the rail (auth code, end-to-end id, network txn id).
    pub rail_ref: String,
    /// How much the rail confirmed (may be less than requested).
    pub amount: Money,
    /// Free-form status from the rail.
    pub status: AcquirerStatus,
}

/// Normalized acquirer status. Drivers map their native codes to this.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AcquirerStatus {
    /// Authorized (card) or accepted-for-settlement (A2A).
    Approved,
    /// Soft decline — caller may retry, possibly with a different method.
    SoftDecline,
    /// Hard decline — do not retry with the same method.
    HardDecline,
    /// Suspected fraud; the caller should freeze and review.
    Fraud,
    /// Transient failure (timeout, network); retry-safe.
    Transient,
}

/// An acquirer driver. Implemented by crates in the `op-rails-*` family.
///
/// The trait is dyn-compatible: callers can hold `Box<dyn Acquirer>` and
/// route across multiple rails at runtime.
pub trait Acquirer: Send + Sync {
    /// The rail family this driver serves. The orchestrator uses this to
    /// route an incoming payment to the correct driver.
    fn rail_kind(&self) -> RailKind;

    /// Human-readable driver name (e.g. `"hyperswitch"`, `"fednow-frb"`).
    fn name(&self) -> &'static str;

    /// True if this driver can handle the given payment method.
    fn supports(&self, method: &PaymentMethod) -> bool;
}
