//! # `op-bnpl`
//!
//! Buy-Now-Pay-Later rails for OpenPay. Three providers ship in this
//! crate: **Affirm**, **Klarna**, and **Afterpay/Clearpay**. Together
//! they account for the overwhelming majority of BNPL checkout volume
//! globally, and BNPL is ~10% of e-commerce checkout volume in 2026.
//!
//! ## Why a separate rail
//!
//! BNPL is not card. The acceptance flow goes:
//!
//! 1. **Origination** — merchant declares cart contents (line items,
//!    shipping address, customer identity hint) and requests an
//!    instalment offer.
//! 2. **Consumer flow** — the provider underwrites and presents an
//!    instalment plan to the consumer (in-app for Affirm Anywhere,
//!    hosted page for Klarna, widget for Afterpay).
//! 3. **Authorization** — consumer accepts; provider returns a
//!    short-lived authorization token to the merchant.
//! 4. **Capture** — merchant captures funds at shipment. Provider
//!    pays the merchant (less merchant-discount) and assumes the
//!    consumer-credit risk.
//! 5. **Settlement** — provider funds the merchant on its own
//!    schedule (T+1 to T+2 typically).
//! 6. **Refund / void** — symmetric to card, but executed against the
//!    instalment plan rather than the underlying card.
//!
//! Card's authorize→capture→settle→refund maps loosely, but the
//! BNPL-specific intermediate states (`OriginationPending`,
//! `Approved`, instalment-plan terms) mean we model BNPL as its own
//! typestate machine — [`BnplCheckout<S>`] — parallel to
//! [`op_core::Payment<S>`] rather than reused.
//!
//! ## Module layout
//!
//! - [`acquirer`] — generic [`BnplAcquirer`] trait + session / checkout
//!   value types.
//! - [`lifecycle`] — `BnplCheckout<S>` typestate, [`BnplProvider`],
//!   [`InstalmentPlan`].
//! - [`intent`] — `BnplIntent`, `LineItem`, `ShippingInfo`, etc.
//! - [`affirm`], [`klarna`], [`afterpay`] — provider-specific drivers.
//! - [`eligibility`] — pre-checkout creditworthiness / geography /
//!   amount-band lookup.
//! - [`webhook`] — inbound signature verification + event types.
//! - [`error`] — sealed error enum.
//!
//! ## Invariants
//!
//! - No `unsafe`.
//! - No `unwrap` / `expect` outside `#[test]`.
//! - All money arithmetic flows through [`op_core::Money`] (integer
//!   minor units; no `f64`).
//! - HTTP transport is `reqwest` with rustls. No platform-native TLS.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::multiple_crate_versions)]

pub mod acquirer;
pub mod affirm;
pub mod afterpay;
pub mod eligibility;
pub mod error;
pub mod intent;
pub mod klarna;
pub mod lifecycle;
pub mod webhook;

pub use acquirer::{
    AuthorizedCheckout, BnplAcquirer, CapturedCheckout, InitiatedSession, RefundedCheckout,
};
pub use affirm::AffirmAcquirer;
pub use afterpay::{AfterpayAcquirer, AfterpayRegion};
pub use eligibility::{EligibilityCheck, EligibilityContext, EligibilityResult};
pub use error::{Error, Result};
pub use intent::{
    BillingInfo, BnplIntent, ConsumerInfo, IdempotencyKey, LineItem, RedirectUrls, ShippingInfo,
};
pub use klarna::{KlarnaAcquirer, KlarnaRegion};
pub use lifecycle::{
    Approved, BnplCheckout, BnplProvider, BnplState, Captured, InstalmentInterval,
    InstalmentPlan, Initiated, Refunded, Settled,
};
pub use webhook::{
    BnplEvent, BnplEventKind, WebhookProvider, verify_affirm_webhook, verify_afterpay_webhook,
    verify_klarna_webhook,
};
