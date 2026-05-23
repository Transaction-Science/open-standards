//! Hyperswitch driver.
//!
//! Wraps the Hyperswitch V1 REST API. Verified against the public API
//! reference at <https://api-reference.hyperswitch.io/v1/payments>.
//!
//! ## Endpoints used
//!
//! | `OpenPay` call | HTTP                          |
//! |--------------|-------------------------------|
//! | `authorize`  | `POST /payments`              |
//! | `capture`    | `POST /payments/{id}/capture` |
//! | `void`       | `POST /payments/{id}/cancel`  |
//! | `refund`     | `POST /refunds`               |
//!
//! ## Auth
//!
//! `api-key` HTTP header. Server-side only — never exposed to clients.
//! The Hyperswitch docs are explicit: "Make sure to never share your
//! API key with your client application as this could potentially
//! compromise your payment flow."
//!
//! ## Idempotency
//!
//! Hyperswitch supports merchant-provided `payment_id` values up to 30
//! characters as the idempotency mechanism. We use UUID v7 (simple
//! form, no hyphens, 32 chars truncated to 30) so identical requests
//! short-circuit to the same payment.
//!
//! ## Status mapping
//!
//! Hyperswitch's V1 status enum has 17 values. We map them to the
//! 9-value [`AuthStatus`]. The mapping is verified against the
//! Hyperswitch docs' status descriptions; unknown values return
//! [`Error::UnknownStatus`] rather than guessing.

pub mod client;
pub mod status_map;
pub mod wire;

pub use client::HyperswitchClient;

#[cfg(test)]
mod tests {
    // Top-level smoke tests live in tests/hyperswitch.rs; module-level
    // tests are in each submodule.
}
