#![allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::needless_pass_by_value,
    clippy::module_name_repetitions
)]

//! Network-token rail integrations.
//!
//! This module hosts the [`NetworkTokenProvider`] trait and concrete
//! adapters for the major card-network tokenization services:
//!
//! - [`vts`] — Visa Token Service.
//! - [`mdes`] — Mastercard Digital Enablement Service.
//!
//! Each adapter ships stub implementations behind the default feature
//! set so the rest of the workspace builds without network access. The
//! real HTTP integrations are gated behind the `live` cargo feature on
//! this crate; stubs return deterministic test values so the
//! conformance harness in
//! [`op_driver_sdk::conformance::network_token`] is reproducible.
//!
//! ## Why a separate trait from [`crate::CardAcquirer`]
//!
//! Acquirers authorize, capture, void and refund — they move funds.
//! Tokenization services *manage credentials*: provision a network
//! token, listen for its lifecycle events, fetch a per-transaction
//! cryptogram. The two surfaces have distinct authentication,
//! distinct webhook contracts, and distinct PCI scope. Conflating
//! them in one trait blurs the boundary; keeping them separate
//! lets a deployment use Hyperswitch for authorization and VTS
//! directly for tokenization (the common production topology).
//!
//! ## Compile-time guarantee
//!
//! Routing code that requires the network-token liability shift can
//! constrain itself with a marker trait such as `TokenOnlyRail` and a
//! function signature like:
//!
//! ```rust,ignore
//! pub trait TokenOnlyRail {}
//!
//! fn route_to_token_only_rail<R: TokenOnlyRail>(
//!     card: op_core::Tokenized<op_core::Card>,
//! ) { /* ... */ }
//! ```
//!
//! Passing `op_core::Vaulted<op_core::Card>` to that function is a
//! compile error; the typestate makes the runtime mistake unreachable.
//! See `op-driver-sdk/tests/network_token_compile_fail.rs` for the
//! `trybuild` fixture that pins this guarantee.

pub mod cryptogram;
pub mod mdes;
pub mod provider;
pub mod vts;

pub use cryptogram::Cryptogram;
pub use mdes::MdesProvider;
pub use provider::{LifecycleEvent, LifecycleStream, NetworkTokenProvider, ProviderConfig};
pub use vts::VtsProvider;

/// Marker trait for rails that **require** a network-tokenized
/// credential.
///
/// Implementors guarantee they will not accept a [`op_core::Vaulted`]
/// card. The constraint is enforced at compile time by the function
/// signatures that take `Tokenized<Card>` directly — `TokenOnlyRail`
/// itself is just documentation that flags a rail as such.
pub trait TokenOnlyRail {
    /// Diagnostic name for the rail (e.g. `"vts-direct"`).
    fn name(&self) -> &'static str;
}

/// Compile-time routing helper: accepts only network-tokenized cards.
///
/// `route_to_token_only_rail::<R>(vaulted)` will not compile. See the
/// `trybuild` fixture in
/// `op-driver-sdk/tests/network_token_compile_fail.rs`.
pub fn route_to_token_only_rail<R: TokenOnlyRail>(
    rail: &R,
    card: op_core::Tokenized<op_core::Card>,
) -> String {
    // The body is intentionally minimal — the value of this function is
    // that its *signature* refuses non-tokenized cards.
    format!(
        "rail={} token_ref={} last4={}",
        rail.name(),
        card.token().token_ref,
        card.token().last4
    )
}
