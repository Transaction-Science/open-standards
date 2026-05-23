//! # `op-vault` — Token-only payment-method storage
//!
//! The scope-cutter for the entire `OpenPay` stack. Without this layer,
//! every component that touches `PaymentMethod` lands in PCI DSS scope
//! and operators face SAQ-D (~$50-200K/year audit overhead). With this
//! layer, raw PAN never crosses the vault boundary — orchestrator,
//! rail drivers, fraud scorer, and FFI bridges all handle opaque
//! [`op_core::VaultRef`] tokens that mean nothing to anyone but the
//! vault, putting them on the SAQ-A / SAQ-A-EP side of the scoping
//! line.
//!
//! ## What this crate ships
//!
//! - [`Vault`] — the trait operators implement against their platform
//!   vault (iOS Keychain, Android Keystore, HSM, AWS KMS, `HashiCorp`
//!   Vault, etc.).
//! - [`CardData`] — the only public type in the `OpenPay` surface that
//!   holds raw PAN. Construction validates Luhn + length + expiration.
//!   `Zeroize` + `ZeroizeOnDrop`. Only `pub(crate)` access to the raw
//!   bytes — vault implementations inside this crate can read them;
//!   everyone else sees only `first_six` / `last_four` / `exp_*`.
//! - [`TokenizationPolicy`] — random vs deterministic format,
//!   single-use vs reusable lifetime, optional TTL.
//! - [`InMemoryVault`] — reference implementation behind the
//!   `in-memory` feature. AES-256-GCM-SIV (misuse-resistant per
//!   RFC 8452). Suitable for tests; **not** PCI-compliant on its own
//!   (no durable storage, no audit logging, no rate limits, no HSM).
//!
//! ## What this crate does NOT ship
//!
//! - Platform vault adapters (iOS / Android / KMS). Those land in
//!   Phases 8-10 alongside the FFI bridges, where the platform-specific
//!   code lives.
//! - A token format spec. Format is per-vault; ours starts with
//!   `tok_v7_` followed by a UUID v7 simple-form. Operators can choose
//!   any non-PAN-confusable format.
//! - Network tokens (Visa Token Service, Mastercard MDES). Those come
//!   from the card networks, are stored as `VaultRef` strings, and
//!   resolved by the rail driver against the network token service.
//!   See Phase 5.2.
//! - Key management. Operators bring their own KMS / HSM; we accept
//!   a 32-byte key and refuse to know where it came from.
//!
//! ## PCI DSS 4.0.1 scoping rules this satisfies
//!
//! Per PCI SSC Tokenization Guidelines and PCI DSS v4.0.1:
//!
//! - **Token irreversibility**: with `TokenFormat::Random`, recovering
//!   PAN from a token requires breaking AES-256-GCM-SIV. Token-only
//!   systems are out of scope per the scoping guidance.
//! - **Token vault segmentation**: the [`Vault`] trait is the
//!   architectural boundary. Operators run the implementation behind
//!   their CDE segmentation; the rest of `OpenPay` stays outside.
//! - **Token distinguishability**: our default prefix (`tok_v7_`)
//!   contains non-digit characters, so tokens cannot be confused with
//!   PANs in storage or logs (§3.3 of the Tokenization Guidelines).
//! - **No detokenization for out-of-scope callers**: the [`Vault`]
//!   trait is the only way to detokenize, and the orchestrator only
//!   passes it to rail drivers at submit time.
//!
//! The crate ships behind a trait so operators can swap in a
//! FIPS 140-2 Level 3 hardware vault for production while keeping the
//! reference vault available for development.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]

pub mod card_data;
pub mod error;
pub mod policy;
pub mod vault;

#[cfg(feature = "in-memory")]
pub mod in_memory;

pub use card_data::CardData;
pub use error::{Error, Result};
pub use policy::{TokenFormat, TokenLifetime, TokenizationPolicy};
pub use vault::Vault;

#[cfg(feature = "in-memory")]
pub use in_memory::{InMemoryVault, TOKEN_PREFIX, generate_key};

// Re-export so consumers don't need a direct dependency on op-core
// just to handle vault references.
pub use op_core::VaultRef;
