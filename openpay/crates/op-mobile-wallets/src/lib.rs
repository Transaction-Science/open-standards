//! # `op-mobile-wallets` — Apple Pay / Google Pay / Samsung Pay acceptance
//!
//! Decrypts the encrypted payment payloads delivered by the three major
//! mobile wallets and emits a normalized [`DecryptedToken`] whose
//! credential reference is a [`op_core::VaultRef`] — never a raw PAN
//! [`String`]. This is the canonical no-PAN-in-merchant-system flow that
//! OpenPay's PCI-zero architecture (see issue #1) depends on for
//! in-person + in-app acceptance.
//!
//! ## What a mobile-wallet payment payload is
//!
//! When a customer taps Apple Pay / Google Pay / Samsung Pay, the wallet
//! ships an *encrypted* payment payload to the merchant. The payload
//! contains a device-PAN (DPAN, a network token), an expiry, and a
//! per-transaction cryptogram. **The merchant decrypts the payload
//! using a merchant-specific certificate-bound private key** — the key
//! the merchant generated on enrollment with the wallet, whose public
//! half lives in the merchant's wallet-issuer-signed certificate.
//!
//! After decryption:
//!
//! - The DPAN is *not* a PAN — it's a network token that the issuer
//!   recognizes alongside the cryptogram. PCI scope for the DPAN
//!   itself is reduced (still PCI-controlled, but no underlying PAN
//!   ever transits the merchant), and OpenPay funnels it into a
//!   `VaultRef` immediately at the decryption boundary so the
//!   downstream API surface only ever sees the opaque vault reference.
//! - The cryptogram authenticates the *use* of the DPAN at
//!   authorization, producing the network-token liability shift.
//!
//! ## Per-wallet crypto
//!
//! All three wallets agree on **ECDH P-256** for ephemeral key
//! agreement. They diverge in KDF and AEAD construction:
//!
//! | Wallet     | KDF              | AEAD                          |
//! |------------|------------------|-------------------------------|
//! | Apple Pay  | X9.63-KDF / SHA-256 | AES-256-GCM (zero IV)      |
//! | Google Pay (ECv2) | HKDF-SHA256 | AES-256-CTR + HMAC-SHA256 |
//! | Samsung Pay | HKDF-SHA256     | AES-256-GCM                   |
//!
//! Each wallet's signing-cert path is also distinct: Apple signs with
//! the Apple Pay leaf chained to the Apple Root CA; Google signs the
//! `signedMessage` with an intermediate key signed by the
//! Google-published root keys; Samsung signs with their own
//! enrollment-bound leaf. We accept signing-key material at the
//! decryptor-construction boundary and verify against it at decryption
//! time.
//!
//! ## PCI scope
//!
//! Every public API in this crate that returns a credential returns
//! it as a [`op_core::VaultRef`]. There is **no** code path that
//! exposes the cleartext DPAN as a `String`, `&str`, `Vec<u8>`, or
//! `&[u8]` to a caller outside this crate. The cleartext DPAN exists
//! transiently inside the decryption boundary, in a
//! `zeroize`-on-drop buffer, just long enough to be encoded into a
//! `VaultRef` (via the operator-supplied vault tokenizer) and then
//! wiped. See [`VaultTokenizer`] for the operator hook.
//!
//! ## Modules
//!
//! - [`apple_pay`]   — Apple Pay payment-token decryption.
//! - [`google_pay`]  — Google Pay ECv2 (current) and ECv1 (stub).
//! - [`samsung_pay`] — Samsung Pay payment-token decryption.
//! - [`cryptogram`]  — ECI translation + 3DS2 indicator helpers.
//! - [`wallet`]      — `Wallet` enum + `WalletDecryptor` trait + factory.
//! - [`error`]       — typed errors used by every backend.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
// Wallet-vendor brand names (Apple Pay, Google Pay, OpenPay, PaymentToken,
// ECv1/ECv2) appear throughout the docs as plain English and would be
// unreadable if every occurrence were backticked.
#![allow(clippy::doc_markdown)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
// Decryptors orchestrate ~10 steps end-to-end; splitting them across
// helpers obscures the protocol-spec correspondence.
#![allow(clippy::too_many_lines)]
// Test-vector builders take every payload field by argument to keep them
// declarative; they're test-only and not part of the public API.
#![allow(clippy::too_many_arguments)]
// p.len() (a usize over a JSON-bounded byte slice) fitting in a u32 is
// guaranteed in this context — payloads are kilobytes, not gigabytes.
#![allow(clippy::cast_possible_truncation)]
// Three-letter abbreviations in domain names (pk / vk, dpan / hash)
// trigger this aggressively without improving clarity.
#![allow(clippy::similar_names)]

pub mod apple_pay;
pub mod cryptogram;
pub mod error;
pub mod google_pay;
pub mod samsung_pay;
pub mod wallet;

pub use apple_pay::{ApplePayDataType, ApplePayToken, CertChain, ExpirationDate};
pub use cryptogram::{EciIndicator, ThreeDsIndicator};
pub use error::{Error, Result};
pub use google_pay::{GooglePayProtocolVersion, GooglePayToken, IntermediateSigningKey};
pub use samsung_pay::SamsungPayToken;
pub use wallet::{
    Base64Encoded, DecryptedToken, PaymentMethodInfo, VaultTokenizer, Wallet, WalletConfig,
    WalletDecryptor, decryptor_for,
};
