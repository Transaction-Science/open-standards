//! Nostr (Notes and Other Stuff Transmitted by Relays) adapter for
//! Smart Byte.
//!
//! Nostr is a relay-based public-key social protocol where every event
//! is a signed JSON object. This crate ingests its deployed footprint
//! into the Smart Byte substrate: event signing/verification,
//! identity encodings, encrypted DMs, gift-wrap, proof of work,
//! delegated signing, relay-list metadata, and the wire-level
//! REQ/EVENT/EOSE/OK/AUTH/CLOSE/CLOSED/NOTICE protocol.
//!
//! ## What's covered
//!
//! * **NIP-01** ([`event`]) — event JSON, canonical id, BIP-340 Schnorr
//!   signature over the SHA-256 of the canonical serialization.
//! * **NIP-04** ([`nip04`]) — legacy DM encryption (AES-256-CBC over a
//!   bare ECDH x-coordinate). Provided for compatibility only.
//! * **NIP-05** ([`nip05`]) — DNS-based human-readable identifiers via
//!   `.well-known/nostr.json`. HTTP fetching left to callers.
//! * **NIP-07** ([`Nip07Provider`]) — `window.nostr` provider interface
//!   abstracted as a Rust trait (no DOM coupling).
//! * **NIP-09** — event deletion is just kind-5 with `e`/`a` tags; we
//!   provide [`event::Event`] helpers via the generic event API and a
//!   convenience constant [`KIND_DELETION`].
//! * **NIP-11** ([`nip11`]) — relay information document.
//! * **NIP-13** ([`nip13`]) — proof-of-work mining and verification.
//! * **NIP-17** ([`nip17`]) — private DMs via NIP-59 gift-wrap +
//!   sealed-event.
//! * **NIP-19** ([`bech32`]) — bech32 encodings: `npub`, `nsec`,
//!   `note`, `nprofile`, `nevent`, `nrelay`. (`naddr` left as a
//!   compatible HRP without a typed payload; callers may use the raw
//!   bech32 encoder.)
//! * **NIP-26** ([`nip26`]) — delegated event signing.
//! * **NIP-44** ([`nip44`]) — versioned encrypted payloads (v2:
//!   ChaCha20 + HMAC-SHA256, padded buckets).
//! * **NIP-65** ([`nip65`]) — relay list metadata (kind 10002).
//! * **Relay client surface** ([`relay`], [`filter`]) — REQ / EVENT /
//!   CLOSE / NOTICE / EOSE / OK / AUTH / CLOSED messages and an
//!   in-memory router. WebSocket transport is intentionally out of
//!   scope.
//!
//! ## What's intentionally scoped out
//!
//! * WebSocket I/O (relay client and relay server). The
//!   [`relay::ClientMessage`] / [`relay::RelayMessage`] types are the
//!   designed extension point for plugging in any transport.
//! * NIP-05 HTTP fetching. [`nip05`] models the document and a pure
//!   `verify` function; the network round-trip is the caller's
//!   responsibility.
//! * `naddr` typed payload. The bech32 encoder accepts arbitrary HRPs
//!   so callers can construct it.
//! * Full NIP-44 v2 reference test vectors. The crate's roundtrip and
//!   MAC tests confirm internal consistency; cross-implementation
//!   vectors should be added when the spec freezes its public corpus.

#![forbid(unsafe_code)]

pub mod bech32;
pub mod error;
pub mod event;
pub mod filter;
pub mod keys;
pub mod nip04;
pub mod nip05;
pub mod nip11;
pub mod nip13;
pub mod nip17;
pub mod nip26;
pub mod nip44;
pub mod nip65;
pub mod relay;

pub use error::NostrError;
pub use event::{Event, UnsignedEvent, canonical_serialize};
pub use filter::Filter;
pub use keys::{NostrPublicKey, NostrSecretKey, hex_decode, hex_encode, schnorr_sign, schnorr_verify};
pub use relay::{ClientMessage, RelayMessage, RelayRouter};

/// Kind 5 — NIP-09 event deletion.
pub const KIND_DELETION: u32 = 5;

/// NIP-07 `window.nostr` provider interface modeled as a Rust trait.
///
/// Browser extensions implement this surface in JavaScript; on the Rust
/// side, native signers (e.g. hardware wallets, KMS-backed services)
/// can implement it directly. Async by design so implementations can
/// prompt for user consent.
#[async_trait::async_trait]
pub trait Nip07Provider: Send + Sync {
    /// Return the active user's hex pubkey.
    async fn get_public_key(&self) -> Result<String, NostrError>;
    /// Sign `UnsignedEvent` and return a fully signed [`Event`].
    async fn sign_event(&self, event: UnsignedEvent) -> Result<Event, NostrError>;
    /// NIP-04 encrypt `plaintext` to `recipient_pubkey_hex`.
    async fn nip04_encrypt(
        &self,
        recipient_pubkey_hex: &str,
        plaintext: &str,
    ) -> Result<String, NostrError>;
    /// NIP-04 decrypt `ciphertext` from `sender_pubkey_hex`.
    async fn nip04_decrypt(
        &self,
        sender_pubkey_hex: &str,
        ciphertext: &str,
    ) -> Result<String, NostrError>;
    /// NIP-44 encrypt `plaintext` to `recipient_pubkey_hex`.
    async fn nip44_encrypt(
        &self,
        recipient_pubkey_hex: &str,
        plaintext: &str,
    ) -> Result<String, NostrError>;
    /// NIP-44 decrypt `ciphertext` from `sender_pubkey_hex`.
    async fn nip44_decrypt(
        &self,
        sender_pubkey_hex: &str,
        ciphertext: &str,
    ) -> Result<String, NostrError>;
}
