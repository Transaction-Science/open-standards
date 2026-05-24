//! Encrypted Data Vaults for Smart Byte.
//!
//! This crate ingests the DIF Encrypted Data Vaults (EDV) specification
//! v0.10 — <https://identity.foundation/edv-spec/> — and provides a
//! pure-Rust reference implementation. An EDV is a *zero-knowledge*
//! storage substrate: the vault operator holds ciphertext only and
//! cannot read the documents it stores, yet can still serve equality
//! lookups (via HMAC-blinded indexes) and enforce delegated access (via
//! ZCAP-LD capability chains).
//!
//! ## Modules
//!
//! * [`spec`] — DIF EDV v0.10 wire types (`EncryptedDocument`, `Config`,
//!   `Provider`, `Hmac`, `Stream`).
//! * [`jwe`] — JWE wrap/unwrap. Key agreement: **ECDH-ES** over **P-256**
//!   (RFC 7518 § 4.6) with HKDF-SHA-256 Concat-KDF. Content encryption:
//!   **AES-256-GCM** (RFC 7518 § 5.3).
//! * [`vault`] — [`vault::Vault`] async trait plus
//!   [`vault::InMemoryVault`], the in-memory reference implementation.
//! * [`index`] — HMAC-over-value encrypted indexes (DIF EDV v0.10 § 4.4),
//!   so the vault can answer equality queries on attribute values it
//!   cannot read.
//! * [`zcap`] — ZCAP-LD capability chains, the delegation model EDV uses
//!   for authorisation.
//! * [`stream`] — chunked stream encryption for payloads too large to
//!   seal in a single AEAD call.
//! * [`replication`] — one-way pull replication between two vaults.
//! * [`error`] — typed errors.
//!
//! ## Cryptographic profile
//!
//! | Layer | Primitive | Crate |
//! |-------|-----------|-------|
//! | Content encryption | AES-256-GCM | `aes-gcm` |
//! | Key agreement | ECDH-ES over P-256 | `p256` |
//! | Key derivation | HKDF-SHA-256 (Concat-KDF framing) | `hkdf`, `sha2` |
//! | Index HMAC | HMAC-SHA-256 | `hmac`, `sha2` |
//!
//! All cryptography is pure Rust; no `unsafe` and no platform crypto
//! dependencies.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod error;
pub mod index;
pub mod jwe;
pub mod replication;
pub mod spec;
pub mod stream;
pub mod vault;
pub mod zcap;

pub use error::EdvError;
pub use index::{IndexKey, Query, search};
pub use jwe::{Jwe, KeyPair, PrivateKey, Recipient, unwrap, wrap};
pub use replication::{PullReplicator, ReplicationReport, Replicator, sync};
pub use spec::{
    Config, EncryptedDocument, Hmac, IndexedAttribute, IndexedEntry,
    KeyDescriptor, Provider, Stream,
};
pub use stream::{StreamChunk, StreamManifest, open as stream_open, seal as stream_seal};
pub use vault::{DocumentRef, InMemoryVault, Vault};
pub use zcap::{Capability, Invocation, verify_chain};
