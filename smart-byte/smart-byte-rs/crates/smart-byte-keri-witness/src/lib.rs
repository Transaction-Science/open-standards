//! # smart-byte-keri-witness
//!
//! Full KERI key-rotation cycle, witness layer, watchers, recovery,
//! and delegation for the Smart Byte substrate.
//!
//! The cryptographic spine (SAID + key-event log layout) is defined in
//! `smart-byte/spec/identity_and_key_rotation.md`. The
//! [`smart-byte-core`] crate already supplies the SAID primitive and
//! Ed25519 sign/verify. This crate ingests the remaining pieces of the
//! KERI spec that are required for end-to-end identity continuity:
//!
//! * the *full* inception -> rotation cycle with pre-rotated key
//!   revelation (events: [`InceptionEvent`], [`RotationEvent`],
//!   [`InteractionEvent`], [`RecoveryEvent`], [`DelegationEvent`]);
//! * the **witness layer** — peers who counter-sign each event so the
//!   controller cannot rewrite their own history (see [`witness`]);
//! * **watchers** — third-party observers who detect duplicity by
//!   collecting receipts and comparing logs (see [`watcher`]);
//! * **recovery** semantics — the reserved `rec` event type elevated
//!   to first-class status here, gated to catastrophic key loss;
//! * **delegated identifiers** so an organisation can delegate to a
//!   department whose events anchor to the parent's log;
//! * **OOBI** (Out-Of-Band Introduction) for bootstrapping "I claim
//!   to be controller X, here are my witnesses" (see [`oobi`]).
//!
//! All events are encoded using deterministic CBOR per the spec; SAIDs
//! are 32-byte BLAKE3 digests via [`smart_byte_core::Said`]. Signing
//! supports both classical Ed25519 and hybrid (Ed25519 + ML-DSA) via
//! [`smart_byte_pq`].

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod controller;
pub mod error;
pub mod events;
pub mod oobi;
pub mod storage;
pub mod verifier;
pub mod watcher;
pub mod witness;

pub use controller::{Controller, KeyPair};
pub use error::{KeriError, Result};
pub use events::{
    Anchor, ConfigTrait, ControllerAid, DelegationEvent, EventType, InceptionEvent,
    InteractionEvent, KeyEvent, PublicKey, RecoveryEvent, RotationEvent, Threshold, WatcherAid,
    WitnessAid,
};
pub use oobi::Oobi;
pub use storage::{EventLogStorage, FileStorage, MemoryStorage};
pub use verifier::{DuplicityEvidence, LogVerifier, VerificationReport};
pub use watcher::{DuplicityDetected, Watcher};
pub use witness::{Witness, WitnessReceipt};

/// CBOR version-tag string used inside every event's `v` field.
///
/// Smart Byte v1.0 with canonical CBOR encoding. The trailing `JSON`
/// tag is held over from KERI for forward-compatibility with a future
/// JSON-serialization variant (see spec §3.3).
pub const VERSION_STRING: &str = "SBYTE10JSON";
