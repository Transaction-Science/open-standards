//! Smart Byte core types.
//!
//! This crate provides the load-bearing primitives of the Smart Byte
//! substrate's reference implementation:
//!
//! * [`Envelope`] — content-addressed signed carrier of arbitrary cargo
//!   with intrinsic provenance, ownership history, and energy cost.
//! * [`Said`] — Self-Addressing IDentifier (BLAKE3 over canonical CBOR).
//! * [`sign`] / [`verify`] — Ed25519 signature over the envelope's SAID.
//! * [`Cargo`] — discriminated union of byte payload, USD claim,
//!   joule claim, or custom typed body.
//! * [`JouleCost`] — measured and estimated microjoules of energy
//!   attributable to producing or transmitting the envelope.
//! * [`Provenance`] — issuer SAID, issuance time, authorization blob.
//! * [`OwnershipChain`] — git-style chain of signed transitions.
//!
//! The substrate's load-bearing rule is *self-addressing*: the
//! envelope's `id` field is the BLAKE3 hash of its own canonical
//! CBOR encoding with the `id` field zeroed. Two implementers in any
//! language can therefore compute the same SAID for the same
//! semantic envelope without trusting each other.

pub mod cargo;
pub mod envelope;
pub mod joule_cost;
pub mod ownership;
pub mod provenance;
pub mod said;
pub mod sign;

pub use cargo::Cargo;
pub use envelope::{Envelope, EnvelopeError};
pub use joule_cost::JouleCost;
pub use ownership::{OwnershipChain, Transition};
pub use provenance::Provenance;
pub use said::{Said, SaidError};
pub use sign::{Signature, SigningKey, VerifyingKey, sign, verify};

/// BLAKE3 32-byte hash output. Re-exported so downstream crates can
/// refer to it without depending on `blake3` directly.
pub type Blake3Hash = [u8; 32];
