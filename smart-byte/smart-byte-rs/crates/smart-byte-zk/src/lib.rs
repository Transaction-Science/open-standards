//! # smart-byte-zk
//!
//! Zero-knowledge proof primitives for **selective disclosure of
//! verifiable credentials** in the Smart Byte substrate.
//!
//! ## Backends
//!
//! This crate exposes a single [`ZkScheme`] trait surface and three
//! concrete backends behind it:
//!
//! * [`bulletproofs`] — **real** Bulletproofs range proofs over
//!   Ristretto255 (`curve25519-dalek` + `merlin`). This is the
//!   working backend used by [`predicates`] for range / inequality /
//!   set-membership proofs.
//! * [`groth16`] — **stub**. Surface-only implementation of
//!   [`ZkScheme`] producing a deterministic dummy proof. Reserved for
//!   a future arkworks-backed integration.
//! * [`plonk`] — **stub**. Same shape as [`groth16`].
//!
//! The stub backends deliberately avoid pulling the arkworks
//! dependency tree (which is heavy and prone to version skew at the
//! workspace level) while still letting downstream code wire against
//! the trait. They are documented as *not* cryptographically binding
//! and panic-free.
//!
//! ## High-level surface
//!
//! * [`predicates`] — credential-attribute predicate proofs (range,
//!   inequality, set-membership) delegating to [`bulletproofs`].
//! * [`anoncreds`] — BBS-style anonymous-credential sketch: link
//!   secrets, blinded commitments, presentation envelopes.
//! * [`presentation`] — Verifiable Presentation builder bundling a
//!   collection of predicate / disclosure proofs against a holder
//!   binding.
//! * [`scheme`] — the [`ZkScheme`] trait and proving / verifying-key
//!   / proof newtypes.
//! * [`error`] — typed [`error::ZkError`].

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod anoncreds;
pub mod bulletproofs;
pub mod error;
pub mod groth16;
pub mod plonk;
pub mod predicates;
pub mod presentation;
pub mod scheme;

pub use error::ZkError;
pub use scheme::{Proof, ProvingKey, VerifyingKey, ZkScheme};
