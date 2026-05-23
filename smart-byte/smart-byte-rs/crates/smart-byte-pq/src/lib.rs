//! # smart-byte-pq
//!
//! Post-quantum signature algorithms for the Smart Byte substrate.
//!
//! NIST finalised the first three post-quantum signature standards in
//! August 2024:
//!
//! * **FIPS 204 — ML-DSA** (Module-Lattice-Based Digital Signature
//!   Algorithm, formerly CRYSTALS-Dilithium). Three parameter sets:
//!   ML-DSA-44, ML-DSA-65, ML-DSA-87.
//! * **FIPS 205 — SLH-DSA** (Stateless Hash-Based Digital Signature
//!   Algorithm, formerly SPHINCS+). Twelve parameter sets across two
//!   hash families (SHA-2, SHAKE), three security levels (128, 192,
//!   256 bit), and two size/speed trade-offs (small `s`, fast `f`).
//! * **Draft FIPS 206 — FN-DSA** (FFT-over-NTRU Digital Signature
//!   Algorithm, formerly Falcon). Two parameter sets: FN-DSA-512 and
//!   FN-DSA-1024. The FIPS 206 standard is in late draft as of mid-2025;
//!   for that reason the FN-DSA implementation in this crate is gated
//!   behind the `falcon` feature. When the feature is off, the public
//!   API is preserved (so callers can still match on the enum variants)
//!   but the signer functions return [`error::PqError::UnsupportedAlgorithm`].
//!
//! Smart Byte signs *envelope SAIDs* (BLAKE3 commitments to the
//! canonical CBOR encoding of an envelope with `id` zeroed). The choice
//! of signature algorithm is selected per envelope via the one-byte
//! *algorithm identifier* defined in the Smart Byte spec
//! (`§4.1 / §8.3 / §8.4 / §19`). This crate provides the wire-side
//! signer/verifier implementations for every algorithm in the enum.
//!
//! The default for v1 envelopes remains Ed25519; PQ algorithms are
//! opt-in per envelope until the substrate-wide migration cutover
//! described in `§19.5` / `§19.8`.
//!
//! ## Hybrid signatures
//!
//! For the transition period NIST and the IETF
//! (`draft-ietf-pquip-pqt-hybrid-terminology`) recommend hybrid
//! signatures that combine a classical scheme (Ed25519 or ECDSA-P256)
//! with a post-quantum scheme (typically ML-DSA). A hybrid signature
//! is valid only if **both** component signatures verify; an attacker
//! must break both schemes to forge one. See [`hybrid`].

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod algorithm;
pub mod error;
pub mod hybrid;
pub mod mldsa;
pub mod signer_trait;
pub mod slhdsa;

#[cfg(feature = "falcon")]
pub mod fndsa;

#[cfg(not(feature = "falcon"))]
#[path = "fndsa_stub.rs"]
pub mod fndsa;

pub use algorithm::SignatureAlgorithm;
pub use error::PqError;
pub use signer_trait::{Signer, Verifier};
