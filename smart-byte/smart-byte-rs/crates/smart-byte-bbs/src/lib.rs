//! # smart-byte-bbs
//!
//! BBS+ signatures (Boneh-Boyen-Shacham extended) over BLS12-381 for
//! the Smart Byte substrate.
//!
//! BBS+ lets a signer commit to an ordered list of messages and lets
//! the holder later prove possession of a valid signature while
//! disclosing only a chosen *subset* of the messages. Unlike SD-JWT,
//! which is hash-based and links presentations of the same credential
//! via shared disclosure hashes, BBS+ is cryptographic — every
//! presentation re-randomises the signature, so two presentations of
//! the same credential are computationally unlinkable (the
//! "unlinkability" property at the heart of AnonCreds and the W3C
//! `bbs-2023` Data Integrity cryptosuite).
//!
//! ## Modules
//!
//! * [`keys`]      — keygen, secret/public-key serialisation, zeroize.
//! * [`generators`] — deterministic G1 generator derivation from a
//!   domain string + index (RFC 9380 `hash_to_curve`).
//! * [`sign`]      — BBS+ `Sign` and `Verify` over the message vector.
//! * [`proof`]     — `ProofGen` and `ProofVerify` for selective
//!   disclosure with Fiat-Shamir over SHA-256.
//! * [`encode`]    — scalar / point / signature wire codecs.
//! * [`cryptosuite`] — W3C `bbs-2023` Data Integrity cryptosuite.
//! * [`anoncreds_bridge`] — Hyperledger AnonCreds 2.0 helpers
//!   (revocation handles, link secrets, predicate-proof envelope).
//! * [`cargo_bridge`] — Smart Byte envelope packaging for credentials
//!   and presentations, with SAID stability under disclosure.
//! * [`error`]     — typed `BbsError`.
//!
//! ## Standard tracked
//!
//! `draft-irtf-cfrg-bbs-signatures-08` (IETF CFRG, 2025) — the
//! stabilising revision of the BBS Signature Scheme.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod anoncreds_bridge;
pub mod cargo_bridge;
pub mod cryptosuite;
pub mod encode;
pub mod error;
pub mod generators;
pub mod keys;
pub mod proof;
pub mod sign;

pub use anoncreds_bridge::{
    ANONCREDS_2_0_TAG, LinkSecret, Predicate, PredicateClaim, PredicateProof,
    RevocationHandle,
};
pub use cargo_bridge::{
    BBS_CREDENTIAL_CARGO_TYPE_URI, BBS_PRESENTATION_CARGO_TYPE_URI,
    BbsCredentialBody, BbsPresentationBody, bbs_credential_envelope,
    bbs_credential_from_envelope, presentation_envelope,
    presentation_from_envelope,
};
pub use cryptosuite::{
    Bbs2023DisclosureProof, Bbs2023IssuanceProof, Bbs2023Suite,
    CRYPTOSUITE_BBS_2023, ProofSuite, create_bbs_2023_disclosure,
    issue_bbs_2023, verify_bbs_2023_disclosure, verify_bbs_2023_issuance,
};
pub use encode::{
    DST_FIAT_SHAMIR, DST_HASH_TO_SCALAR, G1_BYTES, G2_BYTES, SCALAR_BYTES,
    hash_to_scalar, message_to_scalar,
};
pub use error::BbsError;
pub use generators::message_generators;
pub use keys::{KeyPair, PublicKey, SecretKey, keygen};
pub use proof::{DisclosureProof, create_proof, verify_proof};
pub use sign::{Signature, sign, verify};
