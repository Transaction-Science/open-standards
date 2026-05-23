//! W3C Verifiable Credentials and Decentralized Identifiers for Smart Byte.
//!
//! This crate ingests the deployed footprint of content-addressed signed
//! claims — W3C Verifiable Credential Data Model 2.0 (Rec) and Decentralized
//! Identifiers (Rec) — into the Smart Byte substrate.
//!
//! A [`VerifiableCredential`] is a structured claim issued by an
//! [`Issuer`] about one or more [`CredentialSubject`]s, sealed with
//! one of three families of [`Proof`]:
//!
//! * [`DataIntegrityProof`] — embedded W3C Data Integrity proof.
//!   `eddsa-jcs-2022` is implemented natively (canonical JSON via
//!   `serde_jcs` + Ed25519). RDFC variants are reserved behind the
//!   `rdf-canon` feature.
//! * [`JwtProof`] — VC-JWT (RFC 7519 JWT carrying a VC payload).
//! * [`SdJwtProof`] — Selective-Disclosure JWT (IETF draft
//!   `oauth-selective-disclosure-jwt-15`) for holder-driven selective
//!   disclosure.
//!
//! A [`VerifiablePresentation`] wraps one or more credentials with a
//! holder-bound proof.
//!
//! Revocation and suspension are expressed via Bitstring Status List 2021
//! ([`status_list`]).
//!
//! Finally [`cargo_bridge`] packages a `VerifiableCredential` as a
//! `Cargo::Custom` payload on a Smart Byte [`Envelope`][smart_byte_core::Envelope],
//! so a VC's SAID is computed over its canonical JCS encoding and the
//! envelope signature reuses the smart-byte-core primitives.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod cargo_bridge;
pub mod credential;
pub mod did;
pub mod error;
pub mod holder;
pub mod issuer;
pub mod jwt;
pub mod presentation;
pub mod proof;
pub mod sd_jwt;
pub mod status_list;

pub use cargo_bridge::{VC_CARGO_TYPE_URI, vc_envelope, vc_from_envelope};
pub use credential::{
    CredentialStatus, CredentialSubject, Evidence, TermsOfUse, VcBuilder,
    VerifiableCredential, VC_CONTEXT_V2,
};
pub use did::{Did, DidError, DidKey};
pub use error::VcError;
pub use holder::Holder;
pub use issuer::Issuer;
pub use jwt::{issue_vc_jwt, verify_vc_jwt};
pub use presentation::VerifiablePresentation;
pub use proof::{
    DataIntegrityProof, JwtProof, Proof, ProofPurpose, SdJwtProof,
    issue_data_integrity, verify_data_integrity,
};
pub use sd_jwt::{Disclosure, issue_sd_jwt, present_sd_jwt, verify_sd_jwt};
pub use status_list::{
    BitstringStatusList, StatusListCredential, StatusPurpose,
    check_status,
};
