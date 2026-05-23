//! W3C Decentralized Identifier (DID) resolution for Smart Byte.
//!
//! This crate ingests the deployed footprint of DIDs — W3C DID Core 1.0
//! Recommendation (July 2022) — into the Smart Byte substrate. DIDs are
//! the identifier system that sits beneath the Verifiable Credentials
//! ecosystem, and a natural pairing with Smart Byte's content-addressed
//! [`Said`][smart_byte_core::Said]s: SAIDs identify *content*, DIDs identify
//! *controllers*.
//!
//! The crate implements the four most-used DID methods by deployed volume:
//!
//! * [`did:key`][methods::key] — single-key DIDs derived offline from a
//!   multicodec-prefixed, multibase-encoded public key. Ed25519, P-256
//!   and secp256k1 are supported; BLS12-381 G2 behind the `bls12_381`
//!   feature.
//! * [`did:web`][methods::web] — DID document fetched over HTTPS from a
//!   well-known location on a web origin.
//! * [`did:peer`][methods::peer] — inline-encoded DID document; numalgo 2
//!   and numalgo 4 (the practical ones used by Aries / DIDComm) are
//!   implemented.
//! * [`did:jwk`][methods::jwk] — `did:jwk:<base64url-encoded-jwk>`. Offline.
//!   Useful for OIDC4VCI flows.
//!
//! [`did:ion`][methods::ion] is behind the `ion` feature flag and stubbed:
//! real ION resolution requires a Sidetree node, which this crate documents
//! as an operator-side requirement.
//!
//! The high-level entry point is [`UniversalResolver`], which dispatches to
//! the right method-specific resolver based on the DID method.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod dereferencer;
pub mod did;
pub mod document;
pub mod error;
pub mod methods;
pub mod resolver;

pub use dereferencer::{DereferenceResult, dereference};
pub use did::{Did, DidMethod, DidUrl};
pub use document::{
    DidDocument, Jwk, Service, ServiceEndpoint, VerificationMethod,
    VerificationRelationship,
};
pub use error::DidError;
pub use resolver::{
    DocumentMetadata, Resolver, ResolutionMetadata, ResolutionResult,
    UniversalResolver,
};
