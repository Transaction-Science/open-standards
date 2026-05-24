//! # smart-byte-acdc
//!
//! Authentic Chained Data Containers (ACDC) — a verifiable credential
//! format from the KERI ecosystem, ingested into the Smart Byte
//! substrate as an Apache-2.0 Rust reference implementation.
//!
//! The format originates with Smith's *Authentic Chained Data
//! Containers* draft (IETF `draft-ssmith-acdc`) and is the credential
//! companion to KERI's identifier layer. Where [`smart_byte_vc`]
//! ingests the W3C VCDM family, this crate ingests the KERI-native
//! family:
//!
//! * [`acdc::Acdc`] — the credential body with `issuer`, `schema`,
//!   `registry`, attribute, and edge sections, each anchored by a
//!   SAID per spec §3.
//! * [`schema`] — ACDC's JSON-Schema-dialect validator, including
//!   support for the `oneOf` partial-disclosure construction used by
//!   selective disclosure.
//! * [`tel`] — Transaction Event Log carrying `RIP` (registry
//!   inception), `VRT` (registry rotation), `ISS` (credential
//!   issuance), and `REV` (revocation) events.
//! * [`ipex`] — Issuance / Presentation Exchange messages (`grant`,
//!   `admit`, `apply`, `offer`, `agree`, `spurn`) used to move ACDCs
//!   between controllers.
//! * [`edge`] — the edge section linking one ACDC to another by SAID,
//!   plus an in-memory graph + traversal utility.
//! * [`selective`] — selective disclosure derived from
//!   partially-disclosable schemas; an attribute section is split into
//!   a SAIDed digest tree and a holder may reveal a subset while
//!   preserving the credential's overall SAID.
//! * [`registry`] — a [`registry::CredentialRegistry`] trait plus an
//!   in-memory implementation tying credentials to their TEL.
//!
//! ACDC composes directly with the rest of the substrate: SAIDs come
//! from [`smart_byte_core::Said`] and the witness layer in
//! [`smart_byte_keri_witness`] can counter-sign TEL anchoring events
//! the same way it counter-signs KEL events.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod acdc;
pub mod edge;
pub mod error;
pub mod ipex;
pub mod registry;
pub mod schema;
pub mod selective;
pub mod tel;

pub use acdc::{Acdc, AcdcBuilder, AttributeSection, EdgeSection, RegistrySection, SchemaSection};
pub use edge::{Edge, EdgeGraph, EdgeOp};
pub use error::{AcdcError, Result};
pub use ipex::{IpexKind, IpexMessage};
pub use registry::{CredentialRegistry, InMemoryRegistry, RegistryState};
pub use schema::{AcdcSchema, SchemaDialect, SchemaType, validate_attributes};
pub use selective::{DisclosurePlan, SelectiveDisclosure, derive_disclosure};
pub use tel::{Tel, TelEvent, TelEventKind};

/// Canonical version string carried in every ACDC envelope's `v` field.
///
/// `ACDC10JSON` mirrors the KERI / ACDC versioning convention:
/// four-character family tag, two-digit major.minor, serialization
/// tag. v1.0, canonical JSON via RFC 8785 JCS.
pub const VERSION_STRING: &str = "ACDC10JSON";

/// Placeholder used in the `d` field during SAID derivation. 44
/// characters is the textual length of a base32-no-pad BLAKE3-256
/// digest, ensuring the placeholder occupies the same number of bytes
/// as the final SAID and the JCS encoding's byte layout is stable.
pub const SAID_PLACEHOLDER: &str = "############################################";
