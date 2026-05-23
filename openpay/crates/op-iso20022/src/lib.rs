//! # `op-iso20022`
//!
//! Typed, validated facade over the upstream `open-payments-iso20022` and
//! `open-payments-fednow` crates. Three responsibilities:
//!
//! 1. **Idiomatic constructors** — build legal messages without learning
//!    the full ISO 20022 schema. Each rail profile (`FedNow`, RTP, PIX,
//!    SEPA Instant) ships a typed `Builder` that only exposes the fields
//!    the scheme accepts.
//! 2. **Validation** — every constructed message passes scheme-specific
//!    rules (UETR format, mandatory agents, charge bearer codes, status
//!    code subsets) before it can be serialized.
//! 3. **Bidirectional codec** — XML on the wire, in-memory `Document`
//!    inside the process. Round-trip is the conformance contract: every
//!    sample message in `vectors/` must parse, re-serialize, and equal
//!    the canonical form.
//!
//! ## What we re-export
//!
//! The upstream crate exposes a top-level `Document` enum with one
//! variant per message version (e.g. `FIToFICustomerCreditTransferV08`).
//! We surface only the variants `OpenPay` currently supports and provide
//! version-agnostic aliases (`Pacs008`, `Pacs002`, ...) that point at the
//! version chosen by the active profile.
//!
//! ## Module layout
//!
//! - [`error`] — `Error`, `Result`. One sealed enum.
//! - [`message`] — version-agnostic aliases, `Message` enum, helpers.
//! - [`codec`] — XML serialize / deserialize.
//! - [`bah`] — Business Application Header (head.001).
//! - [`status`] — `TransactionStatus`, `StatusReason` (ISO 20022 code
//!   lists: ACTC, ACSC, RJCT, PDNG, ...).
//! - [`profile`] — per-rail profiles: [`profile::FedNow`],
//!   [`profile::Rtp`], [`profile::Pix`], [`profile::SepaInstant`].
//! - [`builder`] — high-level builders that wrap profile rules.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod bah;
pub mod builder;
pub mod codec;
pub mod error;
pub mod message;
pub mod profile;
pub mod statement;
pub mod status;

pub use bah::BusinessApplicationHeader;
pub use builder::{BuiltCreditTransfer, CreditTransferBuilder};
pub use codec::{from_xml, to_xml};
pub use error::{Error, Result};
pub use message::{Message, MessageKind};
pub use statement::StatementEntry;
pub use status::{StatusReason, TransactionStatus};
