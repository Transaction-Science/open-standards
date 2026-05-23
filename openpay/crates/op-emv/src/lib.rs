//! # `op-emv` — EMV BER-TLV codec
//!
//! Parses Tag-Length-Value bundles per **EMV Book 3, Annex B** (BER-TLV).
//! This is the binary format used everywhere in chip-card and Tap-to-Pay
//! payloads: the data returned by Apple's `ProximityReader` on iOS, the
//! Android `IsoDep` APDU responses, and what every EMVCo-certified
//! terminal kernel produces.
//!
//! ## Why a custom parser
//!
//! Generic ASN.1 BER libraries handle a superset of what EMV uses, are
//! `std`-heavy, and don't enforce EMV's restrictions:
//!
//! - **Tag**: 1 to 4 bytes. First byte's lower 5 bits = `0x1F` means
//!   multi-byte tag; subsequent bytes' bit 8 = 1 means another follows.
//!   EMV caps tags at 4 bytes; we enforce that.
//! - **Length**: short form (one byte, 0–127) or long form (`0x8N`
//!   followed by N length bytes). EMV bans indefinite length (`0x80`
//!   alone). We reject it.
//! - **Constructed vs primitive**: bit 6 of the first tag byte. A
//!   constructed TLV's value is itself a sequence of TLVs.
//! - **Padding**: `0x00` bytes between TLVs are allowed and skipped.
//!
//! ## API shape
//!
//! Two layers:
//!
//! - [`stream`] — zero-alloc, no-std iterator. Each step yields a
//!   borrowed [`TlvRef`] pointing into the input slice. This is what
//!   the FFI layer hands to platform shells.
//! - [`tree`] — `std`-only builder that turns a flat stream into a
//!   [`Tlv`] tree with owned values. Useful for diagnostics, caching,
//!   and the AI-fraud pre-processor in `op-fraud`.
//!
//! ## Safety
//!
//! `#![forbid(unsafe_code)]`. All indexing is bounds-checked.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

extern crate alloc;

pub mod error;
pub mod stream;
pub mod tag;
#[cfg(feature = "std")]
pub mod tree;

pub use error::{Error, Result};
pub use stream::{TlvIter, TlvRef};
pub use tag::{Tag, TagClass};
#[cfg(feature = "std")]
pub use tree::Tlv;
