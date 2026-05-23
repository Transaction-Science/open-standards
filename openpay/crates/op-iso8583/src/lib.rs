//! # `op-iso8583`
//!
//! ISO 8583 codec for OpenPay. Used by acquirers and PSPs that connect
//! **directly** to a card network's authorization rail (Visa Base I,
//! Mastercard MDS, Amex GNS, Discover, JCB) rather than via a hosted
//! processor. This is the bottom of the card stack — the wire format
//! the network actually speaks.
//!
//! ## What's here
//!
//! - [`message::Iso8583Message`] — typed message frame (MTI + bitmaps + DEs).
//! - [`bitmap::Bitmaps`] — primary + secondary bitmap codec.
//! - [`fields`] — DE catalog and typed accessors (PAN, amount, STAN,
//!   RRN, response code, EMV blob, MAC, ...).
//! - [`codec`] — wire-format codecs (BCD, ASCII, EBCDIC, binary,
//!   LLVAR, LLLVAR).
//! - [`dialect`] — per-network profiles ([`dialect::VisaBaseI`],
//!   [`dialect::MastercardMds`], [`dialect::AmexGns`],
//!   [`dialect::DiscoverCard`], [`dialect::Jcb`]).
//! - [`network_mgmt`] — 0800/0810 sign-on / echo / key-change / cutover.
//! - [`emv`] — DE 55 EMV TLV walker.
//! - [`error`] — sealed error enum.
//!
//! ## Round-trip contract
//!
//! Every message in `tests/fixtures/` must encode→decode→encode to the
//! exact same byte sequence. Network conformance tests use this as the
//! gatekeeper.
//!
//! ## Safety & quality
//!
//! - `#![forbid(unsafe_code)]`.
//! - No `unwrap` / `expect` outside `#[test]`.
//! - All decoders are bounds-checked; no panics on malformed input.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
// Pedantic carve-outs that don't fit ISO 8583's byte-level codecs:
//   - cast_possible_truncation: byte/nibble math is the whole job here;
//     every narrowing cast is byte-aligned by construction (mod 256,
//     mod 16, etc.) and adding `u8::try_from` everywhere just buries
//     the intent.
//   - cast_sign_loss / cast_lossless: same byte-arithmetic story.
//   - module_name_repetitions: matches the rest of the OpenPay tree.
//   - missing_errors_doc: every `Result`-returning function carries
//     its error documentation in the prose just above; clippy's
//     doc-section heuristic is too narrow.
//   - doc_markdown: card-network proper nouns (VisaNet, AmexNet,
//     V.I.P., GNS, MDS, K_round) trip the "code-fence me" detector;
//     wrapping them is hostile to readers who recognize them as
//     industry terms.
//   - manual_is_multiple_of: `% N != 0` is what the wire spec says;
//     rewriting as `.is_multiple_of` reads worse next to the ISO 8583
//     prose that describes the modular structure.
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::manual_is_multiple_of)]

pub mod bitmap;
pub mod codec;
pub mod dialect;
pub mod emv;
pub mod error;
pub mod fields;
pub mod message;
pub mod network_mgmt;

pub use bitmap::Bitmaps;
pub use dialect::{AmexGns, Dialect, DiscoverCard, Jcb, MastercardMds, VisaBaseI};
pub use emv::{EmvTag, EmvTlv};
pub use error::{Error, Result};
pub use fields::{Encoding, FieldSpec, FieldValue, LengthRule, default_catalog};
pub use message::{Iso8583Message, Mti};
pub use network_mgmt::{NetworkMgmtCode, build_request, build_response};
