//! # `op-bin` — BIN/IIN database and card-network classification
//!
//! Maps the first 6-8 digits of a Primary Account Number (PAN) —
//! the **Bank Identification Number** (BIN), also called the
//! **Issuer Identification Number** (IIN) per ISO/IEC 7812 — to
//! the issuing card network, card type, country, and Durbin
//! regulated-debit status.
//!
//! ## ISO/IEC 7812
//!
//! As of 2022, ISO/IEC 7812 mandates **eight-digit IINs** for
//! newly-allocated ranges. Most legacy ranges remain six digits;
//! networks routinely subdivide a 6-digit prefix into 8-digit
//! sub-ranges. This crate represents every range as a half-open
//! interval `[low, high)` over the **first-8-digit prefix** of a
//! PAN, so the lookup path is uniform regardless of whether the
//! issuer was allocated a 6-, 7-, or 8-digit IIN.
//!
//! ## What the crate ships
//!
//! - [`Bin`] — a validated 6-to-8-digit numeric prefix.
//! - [`BinRange`] — half-open prefix interval annotated with
//!   network, card type, issuer country, and Durbin flag.
//! - [`RangeTree`] — sorted interval-tree lookup; `O(log N)`
//!   point queries over a few thousand ranges.
//! - [`CardNetwork`] + [`classify`](network::classify) — given a
//!   `Bin`, classify the network by structural prefix rules from
//!   `network_ranges` (Visa = `4xxxxx`, Mastercard 51-55 / 2221-2720,
//!   Amex 34/37, Discover, JCB, Diners, UnionPay, RuPay, Maestro,
//!   Elo, Mir, Troy).
//! - [`luhn::is_valid`] / [`luhn::check_digit`] — ISO/IEC 7812-1
//!   Annex B check-digit computation.
//! - [`CardType`] — credit / debit / prepaid / charge.
//! - [`IssuerCountry`] — ISO 3166-1 alpha-2 wrapper.
//! - [`durbin::is_regulated`] — Federal Reserve Regulation II
//!   merchant-side classification helper.
//!
//! ## PCI scope
//!
//! `op-bin` is deliberately **only** about the BIN. It never
//! takes, stores, or returns a full PAN. The API surface accepts
//! at most 8 digits anywhere a `Bin` is constructed; downstream
//! callers in the OpenPay reference stack hand us
//! `pan[..pan.len().min(8)]` and discard the remainder before the
//! BIN lookup runs.
//!
//! ## What the crate does NOT do
//!
//! - **No network lookups.** The static tables in
//!   [`network_ranges`] are structural prefix rules drawn from
//!   public ISO/IEC 7812 documentation and network developer
//!   docs; they are not a substitute for a commercial BIN feed
//!   (e.g. Bin Codes, BinList, FreeBINChecker) and they do not
//!   resolve individual issuing banks.
//! - **No PAN handling.** See PCI note above.
//! - **No I/O.** All tables are compiled in; operators ship
//!   updates by recompiling.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bin;
pub mod card_type;
pub mod durbin;
pub mod error;
pub mod issuer_country;
pub mod luhn;
pub mod network;
pub mod network_ranges;
pub mod range_tree;

pub use bin::{Bin, BinRange};
pub use card_type::CardType;
pub use error::{Error, Result};
pub use issuer_country::IssuerCountry;
pub use network::{classify, CardNetwork};
pub use range_tree::RangeTree;
