//! # `op-fx` — Foreign-exchange primitives for `OpenPay`
//!
//! Integer-exact currency conversion with deterministic rounding,
//! pluggable quote sources, and TTL-based caching. The reference
//! stack ships no network clients — operators wire their preferred
//! FX feed (Wise, Open Exchange Rates, internal mid-market, bank
//! API) behind the [`QuoteProvider`] trait.
//!
//! ## Why integer math
//!
//! FX is one of the most-cited examples of "don't use floats for
//! money." A 0.0001 rounding error on `$10M` is $1000 the bank
//! cares about. We represent rates as
//! **parts per million** (`u64`): `1.000000 = 1_000_000`. Six
//! decimal places of precision is the FX industry standard for
//! spot rates; carrying more is meaningless given bid/ask spreads.
//!
//! Conversion math: `target_minor = source_minor * rate_ppm /
//! 1_000_000`, with a caller-chosen [`RoundingMode`] for the
//! remainder. No floats anywhere in the conversion pipeline.
//!
//! ## What the crate does NOT do
//!
//! - **No HTTP client.** [`QuoteProvider`] is the seam; the only
//!   concrete impls shipped are [`StaticQuoteProvider`] (fixed
//!   rate table, for tests + operator-locked rates) and
//!   [`CachedQuoteProvider`] (TTL wrapper).
//! - **No spread / fee modeling.** Operators apply their own
//!   spread by adjusting the rate they hand to the provider.
//! - **No triangulation.** A USD/JPY quote is direct; we don't
//!   auto-route via USD/EUR + EUR/JPY. Operators who need cross-
//!   rates assemble them themselves.
//! - **No hedging engine.** Quote validity windows are reported,
//!   not enforced — callers check `valid_until_unix_secs`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]

pub mod convert;
pub mod error;
pub mod provider;
pub mod quote;

pub use convert::{RoundingMode, convert};
pub use error::{Error, Result};
pub use provider::{CachedQuoteProvider, QuoteProvider, StaticQuoteProvider};
pub use quote::Quote;
