//! # `op-screening` — Sanctions screening for `OpenPay`
//!
//! Every payment processor that touches USD, EUR, GBP, AUD, or CAD value
//! is, by regulation, required to screen counterparties against the
//! sanctions lists published by the relevant treasury authority. The
//! big eight are:
//!
//! - **OFAC SDN** (US Treasury, Office of Foreign Assets Control)
//! - **OFAC Consolidated** (non-SDN consolidated list)
//! - **EU Consolidated** (Council of the European Union)
//! - **UN Consolidated** (UN Security Council)
//! - **HMT** (HM Treasury, United Kingdom)
//! - **DFAT** (Department of Foreign Affairs and Trade, Australia)
//! - **SEMA** (Special Economic Measures Act, Canada)
//! - **MOF** (Ministry of Finance, Japan)
//!
//! Each list publishes a daily XML or JSON dump. This crate ingests
//! them, normalises names, builds an in-memory bloom + inverted-index
//! pair for fast O(1)-ish lookup, and exposes a [`Screener`] that
//! returns scored hits against a configurable similarity threshold.
//!
//! On top of list screening, BSA / FinCEN reporting helpers detect
//! structuring (CTR) and suspicious-activity (SAR) patterns over a
//! window of transactions.
//!
//! Every call to [`Screener::screen`] is recorded into a cryptographically
//! chained audit log (Ed25519 signatures, BLAKE3 hash chain) so that
//! regulators can inspect the screening history of any payment
//! attempt after the fact.
//!
//! ## What this crate does NOT do
//!
//! - **No production network calls in tests.** The list URLs are
//!   public-record HTTP endpoints; fixtures under
//!   `tests/fixtures/` exercise the parsers offline.
//! - **No human review UI.** The screener returns
//!   [`screener::ScreenDecision::AmbiguousNeedsReview`] for borderline
//!   scores; operators wire their own review queue on top.
//! - **No KYC / customer onboarding flow.** Sanctions screening is one
//!   component of KYC; the rest (document verification, watchlist
//!   adverse media, beneficial-ownership inference) lives elsewhere.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
// Pedantic / nursery lints we explicitly allow for this crate's character:
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::similar_names)]
#![allow(clippy::single_char_pattern)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::single_match_else)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::many_single_char_names)]
// Some match arms intentionally repeat (per-Cyrillic / per-letter mappings
// read as a literal table); collapsing them would lose the row-form
// readability that makes the table auditable.
#![allow(clippy::match_same_arms)]
// Boxed-future trait pattern uses explicit lifetimes for clarity.
#![allow(clippy::elidable_lifetime_names)]
// Multiply-add suggestions hurt readability of weighted blends and Jaro-Winkler.
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::unused_async)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::needless_collect)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::unnested_or_patterns)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::redundant_clone)]
#![allow(clippy::useless_let_if_seq)]
#![allow(clippy::missing_const_for_thread_local)]
#![allow(clippy::manual_assert)]

pub mod audit;
pub mod bsa;
pub mod error;
pub mod lists;
pub mod matching;
pub mod normalize;
pub mod screener;
pub mod storage;
pub mod updater;

pub use audit::{AuditEntry, AuditLog, AuditVerifyError};
pub use bsa::{CtrHelper, SarHelper, Transaction, Trigger, TriggerKind};
pub use error::{Error, Result};
pub use lists::{
    Address, CountryCode, EntityType, Identification, IdentificationKind, SanctionedEntity,
    SanctionsList,
};
pub use matching::{MatchMethod, MatchScore, MatchedField, screen};
pub use normalize::{NormalizedName, normalize};
pub use screener::{
    ScreenDecision, ScreenRequest, ScreenResult, Screener, ScreenerConfig,
};
pub use storage::{EntityRef, SanctionsIndex};
pub use updater::{
    EuUpdater, HmtUpdater, ListUpdater, OfacUpdater, UnUpdater,
};
