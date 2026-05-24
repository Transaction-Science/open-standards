//! # `op-3ds2` — 3-D Secure 2.x protocol stack
//!
//! Implements the EMVCo 3-D Secure protocol versions 2.1.0, 2.2.0, and
//! 2.3.0 — the Requestor / 3DS Server / Directory Server / Access
//! Control Server message catalogue, plus the SCA-exemption evaluator
//! required for PSD2 RTS and the 2026 PSD3 update.
//!
//! ## What lives here
//!
//! - [`message`] — the AReq / ARes / CReq / CRes / RReq / RRes message
//!   catalogue, with serde JSON encoders aligned with the EMVCo
//!   specification.
//! - [`version`] — the per-version field capability matrix.
//! - [`directory_server`] — the [`directory_server::DirectoryServer`]
//!   trait and per-scheme adapters
//!   ([`visa_ds`], [`mc_ds`], [`amex_ds`], [`discover_ds`], [`jcb_ds`]).
//! - [`acs`] — the issuer-side Access Control Server primitives.
//! - [`challenge`] — the CReq/CRes challenge loop and the three
//!   challenge modes (HTML, native-app, OOB).
//! - [`exemption`] — the SCA-exemption evaluator.
//! - [`risk`] — the merchant-risk-indicator and account-info payloads.
//! - [`fingerprint`] — browser-fingerprint collection / parsing.
//! - [`decoupled`] — the decoupled-authentication polling loop.
//! - [`data_only`] — the data-only / no-challenge flow.
//! - [`threeri`] — the 3DS Requestor Initiated subsequent-transaction
//!   flow.
//! - [`auth_response`] — the transaction-status, ECI, and CAVV
//!   primitives.
//!
//! ## What this crate does NOT do
//!
//! - **No production scheme keys are bundled.** Live operators must
//!   provision their own mTLS material and DS URLs from each scheme's
//!   developer portal. The per-scheme DS adapters compile against
//!   stub fixtures suitable for local testing only.
//! - **No I/O at construction.** Like the rest of OpenPay, this crate
//!   is honest about which code paths talk to the network: every
//!   network-bound surface is a trait the operator implements.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::pedantic)]
// Names like `AReq`, `CReq`, `RReq` map straight to the EMVCo spec.
// `doc_markdown` flags every spec-faithful uppercase token; we
// allow it crate-wide to keep the docs readable as written.
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::inconsistent_digit_grouping,
    clippy::unreadable_literal,
    clippy::match_same_arms,
    clippy::too_many_lines,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::single_match_else,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::default_trait_access,
    clippy::if_not_else,
    clippy::redundant_closure_for_method_calls,
    clippy::items_after_statements,
    clippy::single_char_pattern,
    clippy::needless_for_each,
    clippy::option_if_let_else,
    clippy::trivially_copy_pass_by_ref,
    clippy::manual_let_else,
    clippy::ignored_unit_patterns,
    clippy::map_unwrap_or,
    clippy::elidable_lifetime_names,
    clippy::needless_raw_string_hashes,
    clippy::uninlined_format_args
)]

pub mod acs;
pub mod amex_ds;
pub mod auth_response;
pub mod challenge;
pub mod data_only;
pub mod decoupled;
pub mod directory_server;
pub mod discover_ds;
pub mod error;
pub mod exemption;
pub mod fingerprint;
pub mod jcb_ds;
pub mod mc_ds;
pub mod message;
pub mod risk;
pub mod threeri;
pub mod version;
pub mod visa_ds;

pub use acs::{AcsAuthMethod, AcsConfig, AcsServer};
pub use auth_response::{Cavv, Eci, TransactionStatus};
pub use challenge::{ChallengeMode, ChallengeRequest, ChallengeResult, ChallengeSession};
pub use data_only::DataOnlyRequest;
pub use decoupled::{DecoupledPollResult, DecoupledSession};
pub use directory_server::{DirectoryServer, DsRoute, VersionCheckResponse};
pub use error::{Error, Result};
pub use exemption::{
    EligibleExemption, ExemptionContext, FraudRateBracket, SecondFactorClass, evaluate,
};
pub use fingerprint::{BrowserFingerprint, FingerprintMethod, fingerprint_collector_script};
pub use message::{
    ARes, AReq, CRes, CReq, DeviceChannel, ErrorMessage, MessageCategory, RRes, RReq,
    ThreeDsMessage,
};
pub use risk::{AccountInfo, BrowserInfo, DeliveryTimeFrame, MerchantRiskIndicator, RequestorRiskData};
pub use threeri::{ThreeRiCategory, ThreeRiRequest};
pub use version::{FieldRule, ProtocolVersion};
