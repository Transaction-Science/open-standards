//! # `op-rails-a2a` — Account-to-account / instant-payment rails
//!
//! The economic core of `OpenPay`. Cards through Hyperswitch cost 2.5–3.5%
//! per transaction; A2A rails cost cents flat. This crate is where the
//! margin gets unlocked.
//!
//! ## Rails covered
//!
//! | Rail                | Region | Currency | Settlement | Verified spec source                                            |
//! |---------------------|--------|----------|------------|------------------------------------------------------------------|
//! | `FedNow`              | US     | USD      | <20 s      | `FedNow` Service Operating Procedures v3.2 (June 2025)             |
//! | PIX (SPI)           | Brazil | BRL      | <10 s      | Bacen Manual de Iniciação v2.6.3; aws-samples/pix-proxy-samples  |
//! | SEPA SCT Inst (RT1) | EU     | EUR      | <10 s      | EPC SCT Inst Interbank IGs 2019 v1.0; EBA Clearing RT1 docs      |
//! | SEPA SCT Inst (TIPS)| EU     | EUR      | <10 s      | ECB TIPS UDFS; alignment to EPC 2025 SCT Inst Inter PSP IG       |
//!
//! UPI (India) and RTP (US, The Clearing House) are feature-stubbed for
//! Phase 5.1 / 5.2.
//!
//! ## Transport summary
//!
//! Different rails have radically different transport layers:
//!
//! - **`FedNow`** — IBM MQ over `FedLine` Direct/Advantage. APIs reachable
//!   over `FedLine` VPN with FRB-issued API certificates. The
//!   `Authorized Connection Profile` is mandatory; merchants connect
//!   through a Service Provider, not directly.
//! - **PIX** — HTTPS to Bacen's ICOM messaging system using **mTLS**
//!   (mandatory) plus OAuth 2.0 with client-certificate-bound access
//!   tokens (RFC 8705). XML payloads digitally signed; AWS reference
//!   architecture uses `CloudHSM` as the mandatory signing path.
//! - **RT1** — HTTPS to EBA Clearing's gateway with mTLS. ISO 20022
//!   pacs.008.001.08 single transactions only (no bulk).
//! - **TIPS** — Direct settlement in central-bank money via the ECB.
//!   Same message shapes as RT1 with a different counterparty BIC.
//!
//! ## Architecture
//!
//! The [`A2aAcquirer`] trait mirrors the `CardAcquirer` pattern from
//! `op-rails-card`. Each driver implements it. The orchestrator holds
//! `Box<dyn A2aAcquirer>` and never knows which rail it's talking to.
//!
//! Each driver:
//! 1. Builds an ISO 20022 pacs.008 via [`op_iso20022::CreditTransferBuilder<P>`]
//!    where `P` is the rail's [`Profile`](op_iso20022::profile::Profile).
//! 2. Wraps it in the rail-specific transport (HTTP body, MQ envelope, ...).
//! 3. Parses the pacs.002 response back through the same builder type.
//! 4. Maps the rail's status reason codes to the normalized [`A2aStatus`].
//!
//! This means **adding a new rail is a Profile + a transport adapter** —
//! no changes to the ISO 20022 layer, no changes to the trait, no
//! changes to the orchestrator.
//!
//! ## What this crate does NOT do
//!
//! - It does not establish `FedLine` VPN connections. That's the
//!   operator's network deployment.
//! - It does not provision API certificates with the Fed or Bacen.
//!   Operators get those through the standard onboarding processes.
//! - It does not run the CloudHSM-style signing infrastructure for
//!   PIX. We expose a `Signer` trait; operators plug in their own.
//! - It is not a payments license. Operators handle their own
//!   licensing / MTL / payfac relationships.
//!
//! `OpenPay` is the protocol stack; the operator handles compliance.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]

pub mod acquirer;
pub mod error;
pub mod signer;
pub mod xml_common;

#[cfg(feature = "fednow")]
pub mod fednow;
#[cfg(feature = "pix")]
pub mod pix;
#[cfg(feature = "sepa-instant")]
pub mod sepa_instant;

pub use acquirer::{
    A2aAcquirer, A2aDecision, A2aStatus, CreditTransferReq, ParticipantId, StatusQueryReq,
};
pub use error::{Error, Result};
pub use signer::{NoOpSigner, Signer};
