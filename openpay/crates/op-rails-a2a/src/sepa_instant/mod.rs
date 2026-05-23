//! SEPA SCT Inst driver.
//!
//! Routes pacs.008.001.08 messages to either:
//! - **EBA Clearing RT1** — pan-European, 94 participants
//! - **ECB TIPS** — central-bank settlement
//!
//! Both use the same message body shape per EPC SCT Inst Interbank IGs
//! 2019 v1.0; they differ in transport endpoint and a few BIC routing
//! conventions.
//!
//! ## Verified scheme rules
//!
//! Per the EPC SCT Inst Interbank IG and TIPS UDFS:
//!
//! - `LocalInstrument.Code` MUST be `"INST"` (this is what makes the
//!   transfer "instant" at the scheme level).
//! - `SettlementMethod.Cd` MUST be `"CLRG"` for RT1; TIPS accepts
//!   `"CLRG"` or `"INGA"`.
//! - Currency MUST be EUR.
//! - **Number of transactions per pacs.008 is exactly 1.** RT1 and TIPS
//!   reject bulk pacs.008.
//! - End-to-end SLA: 10 seconds (changing to 5/7/9-second sub-timelines
//!   per the 2025 SEPA Regulation amendment).
//!
//! ## NOT covered yet
//!
//! - The 22 November 2026 structured-address-only requirement. We emit
//!   unstructured names today; structured addresses are Phase 5.1.
//! - OCT Inst (one-leg-out) flows; RT1 supports these but most operators
//!   route domestic + same-PSP first.
//! - SDD R-transaction recall flows.

pub mod client;
pub mod status_map;

pub use client::{SepaInstantBackend, SepaInstantClient};
