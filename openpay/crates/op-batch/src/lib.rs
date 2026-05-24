//! # `op-batch` — batch payment rails for OpenPay
//!
//! `op-settlement` produces instant / single-message payouts. The
//! rest of the regulated-payments world still ingests **files of
//! payments**: thousands of credits or debits in one shot, dropped
//! into a bank SFTP folder by cutoff time, results returned hours
//! or days later as a separate file.
//!
//! This crate covers the four canonical batch rails:
//!
//! | Rail | Format | Region |
//! |------|--------|--------|
//! | **NACHA ACH** | Fixed-width records (94 chars / line) | US |
//! | **SEPA** | ISO 20022 XML (`pain.001`, `pain.008`) | EU |
//! | **Wire** | SWIFT MT103 / MT202, ISO 20022 `pacs.008`/`pacs.009`, CHIPS | US / cross-border |
//! | **Bacs** | Fixed-width records | UK |
//!
//! For each rail we ship:
//!
//! 1. A typed in-memory model.
//! 2. An encoder (produce the file bytes) and decoder (parse a
//!    received file — used for return / reject parsing and for tests).
//! 3. File-naming conventions (every scheme has its own; banks
//!    reject unrecognised names).
//! 4. Exception handling: NACHA returns (`R01`..`R99`), SEPA
//!    R-transactions (reject / return / reversal / refund), wire
//!    investigations.
//! 5. Reconciliation: match outbound batch entries to inbound bank
//!    statement lines (`camt.053` for ISO 20022 rails, NACHA prenote
//!    file or BAI2 for ACH).
//!
//! On top of that, an [`orchestrator::BatchOrchestrator`] holds one
//! [`orchestrator::BatchProcessor`] per rail, knows each rail's
//! cutoff windows, and submits or fetches returns on demand.
//!
//! ## What this crate is NOT
//!
//! - **Not a bank connection.** Default submission writes a file
//!   to a local spool the operator's SFTP agent uploads. The `sftp`
//!   feature can be flipped on for direct delivery, but the
//!   defaults assume operators run their own queue.
//! - **Not a settlement engine.** [`op_settlement`](../op_settlement)
//!   stays the source of truth for cutoff windows, holdbacks, and
//!   batch lifecycle. `op-batch` is the **rail-format** layer; the
//!   settlement engine produces `Batch`es, `op-batch` produces
//!   `RailFile`s.
//! - **Not a ledger.** Reconciliation only emits
//!   [`reconciliation::Match`] decisions; persisting them lives
//!   in `op-ledger` / `op-reconciliation`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::similar_names)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::format_push_string)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::needless_continue)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::redundant_closure_for_method_calls)]

pub mod bacs;
pub mod error;
pub mod exception;
pub mod file_io;
pub mod nacha;
pub mod orchestrator;
pub mod reconciliation;
pub mod sepa;
pub mod wire;

pub use error::{Error, Result};
pub use exception::{Exception, ExceptionAction, ExceptionCode, PaymentId};
pub use file_io::{FileNaming, Submission, SubmissionSink, SpoolSink};
pub use nacha::{NachaBatch, NachaFile, NachaProfile, NachaReturn, ReturnCode, SecCode};
pub use orchestrator::{
    BatchOrchestrator, BatchProcessor, CutoffSchedule, RailReceipt, Scheduler,
};
pub use reconciliation::{Match, ReconcileSource, ReconciliationReport};
pub use sepa::{SepaCreditTransfer, SepaDirectDebit, SepaScheme};
pub use wire::{WireMessage, WireFormat};

/// Which batch rail produced a given message.
///
/// Used everywhere this crate crosses the rail boundary
/// (orchestrator routing, exception tagging, reconciliation
/// source matching).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BatchRail {
    /// US ACH (NACHA Operating Rules).
    Nacha,
    /// SEPA Credit Transfer (`pain.001`).
    SepaCt,
    /// SEPA Direct Debit (`pain.008`).
    SepaDd,
    /// US Fedwire (ISO 20022 `pacs.008` since March 2025) or
    /// legacy SWIFT MT103.
    Fedwire,
    /// SWIFT cross-border (MT103 / MT202 / `pacs.008` / `pacs.009`).
    Swift,
    /// CHIPS — The Clearing House clearing system in NY.
    Chips,
    /// UK Bacs Direct Credit / Direct Debit.
    Bacs,
}

impl BatchRail {
    /// Short identifier used in filenames and log lines.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Nacha => "nacha",
            Self::SepaCt => "sepa-ct",
            Self::SepaDd => "sepa-dd",
            Self::Fedwire => "fedwire",
            Self::Swift => "swift",
            Self::Chips => "chips",
            Self::Bacs => "bacs",
        }
    }

    /// True if this rail's file format is ISO 20022 XML.
    #[must_use]
    pub const fn is_iso20022(self) -> bool {
        matches!(self, Self::SepaCt | Self::SepaDd | Self::Fedwire | Self::Swift)
    }
}
