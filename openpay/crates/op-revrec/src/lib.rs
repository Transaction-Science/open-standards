//! # `op-revrec` — Revenue recognition for `OpenPay`
//!
//! ASC 606 / IFRS 15 revenue-recognition engine for payment platforms.
//! The crate implements the canonical five-step model and produces an
//! auditable schedule + ledger from a typed contract description.
//!
//! ## The five steps (verbatim from ASC 606 / IFRS 15)
//!
//! 1. **Identify the contract** with a customer.
//!    → [`contract::Contract`].
//! 2. **Identify the performance obligations** in the contract.
//!    → [`contract::PerformanceObligation`] (one per distinct
//!    promised good or service).
//! 3. **Determine the transaction price.**
//!    → [`contract::TransactionPrice`], with variable-consideration
//!    components constrained per [`variable::VariableConsideration`].
//! 4. **Allocate the transaction price** to the performance
//!    obligations on a relative standalone-selling-price basis.
//!    → [`schedule::allocate_transaction_price`].
//! 5. **Recognize revenue** as (or when) the entity satisfies the
//!    performance obligations.
//!    → [`schedule::generate`] for point-in-time, straight-line,
//!    output-milestone, input-percent-complete; and
//!    [`schedule::usage_recognition`] for the consumption-based case.
//!
//! ## Post-inception machinery
//!
//! - [`ledger::DeferredRevenueLedger`] — append-only subledger that
//!   tracks open deferrals, recognitions, and refunds. Ties to
//!   [`op_core::Payment`] via a correlating `payment_id`. Ships with
//!   an [`ledger::InMemoryLedger`] reference implementation; real GL
//!   backends (NetSuite / Workday / QuickBooks) plug in behind the
//!   trait under the `live` feature.
//! - [`modification::Modification`] — Type-I (separate contract),
//!   Type-II (terminate + new), Type-III (cumulative catch-up) per
//!   ASC 606-10-25-12 / -25-13.
//! - [`principal_agent`] — indicators per ASC 606-10-55-37 through
//!   55-40 driving gross-vs-net presentation
//!   ([`contract::Presentation`]).
//! - [`ledger::translate`] — multi-currency translation at the
//!   recognition date.
//!
//! ## What this crate does NOT do
//!
//! - **Originate the GL postings on the customer's books.** We compute
//!   the recognition events; the operator's accounting system applies
//!   them.
//! - **Authoritative tax classification.** Sales-tax / VAT calculation
//!   lives in `op-tax`; we only consume the principal-vs-agent
//!   decision as it affects gross-vs-net revenue presentation.
//! - **Revenue-leakage reporting / dashboards.** Out of scope; ship
//!   the postings through your BI of choice.
//!
//! ## Decimal discipline
//!
//! All amounts are exact `i64` minor units inside [`op_core::Money`].
//! Allocation, milestone and percent-complete math goes through
//! [`rust_decimal::Decimal`] (28 significant digits) and rounds with
//! round-half-up to integer minor units. The last entry in any
//! schedule absorbs the integer-division remainder so totals are exact.
//!
//! ## Determinism
//!
//! The recognition engine is pure: given the same contract and the
//! same schedule, it produces the same postings. The ledger trait is
//! async to admit network-backed GL implementations, but the in-memory
//! reference implementation is deterministic up to the wall-clock
//! `created_at` timestamp on each posting.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
// Allowed deliberately. Same idioms as op-tax / op-connect:
// - module_name_repetitions: `ContractId` inside `contract`, etc., matches
//   the rest of the workspace.
// - missing_errors_doc / missing_panics_doc: errors are documented on the
//   `Error` enum; per-function repetition is noise.
// - similar_names: revenue math uses short paired identifiers (ssp, sum,
//   spp, tp) that pedantic renaming would hurt.
// - too_many_lines: the schedule generators are cohesive units of math
//   that resist splitting without harming readability.
// - doc_markdown: prose uses domain proper nouns (ASC, IFRS, NetSuite,
//   Workday, QuickBooks) that are not Rust items.
// - cast_possible_wrap / cast_sign_loss / cast_possible_truncation: we
//   carefully bound i64<->u64 conversions in `variable.rs` and
//   `schedule.rs`; the casts are intentional.
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::doc_lazy_continuation)]

pub mod contract;
pub mod error;
pub mod ledger;
pub mod modification;
pub mod principal_agent;
pub mod schedule;
pub mod variable;

pub use contract::{
    Contract, ContractId, Milestone, ObligationId, PercentCompleteSnapshot,
    PerformanceObligation, Presentation, RecognitionPattern, TransactionPrice,
};
pub use error::{Error, Result};
pub use ledger::{
    Balances, DeferredRevenueLedger, InMemoryLedger, Posting, PostingKind, translate,
};
pub use modification::{Modification, ModificationProposal, classify as classify_modification};
pub use principal_agent::{Indicators, classify as classify_principal_agent};
pub use schedule::{
    RecognitionSchedule, ScheduleEntry, allocate_transaction_price, entry_money, generate,
    usage_recognition,
};
pub use variable::{EstimationMethod, Outcome, VariableConsideration};
