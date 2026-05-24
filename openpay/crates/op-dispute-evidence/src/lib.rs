//! # `op-dispute-evidence` — chargeback lifecycle + evidence packaging
//!
//! Where [`op_dispute`] owns the abstract dispute record (id, amount,
//! status, opaque reason), this crate owns the *network-specific*
//! lifecycle and evidence-packaging machinery that turns a fresh
//! chargeback into either a won representment or a swallowed loss.
//!
//! ## What this crate owns
//!
//! - The card-network reason-code taxonomies for **Visa VCR**
//!   (chapters 10/11/12/13), **Mastercard Mastercom** (message
//!   reasons 4837/4853/etc. and the unified MIP 1240/1442 message
//!   shape), **American Express SafeKey + DRR**, **Discover DRR**,
//!   and **PayPal** ("item not received" / "significantly not as
//!   described" / "unauthorized").
//! - A canonical [`lifecycle::Phase`] state machine —
//!   `retrieval → first chargeback → representment → pre-arb →
//!   arbitration → final` — that maps each network's idiosyncratic
//!   stage names to a single workflow operators can code against.
//! - The [`evidence::EvidencePackage`] builder: type-checked
//!   bundling of AVS/CVV results, 3-D Secure 2 auth values,
//!   delivery confirmations, customer-comms transcripts, device /
//!   IP fingerprints, and the receipt itself.
//! - The Visa **CE3.0 Compelling Evidence 3.0** qualifier — given a
//!   disputed transaction and an evidence package, decide whether
//!   the merchant has the 2+ qualifying historical transactions
//!   that flip a "fraud" 10.4 chargeback to ineligible.
//! - A win-rate [`scoring::WinScore`] heuristic that the operator
//!   can use to triage which disputes are worth representing vs.
//!   accepting.
//!
//! ## What this crate does NOT own
//!
//! - **The PSP / acquirer wire protocol.** We model the lifecycle
//!   and the evidence shape; the actual `POST /disputes/{id}/evidence`
//!   to Stripe / Adyen / Worldpay / a direct VCR feed is the
//!   operator's adapter to write.
//! - **The ledger reversal.** When a representment is won, money
//!   un-claws back; that booking happens in `op-ledger` via the
//!   operator's `LedgerStore`, not here.
//! - **Decisioning policy.** [`scoring::WinScore`] is a heuristic,
//!   not a regulator. The operator may override.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod ce3;
pub mod error;
pub mod evidence;
pub mod lifecycle;
pub mod network;
pub mod reason_codes;
pub mod scoring;

pub use ce3::{Ce3Eligibility, Ce3Qualifier, QualifyingTransaction};
pub use error::{Error, Result};
pub use evidence::{EvidenceItem, EvidencePackage, EvidencePackageBuilder};
pub use lifecycle::{LifecycleEvent, LifecycleMachine, Phase};
pub use network::{
    AmexReasonCode, DiscoverReasonCode, MastercardReasonCode, Network, PayPalReasonCode,
    VisaReasonCode,
};
pub use reason_codes::{EvidenceRequirement, ReasonCode, ReasonCodeCatalog};
pub use scoring::{WinScore, WinScoreBand};
