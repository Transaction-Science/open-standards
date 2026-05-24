//! # `op-connect` — Platform / marketplace tier for OpenPay
//!
//! `op-connect` is the Stripe-Connect / Adyen-MarketPay / Square-Capital
//! equivalent inside OpenPay. A platform that wants to onboard
//! sub-merchants, KYB them, split payments across multiple
//! beneficiaries, run a payout schedule, hold rolling reserves, and
//! file 1099-Ks plugs this crate in between [`op-core`](../op_core)
//! (payment primitives) and the rail-driver crates (`op-batch`,
//! `op-rails-a2a`, `op-rails-card`).
//!
//! ## What this crate ships
//!
//! - [`account`] — connected-account model (Standard / Express / Custom),
//!   capability set, requirements vector.
//! - [`kyb`] — Know-Your-Business profile, beneficial-owner model,
//!   FinCEN CDD (31 CFR § 1010.230) and EU AMLD5 (Directive (EU)
//!   2018/843, Art. 3(6)) compliance helpers.
//! - [`onboarding`] — stepwise onboarding state machine and the
//!   [`OnboardingProvider`](onboarding::OnboardingProvider) trait that
//!   isolates side-effects from policy. Ships a native in-process
//!   provider and documents the adapter shape for Stripe Connect /
//!   Adyen MarketPay migration.
//! - [`screening`] — wires the [`op-screening`] sanctions index, plus
//!   adds PEP and adverse-media coordination per FATF Recommendation 12.
//! - [`split`] — split-payments engine with sum / negativity /
//!   currency / duplicate-destination validation.
//! - [`transfer`] — internal-ledger transfer between connected accounts.
//! - [`payout`] — schedule arithmetic, rolling-reserve withholding,
//!   payout-instruction builder.
//! - [`liability`] — Platform / SubMerchant / Hybrid liability models
//!   driving tax attribution, dispute pass-through, and PCI scope
//!   inheritance.
//! - [`tax_reporting`] — IRS Form 1099-K and Form 1042-S generators,
//!   driven by the IRS 2024-instructions threshold table (IRS Notice
//!   2024-85 phased step-down from $20k/200tx to $600 by 2026).
//! - [`tos`] — terms-of-service acceptance ledger with idempotent
//!   replay semantics.
//!
//! ## What this crate does NOT do
//!
//! - **Rail submission.** Payout instructions hand off to `op-batch`
//!   (ACH / SEPA / BACS / wire) or `op-rails-a2a` (FedNow / RTP); we
//!   compute the instruction and stop.
//! - **Sanctions list ingestion.** That's `op-screening`'s job; we
//!   compose its `Screener` with PEP and adverse-media variants.
//! - **Document verification.** Identity-document review is delegated
//!   to a third-party provider (Persona, Onfido, Stripe Identity); we
//!   carry vault references and a verification state.
//! - **State-by-state tax-license tracking.** Out of scope; integrate
//!   a service like Avalara or TaxJar at the application layer.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::similar_names)]
#![allow(clippy::single_char_pattern)]
#![allow(clippy::unused_async)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::inconsistent_digit_grouping)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::float_cmp)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::redundant_clone)]
#![allow(clippy::single_match_else)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::ref_option_ref)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::option_option)]
#![allow(clippy::if_not_else)]
#![allow(clippy::unnested_or_patterns)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::manual_assert)]
#![allow(clippy::useless_let_if_seq)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::wildcard_imports)]

pub mod account;
pub mod error;
pub mod kyb;
pub mod liability;
pub mod onboarding;
pub mod payout;
pub mod screening;
pub mod split;
pub mod tax_reporting;
pub mod tos;
pub mod transfer;

pub use account::{
    AccountId, AccountSettings, AccountType, Capability, ConnectedAccount,
};
pub use error::{Error, Result};
pub use kyb::{
    Address, BeneficialOwner, BusinessProfile, BusinessStructure, CountryCode,
    EncryptedField, GovernmentId, Person, RequirementId, Requirements, TaxId, validate_cdd,
};
pub use liability::{DisputeResponder, LiabilityModel, RailKind};
pub use onboarding::{
    ExternalAccount, NativeProvider, OnboardingFlow, OnboardingProvider, OnboardingStatus,
    OnboardingStep, StepPayload, StepResult,
};
pub use payout::{PayoutInstruction, PayoutMode, PayoutSchedule};
pub use screening::{
    ConnectScreener, Decision, ScreeningResult, annotate_pep_flags, build_pep_index_from_names,
};
pub use split::{PaymentId, PaymentSplit, SplitLeg};
pub use tax_reporting::{
    AnnualTransaction, Form1042S, Form1099K, Form1099KThresholds, State, build_1042s,
    build_1099k,
};
pub use tos::{AcceptanceStore, RecordOutcome, TosAcceptance};
pub use transfer::{Ledger, LedgerEntry, TransferId, transfer};
