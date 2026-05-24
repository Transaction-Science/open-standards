//! # `op-openbanking` — Open Banking / PSD2 / FDX adapters for OpenPay
//!
//! The "open data, open money" tier of OpenPay. Where `op-rails-card`
//! talks to schemes and `op-rails-a2a` talks to instant-payment rails,
//! `op-openbanking` talks to **regulated bank APIs** — the consent
//! economy specified by PSD2 (EU/EEA), the CMA9 / OBIE Read/Write
//! standard (UK), the Consumer Data Right (Australia), the SGFinDex
//! aggregator (Singapore), and the Financial Data Exchange (US).
//!
//! ## Standards covered
//!
//! | Standard                 | Region          | Scope         | Verified spec source                                                         |
//! |--------------------------|-----------------|---------------|------------------------------------------------------------------------------|
//! | UK Open Banking R/W v3.1 | United Kingdom  | AISP/PISP/CBPII/VRP | OBIE Read/Write Data API Specification v3.1.11                          |
//! | Berlin Group NextGenPSD2 | EEA             | XS2A          | NextGenPSD2 XS2A Framework Implementation Guidelines v1.3.13                 |
//! | STET PSD2                | France          | XS2A          | STET PSD2 API Documentation Part 1 v1.7                                      |
//! | Australia CDR (Banking)  | Australia       | Data sharing  | Consumer Data Standards v1.31 (ACCC / Data Standards Body)                   |
//! | SGFinDex                 | Singapore       | Aggregator    | MAS / SGFinDex API specification (consent-driven aggregation)                |
//! | FDX v6                   | United States   | Data sharing  | Financial Data Exchange API v6.x                                             |
//!
//! ## Security profile
//!
//! All six standards converge on **OAuth 2.0 + FAPI 1.0 Advanced**
//! (Financial-grade API, [`fapi`]) plus **mTLS** for transport-layer
//! client authentication (RFC 8705) and **JWS** (RFC 7515) for
//! request-object signing. We expose the *interface*: trait surfaces
//! for [`fapi::JwsSigner`], [`fapi::MtlsClientCert`], and
//! [`fapi::JwkRegistration`]. Operators bring real crypto (KMS,
//! HSM, eIDAS QSealC, OBSeal). The crate ships **no** soft-key
//! signing path — that would be a footgun on a regulated rail.
//!
//! ## Service surfaces
//!
//! - [`aisp`] — Account Information Service Provider. Pull balances,
//!   transactions, standing orders, direct debits, beneficiaries.
//!   PSD2 Article 67 ("payment account information").
//! - [`pisp`] — Payment Initiation Service Provider. Initiate a single
//!   immediate payment, future-dated, or standing-order. PSD2
//!   Article 66.
//! - [`piis`] — Payment Instrument Issuer Service Provider /
//!   Confirmation of Funds. CBPII / CoF. PSD2 Article 65.
//! - [`vrp`] — Variable Recurring Payments. Sweeping (own-account)
//!   and non-sweeping (commercial) VRP under the UK Open Banking VRP
//!   profile + OBIE consent model.
//!
//! ## Standard bindings
//!
//! - [`uk_ob`] — UK Open Banking R/W mapping.
//! - [`berlin_group`] — Berlin Group NextGenPSD2 XS2A mapping.
//! - [`stet`] — STET PSD2 mapping.
//! - [`cdr`] — Australia CDR (banking-tier) mapping.
//! - [`fdx`] — FDX v6 mapping.
//!
//! Each binding module declares its standard-specific consent shape,
//! endpoint roots, and version markers. The service traits ([`aisp`],
//! [`pisp`], [`piis`], [`vrp`]) are vendor-neutral; the bindings
//! describe how to ferry that vendor-neutral request to a specific
//! Account Servicing Payment Service Provider (ASPSP) over its
//! standard's wire format.
//!
//! ## What this crate does NOT do
//!
//! - **Crypto.** No private-key material is ever held in-process. The
//!   [`fapi::JwsSigner`] / [`fapi::MtlsClientCert`] traits are the
//!   exclusive integration points. Operators wire a KMS, HSM, eIDAS
//!   QSealC card, or an OBSeal directory entry behind them.
//! - **HTTP.** No `reqwest`. The crate constructs request payloads
//!   and parses response payloads; transport is an operator concern
//!   (FAPI deployments often pin specific TLS profiles, OCSP staples,
//!   or proxy through bank-side firewalls in ways no generic HTTP
//!   client can satisfy).
//! - **Consent UI.** Browser flows / app-to-app redirects are the
//!   operator's frontend job; we expose the consent state machine.
//! - **eIDAS QSealC / OBSeal issuance.** Operators get those from a
//!   qualified trust service provider.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::similar_names)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::single_match_else)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::elidable_lifetime_names)]

pub mod aisp;
pub mod berlin_group;
pub mod cdr;
pub mod error;
pub mod fapi;
pub mod fdx;
pub mod piis;
pub mod pisp;
pub mod stet;
pub mod uk_ob;
pub mod vrp;

pub use aisp::{
    Account, AccountInfoService, AccountSubtype, AccountType, Balance, BalanceType, ConsentId,
    Transaction, TransactionCredit, TransactionStatus,
};
pub use berlin_group::{BerlinConsent, BerlinPaymentProduct, BerlinService};
pub use cdr::{CdrArrangement, CdrBankingService, CdrScope};
pub use error::{Error, Result};
pub use fapi::{
    FapiProfile, JwkRegistration, JwkThumbprint, JwsSigner, MtlsClientCert, OAuth2Token,
    RequestObject,
};
pub use fdx::{FdxConsent, FdxResource, FdxService, FdxVersion};
pub use piis::{CofRequest, CofResponse, FundsConfirmationService};
pub use pisp::{
    PaymentInitiation, PaymentInitiationService, PaymentInitiationStatus, PaymentKind, PaymentRef,
};
pub use stet::{StetConsent, StetEndpointKind, StetService};
pub use uk_ob::{UkOpenBankingService, UkRwVersion, UkScope};
pub use vrp::{
    VrpConsent, VrpControlParameters, VrpExecution, VrpKind, VrpService, VrpSweep, VrpWindow,
};
