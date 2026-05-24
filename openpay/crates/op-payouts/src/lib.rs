//! # `op-payouts` — Multi-rail payout engine
//!
//! Sibling to the acceptance rails (`op-rails-card`, `op-rails-a2a`,
//! `op-rails-crypto`). Where the rails crates move funds **into** the
//! operator's balance, this crate moves funds **out** — to bank
//! accounts, debit cards, wallets and crypto addresses.
//!
//! ## Rails covered
//!
//! | Family            | Module          | Networks / Schemes                                         |
//! |-------------------|-----------------|------------------------------------------------------------|
//! | Card push         | [`visa_direct`] / [`mc_send`] | Visa Direct OCT, Mastercard Send             |
//! | US batch          | [`ach`]         | NACHA ACH credits (PPD/CCD)                                |
//! | Wire              | [`wire`]        | Fedwire, SWIFT MT103, ISO 20022 `pacs.008`                 |
//! | SEPA              | [`sepa`]        | SCT, SCT Instant                                           |
//! | UK                | [`uk_fps`]      | Faster Payments Service                                    |
//! | US instant        | [`fednow`] / [`rtp`] | FedNow, TCH RTP                                       |
//! | Wallet            | [`paypal`] / [`wise`] | PayPal Payouts API, Wise Platform API                |
//! | Crypto            | [`crypto`]      | USDC/USDT on ETH/Polygon/Base/Solana, BTC, Lightning       |
//!
//! ## Architecture
//!
//! Every driver implements the [`Payout`] trait. The orchestrator holds
//! `Box<dyn Payout>` and routes by [`PayoutMethod`]. KYC of the
//! beneficiary is delegated to `op-screening`; 1099/T5 triggers are
//! delegated to `op-tax`. Funding sources (prefunded balance vs
//! pull-based debit of the operator's settlement account) are modelled
//! in [`funding`].
//!
//! ## Determinism
//!
//! Per the OpenPay deterministic-contract doctrine, every driver in
//! this crate is offline-pure: it builds the rail-specific message and
//! returns it as bytes / a strongly-typed envelope alongside a planned
//! [`PayoutResult`]. Network submission is the operator's job. This
//! keeps payout logic testable without sandbox credentials and keeps
//! the inference loop out of the control plane.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]

pub mod ach;
pub mod crypto;
pub mod error;
pub mod fednow;
pub mod funding;
pub mod mc_send;
pub mod payout;
pub mod paypal;
pub mod rtp;
pub mod sepa;
pub mod uk_fps;
pub mod visa_direct;
pub mod wire;
pub mod wise;

pub use error::{Error, Result};
pub use funding::{FundingMode, FundingSource, PullDebitInstruction};
pub use payout::{
    Beneficiary, BeneficiaryAccount, Payout, PayoutMethod, PayoutRequest, PayoutResult,
    PayoutStatus,
};
