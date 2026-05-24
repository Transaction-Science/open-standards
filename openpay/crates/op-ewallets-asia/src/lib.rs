//! # `op-ewallets-asia` — APAC e-wallet rail adapters
//!
//! Acceptance adapters for the dominant Asia-Pacific e-wallets and
//! account-to-account fast rails:
//!
//! - **Alipay** (Cross-Border Open API v3) — direct-debit, QR-code,
//!   mini-program.
//! - **WeChat Pay** — JSAPI, Native (merchant-presented QR), H5,
//!   MicroPay (consumer-presented QR).
//! - **UPI 2.0** (NPCI, India) — collect, intent, and recurring-mandate
//!   flows + VPA resolution.
//! - **Paytm** — Standard Checkout + Paytm UPI intent.
//! - **GrabPay** (Pay-with-Grab) — SEA super-app rail.
//! - **GoPay** (Gojek) — Indonesia super-app rail.
//! - **Touch'n Go eWallet** — Malaysia.
//! - **PromptPay** (Thai BoT EMVCo MPM), **PayNow** (Singapore ABS),
//!   **DuitNow** (Malaysia PayNet), **QRIS** (Indonesia BI) —
//!   interoperable A2A QR rails over EMVCo MPM.
//! - **Refund + reconciliation** — symmetric semantics across all rails.
//!
//! ## Domain shape
//!
//! Every wallet boils down to a [`ChargeIntent`] in some
//! [`op_core::Currency`] for a [`op_core::Money`] amount, presented to
//! the consumer in a rail-specific way (QR, deeplink, JSAPI prepay
//! handle), then either authorized synchronously (push payment) or
//! confirmed asynchronously by a notify callback (pull payment).
//! The [`AsiaWallet`] trait normalizes that into one shape so a
//! merchant orchestrator can dispatch across rails without
//! pattern-matching at every site.
//!
//! ## Determinism + offline first
//!
//! This crate is **transport-free**. Codecs (EMVCo MPM TLV, JSON
//! request shapers) are pure functions over `&[u8]` / `&str`.
//! Signature verification is constant-time over operator-supplied
//! key material. The HTTP transport for Alipay / WeChat / Paytm is
//! injected via a `Transport` trait the operator implements (or
//! mocks in tests) — keeping the crate small, testable, and free
//! of `reqwest`/`tokio` baggage in environments (Cloudflare Workers,
//! POS firmware) that ship their own HTTP layer.
//!
//! ## Modules
//!
//! - [`wallet`]    — `AsiaWallet` trait + `WalletKind` + intent/result.
//! - [`alipay`]    — Alipay Cross-Border Open API v3 adapter.
//! - [`wechat`]    — WeChat Pay v3 adapter.
//! - [`upi`]       — UPI 2.0 (collect + intent + mandate + VPA).
//! - [`paytm`]     — Paytm Standard Checkout + Paytm UPI intent.
//! - [`grab`]      — GrabPay (Pay-with-Grab).
//! - [`gopay`]     — GoPay (Gojek).
//! - [`promptpay`] — PromptPay (Thai EMVCo MPM).
//! - [`paynow`]    — PayNow (Singapore EMVCo MPM).
//! - [`duitnow`]   — DuitNow (Malaysia EMVCo MPM).
//! - [`qris`]      — QRIS (Indonesia EMVCo MPM).
//! - [`tng`]       — Touch'n Go eWallet (Malaysia).
//! - [`refund`]    — symmetric refund + reconciliation semantics.
//! - [`error`]     — sealed error enum.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

pub mod alipay;
pub mod duitnow;
pub mod error;
pub mod gopay;
pub mod grab;
pub mod paynow;
pub mod paytm;
pub mod promptpay;
pub mod qris;
pub mod refund;
pub mod tng;
pub mod upi;
pub mod wallet;
pub mod wechat;

pub use error::{Error, Result};
pub use refund::{RefundIntent, RefundResult, ReconciliationLine, ReconciliationOutcome};
pub use wallet::{
    AsiaWallet, ChargeIntent, ChargeResult, ChargeStatus, PresentmentMode, WalletKind,
};
