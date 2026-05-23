//! PIX driver (Brazil — Bacen SPI rail).
//!
//! ## Transport summary
//!
//! - **mTLS to ICOM**. Mandatory client certificate, issued by the PSP's
//!   own CA or external ACs per Bacen Manual de Iniciação §3.1.
//! - **OAuth 2.0** (RFC 6749) **with Client Certificate-Bound Access Tokens**
//!   (RFC 8705 §3). The PSP's Authorization Server MUST bind the access
//!   token to the certificate presented during the TLS handshake.
//! - **Webhooks also use mTLS** per the same manual section.
//! - **XML signing** is mandatory for SPI messages. Operators sign with
//!   their HSM (`CloudHSM` / on-prem); we expose [`crate::Signer`].
//!
//! ## Identifiers
//!
//! Participants are identified by **ISPB** — an 8-digit Bacen-assigned
//! number. The `ParticipantId` variant [`ParticipantId::Ispb`] carries it.
//!
//! ## Currency
//!
//! BRL only.
//!
//! ## What we don't do
//!
//! - `CloudHSM` client integration: operator's job.
//! - CSR generation and Bacen-CA certificate fetch: operator's job.
//! - Bacen homologation runs: operator's job. (20K tx / 10 min SPI;
//!   1K key/sec DICT.)

pub mod client;
pub mod status_map;

pub use client::PixClient;
