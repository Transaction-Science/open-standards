//! `FedNow` driver.
//!
//! ## Transport
//!
//! Per the `FedNow` Service Operating Procedures v3.2 (June 2025) and
//! the `FedLine` Connectivity Guide:
//!
//! - Primary message transport is **IBM MQ** over `FedLine` Direct or
//!   `FedLine` Advantage. Each participant has an MQ queue manager
//!   with a `FedNow` Service server certificate.
//! - API access (status query, profile management) is via REST over
//!   **`FedLine` VPN** with FRB-issued API certificates.
//! - Every participant has an `Authorized Connection Profile` (ACP)
//!   that defines the connectivity for one or many RTNs. A participant
//!   either owns an ACP directly or connects via a Service Provider's ACP.
//!
//! ## What this driver does
//!
//! We ship two transport modes:
//!
//! - [`FedNowMqClient`] — synchronous wrapper around an MQ-style
//!   request/response interface. Operators implement [`MqChannel`]
//!   to bridge to their actual IBM MQ deployment (the JMS client,
//!   amqp-mq, or a sidecar like `FedNow` Connect).
//! - [`FedNowApiClient`] — REST client for status queries that go
//!   over the `FedLine` VPN. Same `ureq` + rustls setup as Phase 4,
//!   but configured with a client certificate.
//!
//! ## Status codes
//!
//! `FedNow` uses the standard ISO 20022 `pacs.002` status codes:
//! `ACTC`, `ACSC`, `RJCT`, `PDNG`. Reason codes follow the
//! `ExternalStatusReason1Code` set on RJCT.
//!
//! We map them via [`crate::acquirer::A2aStatus`].
//!
//! ## NOT included
//!
//! - VPN tunnel setup. Operators provision `FedLine` Direct circuits.
//! - MQ queue manager deployment. We assume an operator-side bridge.
//! - FRB API certificate provisioning. Operators get those during
//!   onboarding.

pub mod client;
pub mod mq;
pub mod status_map;
pub mod xml;

pub use client::{FedNowApiClient, FedNowMqClient};
pub use mq::{MqChannel, MqMessage, MqResponse};
