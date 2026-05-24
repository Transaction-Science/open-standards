//! DIDComm v2 messaging for Smart Byte.
//!
//! This crate ingests the deployed footprint of agent-to-agent messaging
//! that pairs with W3C DIDs: the DIF DIDComm Messaging v2.1 specification
//! (`https://identity.foundation/didcomm-messaging/spec/v2.1/`) and the
//! canonical Aries RFC application protocols layered on top of it.
//!
//! DIDComm v2 is the standard messaging protocol that pairs with DIDs to
//! deliver credentials, present proofs, and exchange application messages
//! between two parties without intermediaries. The largest deployed
//! ecosystem is Hyperledger Aries (Aries Framework JavaScript, Aries
//! Framework .NET, Aries Cloud Agent Python).
//!
//! ## Wire format
//!
//! [`pack`] implements the three DIDComm v2 packaging modes:
//!
//! * **Plaintext** — JSON only. Testing / debugging.
//! * **Signed** — JWS (RFC 7515) over the plaintext payload. EdDSA / ES256.
//! * **Encrypted** — JWE (RFC 7516) with content encryption (A256GCM,
//!   XC20P, A256CBC-HS512) and key agreement (ECDH-1PU authenticated or
//!   ECDH-ES anonymous) using X25519 or P-256.
//!
//! ## Application protocols
//!
//! [`protocols`] implements the canonical Aries application protocols:
//!
//! * [`protocols::issue_credential`] — Aries RFC 0453 issue-credential v3.
//! * [`protocols::present_proof`] — Aries RFC 0454 present-proof v3.
//! * [`protocols::basic_message`] — Aries RFC 0095 basic-message v2.
//! * [`protocols::trust_ping`] — Aries RFC 0048 trust-ping v2.
//! * [`protocols::discover_features`] — Aries RFC 0557 discover-features v2.
//! * [`protocols::out_of_band`] — Aries RFC 0434 out-of-band v2 invitations.
//! * [`protocols::coordinate_mediation`] — Aries RFC 0211 mediation
//!   coordination.
//! * [`protocols::messagepickup`] — Aries RFC 0685 message pickup v3.
//!
//! ## Mediators and service endpoints
//!
//! [`mediator`] implements the mediator role (an agent that holds messages
//! for a mobile/edge agent). [`service`] parses the
//! `DIDCommMessaging`-typed `service` block from a DID document.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod error;
pub mod key_agreement;
pub mod mediator;
pub mod message;
pub mod pack;
pub mod protocol;
pub mod protocols;
pub mod service;
pub mod state;

pub use error::DidcommError;
pub use key_agreement::{ContentEncryption, KeyAgreementAlgorithm, KeyPair};
pub use mediator::{InMemoryMediator, Mediator, MediatorStorage};
pub use message::{Attachment, AttachmentData, DidcommMessage};
pub use pack::{
    UnpackedMessage, pack_encrypted, pack_plaintext, pack_signed, unpack,
};
pub use protocol::{Protocol, ProtocolMessage};
pub use service::{DidcommServiceEndpoint, parse_didcomm_service};
pub use state::{Connection, InMemoryConnection};
