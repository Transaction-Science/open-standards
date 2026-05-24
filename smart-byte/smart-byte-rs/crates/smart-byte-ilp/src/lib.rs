//! Interledger Protocol (ILP) v4 adapter for Smart Byte.
//!
//! Interledger is the IETF + Interledger Foundation specification for
//! routing value packets across heterogeneous ledgers — banks, card
//! rails, stablecoins, central-bank ledgers, layer-2 chains — without
//! any of them having to peer with each other. A connector forwards an
//! ILP packet just like an IP router forwards an IP datagram: it knows
//! the next hop, not the destination ledger.
//!
//! This crate ingests the deployed ILPv4 protocol family into the
//! Smart Byte substrate so other crates can construct, parse, and route
//! Interledger packets without taking a runtime dependency on any
//! specific connector implementation.
//!
//! ## Layout
//!
//! * **Packet** ([`packet`]) — the three ILPv4 wire types `Prepare`,
//!   `Fulfill`, `Reject`, encoded with Octet Encoding Rules (OER).
//! * **OER** ([`oer`]) — the minimal slice of ASN.1 OER required by
//!   ILPv4: 8-bit length-prefix, length-determinant, variable-length
//!   octet strings, and the BE-uint encoding used for amounts.
//! * **Condition** ([`condition`]) — the SHA-256 hashlock that ties a
//!   `Prepare` to its eventual `Fulfill`.
//! * **Address** ([`address`]) — the dotted ILP address scheme
//!   (`g.us.bank.alice`) and its allocator-prefix validator.
//! * **STREAM** ([`stream`]) — frame types for the STREAM protocol that
//!   multiplexes money + data over a single ILP connection.
//! * **BTP** ([`btp`]) — the Bilateral Transfer Protocol envelope used
//!   for peer connections between connectors.
//! * **SPSP** ([`spsp`]) — the Simple Payment Setup Protocol used by
//!   senders to discover a receiver's destination address +
//!   shared secret.
//! * **Open Payments** ([`open_payments`]) — the W3C / Interledger
//!   Foundation HTTP API surface (`incoming-payments`,
//!   `outgoing-payments`, `quotes`).
//! * **Connector** ([`connector`]) — a longest-prefix route table and
//!   the per-hop forwarding decision (next-hop selection, currency
//!   conversion, rate-limit + balance check).
//!
//! ## What's intentionally scoped out
//!
//! * Settlement engines. We carry the prepare/fulfill loop and the
//!   bilateral balance counter; clearing real money against a real
//!   ledger lives in the connector operator's code.
//! * Wire I/O. The packet and BTP types serialize to bytes; the
//!   transport (WebSocket / HTTPS / QUIC) lives in the caller's runtime
//!   crate.
//! * Full ASN.1 OER. We hand-code the subset ILPv4 actually uses; a
//!   full OER codec is out of scope.
//! * Pre-shared-key derivation for STREAM. We carry the frame layer and
//!   leave AEAD framing to the caller; the encryption envelope is a
//!   straightforward AES-128-GCM over the frame bytes and does not need
//!   to live in this crate.

#![forbid(unsafe_code)]

pub mod address;
pub mod btp;
pub mod condition;
pub mod connector;
pub mod error;
pub mod oer;
pub mod open_payments;
pub mod packet;
pub mod spsp;
pub mod stream;

pub use address::{Address, AddressScheme};
pub use btp::{BtpMessage, BtpPacketType, BtpSubProtocol};
pub use condition::{Condition, Fulfillment};
pub use connector::{Connector, ForwardDecision, Route, RouteTable};
pub use error::{IlpError, Result};
pub use oer::{decode_var_octet_string, encode_var_octet_string};
pub use open_payments::{
    GrantAccess, IncomingPayment, OpenPaymentsClient, OutgoingPayment, PaymentPointer, Quote,
};
pub use packet::{Fulfill, IlpPacket, PacketType, Prepare, Reject, RejectCode};
pub use spsp::{SpspQuery, SpspResponse};
pub use stream::{Frame, FrameType, StreamPacket};
