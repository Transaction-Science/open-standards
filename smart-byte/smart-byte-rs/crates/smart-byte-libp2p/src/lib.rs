//! Smart Byte networking: libp2p transport.
//!
//! This crate is the second transport for Smart Byte alongside
//! [`smart-byte-net`] (which wraps Iroh). It exposes the same conceptual
//! shape — publish an [`Envelope`](smart_byte_core::Envelope) on a
//! topic, subscribe to a topic, fetch an envelope by its
//! [`Said`](smart_byte_core::Said) — but does so over libp2p's broader
//! protocol surface:
//!
//! * **Transports:** TCP + QUIC.
//! * **Security:** Noise (XX handshake).
//! * **Stream multiplexer:** Yamux.
//! * **Peer / content discovery:** Kademlia DHT, mDNS (LAN).
//! * **Publish / subscribe:** Gossipsub.
//! * **NAT traversal:** AutoNAT + Circuit Relay v2 + DCUtR hole-punch.
//! * **Liveness + identification:** identify + ping.
//! * **Direct fetch:** request-response with a CBOR envelope codec.
//!
//! Operators choose between the two transports based on deployment
//! shape. Iroh is the lower-friction QUIC + hole-punching path; libp2p
//! is the broader-interop path that connects Smart Byte to the existing
//! P2P ecosystem (IPFS, Filecoin, Ethereum, Substrate, …).

#![forbid(unsafe_code)]

pub mod behaviour;
pub mod codec;
pub mod discovery;
pub mod error;
pub mod event;
pub mod identity;
pub mod node;
pub mod relay;
pub mod transport;

pub use behaviour::{SmartByteBehaviour, SmartByteBehaviourEvent};
pub use codec::{EnvelopeCodec, EnvelopeRequest, EnvelopeResponse};
pub use error::{Error, Result};
pub use event::{NodeEvent, run};
pub use identity::{load_or_create_keypair, load_keypair, save_keypair};
pub use node::{Libp2pNode, NodeConfig};

// Re-export the libp2p surface callers most commonly need so they do
// not have to depend on `libp2p` directly.
pub use libp2p::{Multiaddr, PeerId, identity::Keypair};
pub use libp2p::gossipsub::{Config as GossipsubConfig, MessageId};
