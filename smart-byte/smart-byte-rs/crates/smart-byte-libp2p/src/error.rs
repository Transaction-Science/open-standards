//! Typed errors for the libp2p transport.

use libp2p::PeerId;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced by the libp2p transport.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Failure constructing a libp2p swarm or behaviour.
    #[error("swarm build failed: {0}")]
    Build(String),

    /// Failure dialing or listening on an address.
    #[error("transport: {0}")]
    Transport(String),

    /// CBOR encode/decode failed on an envelope.
    #[error("cbor: {0}")]
    Cbor(String),

    /// The underlying core envelope library returned an error.
    #[error("envelope: {0}")]
    Envelope(#[from] smart_byte_core::EnvelopeError),

    /// Gossipsub subscription or publish failed.
    #[error("gossipsub: {0}")]
    Gossipsub(String),

    /// Kademlia query failed or returned no provider.
    #[error("kademlia: {0}")]
    Kademlia(String),

    /// Request-response fetch failed against a known peer.
    #[error("fetch from {peer} failed: {reason}")]
    Fetch {
        /// Peer the fetch was directed at.
        peer: PeerId,
        /// Human-readable reason.
        reason: String,
    },

    /// The control channel for driving the swarm was closed.
    #[error("control channel closed")]
    ControlClosed,

    /// I/O failure (typically while loading or saving identity keys).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Identity decode/encode failure.
    #[error("identity: {0}")]
    Identity(String),

    /// Any other anyhow-wrapped error bubbling out of libp2p.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<serde_cbor::Error> for Error {
    fn from(e: serde_cbor::Error) -> Self {
        Error::Cbor(e.to_string())
    }
}
