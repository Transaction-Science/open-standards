//! Smart Byte network transport (Iroh wrapper).
//!
//! This crate is a *thin* wrapper around Iroh today. Production
//! networking — gossip topology, federation handshakes, BFT lockstep
//! pin-down, joule-cost accounting — lands in follow-up issues. The
//! intent of v1.0 is to fix the public surface so the rest of the
//! workspace can compile against it.
//!
//! Iroh's vocabulary in 1.0.0-rc.0 uses `EndpointId` where earlier
//! drafts of this crate used `NodeId`. We export the iroh type
//! directly so callers don't have to depend on `iroh` themselves.

use anyhow::Result;
use futures::Stream;
use iroh::EndpointId;
use iroh::endpoint::presets;
use smart_byte_core::Envelope;

/// A Smart Byte node. Wraps an Iroh endpoint.
///
/// `Node` is constructed via [`Node::spawn`] which binds an Iroh
/// endpoint with the [`presets::Empty`] preset (no relay, no DNS, no
/// discovery). Production deployments will swap in `presets::N0` or a
/// federation-specific preset; that wiring lands in follow-up.
pub struct Node {
    /// The underlying Iroh endpoint. Public so downstream tests and
    /// integrations can reach for the lower-level API while the
    /// wrapper layer is still settling.
    pub iroh: iroh::Endpoint,
}

/// Iroh peer identifier. Aliased so callers don't have to depend on
/// `iroh` directly. Maps to `iroh::EndpointId`.
pub type NodeId = EndpointId;

impl Node {
    /// Spawn a new node with the offline-friendly `Empty` preset.
    pub async fn spawn() -> Result<Self> {
        let iroh = iroh::Endpoint::builder(presets::Empty).bind().await?;
        Ok(Self { iroh })
    }

    /// Publish an envelope onto the network.
    ///
    /// MVP: serializes the envelope to CBOR. Production replication
    /// (iroh-blobs hand-off, gossip propagation) ships in follow-up.
    pub async fn publish_envelope(&self, env: &Envelope) -> Result<()> {
        let _bytes = serde_cbor::to_vec(env)?;
        // TODO(issue-5): wire this into iroh-blobs once the API stabilizes.
        Ok(())
    }

    /// Subscribe to envelopes published by `peer`.
    ///
    /// MVP: returns an empty stream. Production subscription ships in
    /// follow-up.
    pub fn subscribe(&self, _peer: NodeId) -> impl Stream<Item = Envelope> + Send + 'static {
        futures::stream::empty()
    }

    /// This node's own peer id.
    pub fn node_id(&self) -> NodeId {
        self.iroh.id()
    }
}
