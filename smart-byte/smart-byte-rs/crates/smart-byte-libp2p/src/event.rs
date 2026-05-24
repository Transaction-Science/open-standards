//! Event loop translating libp2p `SwarmEvent`s into high-level
//! [`NodeEvent`]s on an `mpsc` channel.

use libp2p::futures::StreamExt;
use libp2p::swarm::SwarmEvent;
use libp2p::{Multiaddr, PeerId, gossipsub, kad, mdns, relay};
use smart_byte_core::Envelope;
use tokio::sync::mpsc;

use crate::behaviour::SmartByteBehaviourEvent;
use crate::error::{Error, Result};
use crate::node::Libp2pNode;

/// High-level events emitted by the Smart Byte libp2p event loop.
#[derive(Debug)]
pub enum NodeEvent {
    /// An envelope was received on a gossipsub topic. Boxed so the
    /// enum does not grow to the size of an [`Envelope`].
    EnvelopeReceived {
        /// The decoded envelope.
        envelope: Box<Envelope>,
        /// The PeerId that propagated it (not necessarily the issuer).
        from: PeerId,
        /// The topic the envelope arrived on.
        topic: String,
    },
    /// A new peer has been connected.
    PeerConnected(PeerId),
    /// A peer has been disconnected.
    PeerDisconnected(PeerId),
    /// A peer was discovered via Kademlia.
    DiscoveredViaKad(PeerId),
    /// A peer was discovered via mDNS.
    DiscoveredViaMdns(PeerId),
    /// We received a Circuit Relay v2 reservation we can advertise.
    RelayReservation(Multiaddr),
}

/// Drive the swarm forward, translating each `SwarmEvent` into a
/// [`NodeEvent`] on `tx`. Returns when `tx` is closed by the receiver.
pub async fn run(node: &mut Libp2pNode, tx: mpsc::Sender<NodeEvent>) -> Result<()> {
    loop {
        tokio::select! {
            ev = node.swarm.select_next_some() => {
                if let Some(out) = translate(ev)
                    && tx.send(out).await.is_err()
                {
                    return Err(Error::ControlClosed);
                }
            }
            _ = tx.closed() => {
                return Ok(());
            }
        }
    }
}

fn translate(ev: SwarmEvent<SmartByteBehaviourEvent>) -> Option<NodeEvent> {
    match ev {
        SwarmEvent::ConnectionEstablished { peer_id, .. } => Some(NodeEvent::PeerConnected(peer_id)),
        SwarmEvent::ConnectionClosed { peer_id, .. } => Some(NodeEvent::PeerDisconnected(peer_id)),
        SwarmEvent::Behaviour(SmartByteBehaviourEvent::Gossipsub(
            gossipsub::Event::Message {
                propagation_source,
                message,
                ..
            },
        )) => match Envelope::from_cbor(&message.data) {
            Ok(envelope) => Some(NodeEvent::EnvelopeReceived {
                envelope: Box::new(envelope),
                from: propagation_source,
                topic: message.topic.into_string(),
            }),
            Err(e) => {
                tracing::warn!(error = %e, "received malformed envelope on gossipsub");
                None
            }
        },
        SwarmEvent::Behaviour(SmartByteBehaviourEvent::Mdns(mdns::Event::Discovered(peers))) => {
            // Surface the first newly-discovered peer; additional peers
            // will be returned through subsequent mDNS events.
            peers
                .into_iter()
                .next()
                .map(|(peer, _addr)| NodeEvent::DiscoveredViaMdns(peer))
        }
        SwarmEvent::Behaviour(SmartByteBehaviourEvent::Kad(kad::Event::RoutingUpdated {
            peer,
            ..
        })) => Some(NodeEvent::DiscoveredViaKad(peer)),
        SwarmEvent::Behaviour(SmartByteBehaviourEvent::Relay(
            relay::Event::ReservationReqAccepted { src_peer_id: _, .. },
        )) => None,
        _ => None,
    }
}

/// Build an `Multiaddr` describing a relay reservation for this peer.
/// Helper used by tests and operator tooling that need to display the
/// reservation address to humans.
pub fn relay_circuit_addr(relay: &Multiaddr, peer: PeerId) -> Multiaddr {
    use libp2p::multiaddr::Protocol;
    let mut addr = relay.clone();
    addr.push(Protocol::P2pCircuit);
    addr.push(Protocol::P2p(peer));
    addr
}
