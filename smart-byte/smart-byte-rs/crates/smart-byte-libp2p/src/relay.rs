//! Circuit Relay v2 helpers.
//!
//! Two roles:
//!
//! * **Server role.** A publicly-addressable node can opt in by setting
//!   [`crate::NodeConfig::enable_relay`]. Other nodes can then request
//!   a reservation from it and be reachable via a `/p2p-circuit/`
//!   suffix.
//! * **Client role.** A NATed node dials a relay's reservation
//!   endpoint with [`request_reservation`] and learns the resulting
//!   circuit address from a `relay::client::Event::ReservationReqAccepted`
//!   event on the swarm.
//!
//! libp2p 0.56 splits the relay server (`relay::Behaviour`) and the
//! relay client (`relay::client::Behaviour`) into two behaviours. The
//! Smart Byte node currently composes only the server. NATed nodes
//! that need the client side can extend [`crate::SmartByteBehaviour`]
//! or compose a separate node — that wiring lands in follow-up so the
//! crate stays focused on the substrate's server-side surface.

use libp2p::{Multiaddr, PeerId};

use crate::error::{Error, Result};
use crate::node::Libp2pNode;

/// Issue a `dial` to a relay so the swarm establishes a connection to
/// it. Once connected, a reservation will be negotiated automatically
/// if the local node has the relay client behaviour enabled.
pub fn dial_relay(node: &mut Libp2pNode, addr: Multiaddr) -> Result<()> {
    node.swarm
        .dial(addr)
        .map_err(|e| Error::Transport(e.to_string()))
}

/// Compose a `/p2p-circuit/p2p/<peer>` multiaddr — the address a third
/// party would dial to reach `target_peer` through the relay at
/// `relay_addr`.
pub fn circuit_address_for(relay_addr: &Multiaddr, target_peer: PeerId) -> Multiaddr {
    use libp2p::multiaddr::Protocol;
    let mut addr = relay_addr.clone();
    addr.push(Protocol::P2pCircuit);
    addr.push(Protocol::P2p(target_peer));
    addr
}

/// Returns `true` if the configured node has the relay server role
/// enabled (i.e. it can serve as a relay for other peers).
pub fn is_relay_server(node: &Libp2pNode) -> bool {
    node.swarm.behaviour().relay.is_enabled()
}
