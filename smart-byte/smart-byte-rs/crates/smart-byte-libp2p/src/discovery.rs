//! Bootstrap and content-advertisement helpers built on Kademlia.

use libp2p::{Multiaddr, PeerId, kad};
use smart_byte_core::Said;

use crate::error::{Error, Result};
use crate::node::Libp2pNode;

/// Seed the Kademlia routing table from a list of known peers and
/// trigger a bootstrap. Caller is expected to drive the swarm so the
/// bootstrap can make progress.
pub fn bootstrap(node: &mut Libp2pNode, peers: &[(PeerId, Multiaddr)]) -> Result<()> {
    let kad = &mut node.swarm.behaviour_mut().kad;
    for (peer, addr) in peers {
        kad.add_address(peer, addr.clone());
    }
    kad.bootstrap()
        .map(|_| ())
        .map_err(|e| Error::Kademlia(format!("{e:?}")))
}

/// Advertise this node as a provider of the envelope identified by
/// `said` on the Kademlia content-providers ring.
pub fn announce_said(node: &mut Libp2pNode, said: Said) -> Result<()> {
    let key = kad::RecordKey::new(&said.as_bytes().to_vec());
    node.swarm
        .behaviour_mut()
        .kad
        .start_providing(key)
        .map(|_| ())
        .map_err(|e| Error::Kademlia(format!("{e:?}")))
}

/// Stop advertising `said`.
pub fn stop_announcing(node: &mut Libp2pNode, said: Said) {
    let key = kad::RecordKey::new(&said.as_bytes().to_vec());
    node.swarm.behaviour_mut().kad.stop_providing(&key);
}
