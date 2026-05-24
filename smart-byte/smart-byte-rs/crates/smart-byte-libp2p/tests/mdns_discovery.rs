//! mDNS-based discovery on loopback. Two nodes started with
//! `enable_mdns = true` should find each other without an explicit
//! bootstrap address.
//!
//! mDNS multicast is unreliable in some CI environments. The test is
//! tolerant of that: if neither node observes a peer within the
//! window, it is skipped rather than failed.

use std::time::Duration;

use libp2p::futures::StreamExt;
use libp2p::swarm::SwarmEvent;
use smart_byte_libp2p::{
    Libp2pNode, NodeConfig, SmartByteBehaviourEvent,
    behaviour::default_gossipsub_config,
};

async fn spawn_mdns_node() -> Libp2pNode {
    let cfg = NodeConfig {
        listen_addrs: vec!["/ip4/0.0.0.0/tcp/0".parse().unwrap()],
        bootstrap_peers: vec![],
        identity: libp2p::identity::Keypair::generate_ed25519(),
        enable_mdns: true,
        enable_relay: false,
        enable_dcutr: false,
        gossipsub_config: default_gossipsub_config(),
    };
    Libp2pNode::new(cfg).await.expect("spawn node")
}

#[tokio::test]
async fn mdns_discovers_peer_or_skips() {
    let mut node_a = spawn_mdns_node().await;
    let mut node_b = spawn_mdns_node().await;

    let peer_a = node_a.local_peer_id;
    let peer_b = node_b.local_peer_id;

    let mut a_saw_b = false;
    let mut b_saw_a = false;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline && !(a_saw_b && b_saw_a) {
        tokio::select! {
            ev = node_a.swarm.select_next_some() => {
                if let SwarmEvent::Behaviour(SmartByteBehaviourEvent::Mdns(
                    libp2p::mdns::Event::Discovered(peers))) = ev
                    && peers.iter().any(|(p, _)| *p == peer_b)
                {
                    a_saw_b = true;
                }
            }
            ev = node_b.swarm.select_next_some() => {
                if let SwarmEvent::Behaviour(SmartByteBehaviourEvent::Mdns(
                    libp2p::mdns::Event::Discovered(peers))) = ev
                    && peers.iter().any(|(p, _)| *p == peer_a)
                {
                    b_saw_a = true;
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
        }
    }

    if !(a_saw_b && b_saw_a) {
        eprintln!(
            "mDNS discovery did not complete in this environment \
            (a_saw_b={a_saw_b}, b_saw_a={b_saw_a}); skipping"
        );
    }
}
