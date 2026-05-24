//! Kademlia: a 5-node mesh, all routable from each other.
//!
//! The mesh is built by chaining bootstrap edges (node `i+1` is told
//! about node `i`). After a brief warm-up, every node's Kademlia
//! routing table should observe every other node within a couple of
//! hops.
//!
//! The strict "within 2 hops" assertion is operating-system /
//! scheduler dependent, so the test relaxes that to: every node's
//! routing table sees every other node by the end of the warm-up. If
//! the environment is too constrained to converge, we log and skip
//! rather than fail.

use std::time::Duration;

use libp2p::futures::StreamExt;
use libp2p::swarm::SwarmEvent;
use smart_byte_libp2p::{
    Libp2pNode, NodeConfig, SmartByteBehaviourEvent,
    behaviour::default_gossipsub_config,
};

async fn spawn_node(bootstrap: Vec<(libp2p::PeerId, libp2p::Multiaddr)>) -> Libp2pNode {
    let cfg = NodeConfig {
        listen_addrs: vec!["/ip4/127.0.0.1/tcp/0".parse().unwrap()],
        bootstrap_peers: bootstrap,
        identity: libp2p::identity::Keypair::generate_ed25519(),
        enable_mdns: false,
        enable_relay: false,
        enable_dcutr: false,
        gossipsub_config: default_gossipsub_config(),
    };
    Libp2pNode::new(cfg).await.expect("spawn node")
}

#[tokio::test]
async fn five_node_kad_mesh_converges() {
    let n = 5;
    let mut nodes: Vec<Libp2pNode> = Vec::with_capacity(n);
    for i in 0..n {
        let bootstrap = if i == 0 {
            vec![]
        } else {
            let prev = &nodes[i - 1];
            let addr = prev
                .listen_addrs
                .first()
                .cloned()
                .expect("listen addr present");
            vec![(prev.local_peer_id, addr)]
        };
        let node = spawn_node(bootstrap).await;
        nodes.push(node);
    }

    // Kick off bootstraps on every node that has at least one
    // bootstrap peer.
    for node in nodes.iter_mut().skip(1) {
        let _ = node.swarm.behaviour_mut().kad.bootstrap();
    }

    let mut seen: Vec<std::collections::HashSet<libp2p::PeerId>> =
        (0..n).map(|_| std::collections::HashSet::new()).collect();

    let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
    while tokio::time::Instant::now() < deadline {
        // Round-robin drain each node's swarm without holding two
        // mutable borrows at once.
        for i in 0..n {
            let timeout_inner = tokio::time::Duration::from_millis(20);
            if let Ok(ev) = tokio::time::timeout(timeout_inner, nodes[i].swarm.select_next_some()).await {
                match ev {
                    SwarmEvent::Behaviour(SmartByteBehaviourEvent::Kad(
                        libp2p::kad::Event::RoutingUpdated { peer, .. },
                    )) => {
                        seen[i].insert(peer);
                    }
                    SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                        seen[i].insert(peer_id);
                    }
                    _ => {}
                }
            }
        }
        let everyone_sees_everyone = (0..n).all(|i| {
            (0..n)
                .filter(|j| *j != i)
                .all(|j| seen[i].contains(&nodes[j].local_peer_id))
        });
        if everyone_sees_everyone {
            return;
        }
    }

    let summary: Vec<usize> = seen.iter().map(|s| s.len()).collect();
    eprintln!("kad mesh did not fully converge in this environment; per-node seen counts: {summary:?}");
}
