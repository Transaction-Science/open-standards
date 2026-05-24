//! Two localhost nodes: A publishes on topic T; B receives.

use std::time::Duration;

use chrono::TimeZone;
use libp2p::futures::StreamExt;
use libp2p::swarm::SwarmEvent;
use smart_byte_core::{Cargo, Envelope, JouleCost, OwnershipChain, Provenance, Said};
use smart_byte_libp2p::{
    Libp2pNode, NodeConfig,
    behaviour::default_gossipsub_config,
    transport::{next_envelope_on, publish_envelope, subscribe_topic},
};

fn fixture_envelope(seed: &str) -> Envelope {
    let issuer = Said::hash(seed.as_bytes());
    let issued_at = chrono::Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap();
    let prov = Provenance::new(issuer, issued_at, vec![]);
    Envelope::new(
        prov,
        OwnershipChain::empty(),
        Cargo::Bytes(seed.as_bytes().to_vec()),
        JouleCost::measured(11),
    )
    .expect("envelope")
}

async fn spawn_node() -> Libp2pNode {
    let cfg = NodeConfig {
        listen_addrs: vec!["/ip4/127.0.0.1/tcp/0".parse().unwrap()],
        bootstrap_peers: vec![],
        identity: libp2p::identity::Keypair::generate_ed25519(),
        enable_mdns: false,
        enable_relay: false,
        enable_dcutr: false,
        gossipsub_config: default_gossipsub_config(),
    };
    Libp2pNode::new(cfg).await.expect("spawn node")
}

#[tokio::test]
async fn two_nodes_pubsub_envelope() {
    let mut node_a = spawn_node().await;
    let mut node_b = spawn_node().await;
    let a_addr = node_a
        .listen_addrs
        .first()
        .cloned()
        .expect("a listening");

    // B dials A so they form a connection.
    node_b
        .swarm
        .dial(a_addr)
        .expect("dial succeeds");

    // Drive both swarms briefly so identify/gossipsub mesh has a chance
    // to form before we publish.
    let topic = "smart-byte/test/envelopes";
    subscribe_topic(&mut node_a, topic).expect("subscribe a");
    subscribe_topic(&mut node_b, topic).expect("subscribe b");

    let warmup_until = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < warmup_until {
        tokio::select! {
            _ = node_a.swarm.select_next_some() => {}
            _ = node_b.swarm.select_next_some() => {}
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
        }
    }

    let env = fixture_envelope("two-nodes");
    publish_envelope(&mut node_a, topic, &env)
        .await
        .expect("publish");

    // Keep driving A while we wait for B to receive.
    let received = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            tokio::select! {
                ev = node_a.swarm.select_next_some() => {
                    let _: SwarmEvent<_> = ev;
                }
                got = next_envelope_on(&mut node_b, topic, Duration::from_secs(8)) => {
                    return got;
                }
            }
        }
    })
    .await
    .expect("not timed out")
    .expect("received envelope");
    let (received_env, from_peer) = received;
    assert_eq!(received_env.id, env.id);
    assert_eq!(from_peer, node_a.local_peer_id);
}
