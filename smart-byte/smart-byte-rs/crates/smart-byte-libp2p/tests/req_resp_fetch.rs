//! Request-response: A asks B for an envelope by SAID; B serves it
//! directly (no DHT involved).

use std::time::Duration;

use chrono::TimeZone;
use libp2p::futures::StreamExt;
use libp2p::request_response::{self, Message};
use libp2p::swarm::SwarmEvent;
use smart_byte_core::{Cargo, Envelope, JouleCost, OwnershipChain, Provenance, Said};
use smart_byte_libp2p::{
    EnvelopeResponse, Libp2pNode, NodeConfig, SmartByteBehaviourEvent,
    behaviour::default_gossipsub_config,
};

fn fixture_envelope() -> Envelope {
    let issuer = Said::hash(b"req-resp");
    let issued_at = chrono::Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap();
    let prov = Provenance::new(issuer, issued_at, vec![]);
    Envelope::new(
        prov,
        OwnershipChain::empty(),
        Cargo::Bytes(b"reqresp-payload".to_vec()),
        JouleCost::measured(3),
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
async fn request_response_fetch() {
    let mut node_a = spawn_node().await;
    let mut node_b = spawn_node().await;

    let b_addr = node_b
        .listen_addrs
        .first()
        .cloned()
        .expect("b listening");

    // A dials B.
    node_a.swarm.dial(b_addr).expect("dial");

    // Warmup so the connection is fully established.
    let warmup = tokio::time::Instant::now() + Duration::from_secs(2);
    while tokio::time::Instant::now() < warmup {
        tokio::select! {
            _ = node_a.swarm.select_next_some() => {}
            _ = node_b.swarm.select_next_some() => {}
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
        }
    }

    let env = fixture_envelope();
    let env_for_b = env.clone();

    // Send the request from A. We then concurrently:
    //  - drive A's swarm and read the response.
    //  - drive B's swarm and serve the response when the inbound
    //    request arrives.
    let req = smart_byte_libp2p::EnvelopeRequest { said: env.id };
    let request_id = node_a
        .swarm
        .behaviour_mut()
        .req_resp
        .send_request(&node_b.local_peer_id, req);

    let result = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            tokio::select! {
                ev_b = node_b.swarm.select_next_some() => {
                    if let SwarmEvent::Behaviour(SmartByteBehaviourEvent::ReqResp(
                        request_response::Event::Message {
                            message: Message::Request { request: _, channel, .. },
                            ..
                        })) = ev_b
                    {
                        let _ = node_b
                            .swarm
                            .behaviour_mut()
                            .req_resp
                            .send_response(channel, EnvelopeResponse { envelope: Some(env_for_b.clone()) });
                    }
                }
                ev_a = node_a.swarm.select_next_some() => {
                    if let SwarmEvent::Behaviour(SmartByteBehaviourEvent::ReqResp(
                        request_response::Event::Message {
                            message: Message::Response { request_id: rid, response },
                            ..
                        })) = ev_a
                        && rid == request_id
                    {
                        return response;
                    }
                }
            }
        }
    })
    .await
    .expect("not timed out");

    let env_back = result.envelope.expect("envelope returned");
    assert_eq!(env_back.id, env.id);
    env_back.verify_said().expect("said valid");
}
