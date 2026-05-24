//! Construction of a Smart Byte libp2p node.
//!
//! [`Libp2pNode`] is a thin wrapper around a configured `Swarm`. The
//! event loop is intentionally not driven inside `new`; callers run
//! either [`crate::event::run`] (which forwards `NodeEvent`s into an
//! `mpsc` channel) or the transport helpers in [`crate::transport`]
//! that own the swarm for the duration of a single request.

use std::time::Duration;

use libp2p::futures::StreamExt;
use libp2p::swarm::SwarmEvent;
use libp2p::{Multiaddr, PeerId, Swarm, SwarmBuilder, gossipsub, identity, noise, tcp, yamux};

use crate::behaviour::{BehaviourConfig, SmartByteBehaviour, default_gossipsub_config};
use crate::error::{Error, Result};

/// Configuration consumed by [`Libp2pNode::new`].
pub struct NodeConfig {
    /// Multiaddrs the node should listen on. Empty disables listeners.
    pub listen_addrs: Vec<Multiaddr>,
    /// Bootstrap peers to seed the Kademlia routing table.
    pub bootstrap_peers: Vec<(PeerId, Multiaddr)>,
    /// Ed25519 keypair used to derive the local PeerId.
    pub identity: identity::Keypair,
    /// Enable mDNS LAN discovery.
    pub enable_mdns: bool,
    /// Enable the Circuit Relay v2 server role.
    pub enable_relay: bool,
    /// Enable DCUtR hole-punching.
    pub enable_dcutr: bool,
    /// Gossipsub configuration. Use [`default_gossipsub_config`] for a
    /// sensible default.
    pub gossipsub_config: gossipsub::Config,
}

impl NodeConfig {
    /// Convenience constructor: localhost-friendly defaults using a
    /// freshly-generated Ed25519 keypair.
    pub fn local_dev() -> Self {
        let identity = identity::Keypair::generate_ed25519();
        Self {
            listen_addrs: vec!["/ip4/127.0.0.1/tcp/0"
                .parse()
                .expect("static multiaddr is valid")],
            bootstrap_peers: vec![],
            identity,
            enable_mdns: false,
            enable_relay: false,
            enable_dcutr: false,
            gossipsub_config: default_gossipsub_config(),
        }
    }
}

/// A configured Smart Byte libp2p node.
pub struct Libp2pNode {
    /// The underlying libp2p swarm. Exposed so advanced callers can
    /// drive it directly while the wrapper layer is still settling.
    pub swarm: Swarm<SmartByteBehaviour>,
    /// This node's PeerId.
    pub local_peer_id: PeerId,
    /// Listen addresses observed after binding.
    pub listen_addrs: Vec<Multiaddr>,
}

impl Libp2pNode {
    /// Construct a new node from a [`NodeConfig`]. Binds the configured
    /// listeners and seeds Kademlia with the supplied bootstrap peers.
    pub async fn new(config: NodeConfig) -> Result<Self> {
        let local_peer_id = PeerId::from(config.identity.public());

        let public_key = config.identity.public();
        let behaviour_cfg = BehaviourConfig {
            local_peer_id,
            local_public_key: public_key,
            enable_mdns: config.enable_mdns,
            enable_relay: config.enable_relay,
            enable_dcutr: config.enable_dcutr,
            gossipsub_config: config.gossipsub_config,
        };

        let identity_for_swarm = config.identity.clone();

        let swarm_builder = SwarmBuilder::with_existing_identity(identity_for_swarm)
            .with_tokio()
            .with_tcp(
                tcp::Config::default().nodelay(true),
                noise::Config::new,
                yamux::Config::default,
            )
            .map_err(|e| Error::Build(format!("tcp transport: {e}")))?
            .with_quic()
            .with_behaviour(|_| {
                SmartByteBehaviour::build(behaviour_cfg)
                    .expect("behaviour build is infallible with validated config")
            })
            .map_err(|e| Error::Build(format!("behaviour: {e}")))?
            .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)));

        let mut swarm = swarm_builder.build();

        // Seed Kademlia.
        for (peer, addr) in &config.bootstrap_peers {
            swarm.behaviour_mut().kad.add_address(peer, addr.clone());
        }

        // Start listeners.
        for addr in &config.listen_addrs {
            swarm
                .listen_on(addr.clone())
                .map_err(|e| Error::Transport(e.to_string()))?;
        }

        // Drain the swarm briefly to collect listen addrs.
        let mut listen_addrs = Vec::new();
        let expected = config.listen_addrs.len();
        if expected > 0 {
            // Give listeners a brief window to bind. We poll the swarm
            // until we either see the expected number of NewListenAddr
            // events or a short timeout elapses.
            let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
            while listen_addrs.len() < expected
                && tokio::time::Instant::now() < deadline
            {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                match tokio::time::timeout(remaining, swarm.select_next_some()).await {
                    Ok(SwarmEvent::NewListenAddr { address, .. }) => {
                        listen_addrs.push(address);
                    }
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
        }

        Ok(Self {
            swarm,
            local_peer_id,
            listen_addrs,
        })
    }
}
