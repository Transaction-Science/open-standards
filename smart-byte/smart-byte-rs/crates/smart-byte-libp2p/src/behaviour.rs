//! Composed `NetworkBehaviour` for a Smart Byte libp2p node.
//!
//! Combines all the protocols Smart Byte exercises:
//!
//! * `kad` — Kademlia DHT for peer + content discovery.
//! * `gossipsub` — topic publish / subscribe.
//! * `mdns` — LAN discovery (toggleable).
//! * `autonat` — NAT detection.
//! * `identify` — peer-info exchange.
//! * `ping` — liveness.
//! * `relay` — Circuit Relay v2 server role (toggleable).
//! * `dcutr` — Direct Connection Upgrade through Relay (toggleable).
//! * `req_resp` — point-to-point envelope fetch.

use libp2p::kad::store::MemoryStore;
use libp2p::swarm::NetworkBehaviour;
use libp2p::swarm::behaviour::toggle::Toggle;
use libp2p::{StreamProtocol, autonat, dcutr, gossipsub, identify, kad, mdns, ping, relay};
use libp2p::request_response;

use crate::codec::EnvelopeCodec;

/// The Smart Byte libp2p protocol identifier for point-to-point
/// envelope fetch.
pub const ENVELOPE_PROTOCOL: &str = "/smart-byte/envelope/1.0.0";

/// The Smart Byte libp2p protocol identifier emitted via `identify`.
pub const IDENTIFY_PROTOCOL: &str = "/smart-byte/identify/1.0.0";

/// Composed network behaviour driving a Smart Byte libp2p node.
#[derive(NetworkBehaviour)]
pub struct SmartByteBehaviour {
    /// Kademlia DHT.
    pub kad: kad::Behaviour<MemoryStore>,
    /// Gossipsub publish/subscribe.
    pub gossipsub: gossipsub::Behaviour,
    /// mDNS LAN discovery (optional).
    pub mdns: Toggle<mdns::tokio::Behaviour>,
    /// AutoNAT for inferring our own reachability.
    pub autonat: autonat::Behaviour,
    /// Identify protocol — peer-info exchange.
    pub identify: identify::Behaviour,
    /// Liveness ping.
    pub ping: ping::Behaviour,
    /// Circuit Relay v2 server role (optional).
    pub relay: Toggle<relay::Behaviour>,
    /// DCUtR hole-punching (optional).
    pub dcutr: Toggle<dcutr::Behaviour>,
    /// Point-to-point envelope fetch.
    pub req_resp: request_response::Behaviour<EnvelopeCodec>,
}

/// Configuration toggles consumed by [`SmartByteBehaviour::build`].
pub struct BehaviourConfig {
    /// Local PeerId; needed for Kademlia and other protocols.
    pub local_peer_id: libp2p::PeerId,
    /// Local public key; needed for `identify`.
    pub local_public_key: libp2p::identity::PublicKey,
    /// Whether to enable mDNS.
    pub enable_mdns: bool,
    /// Whether to enable relay server role.
    pub enable_relay: bool,
    /// Whether to enable DCUtR hole-punching.
    pub enable_dcutr: bool,
    /// Gossipsub configuration.
    pub gossipsub_config: gossipsub::Config,
}

impl SmartByteBehaviour {
    /// Build the composed behaviour from a configuration.
    pub fn build(cfg: BehaviourConfig) -> crate::Result<Self> {
        // Kademlia.
        let store = MemoryStore::new(cfg.local_peer_id);
        let kad_cfg = kad::Config::default();
        let mut kad = kad::Behaviour::with_config(cfg.local_peer_id, store, kad_cfg);
        // Use server mode so peers in tests can respond to queries.
        kad.set_mode(Some(kad::Mode::Server));

        // Gossipsub.
        let gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(libp2p::identity::Keypair::generate_ed25519()),
            cfg.gossipsub_config.clone(),
        )
        .map_err(|e| crate::Error::Gossipsub(e.to_string()))?;

        // mDNS.
        let mdns = if cfg.enable_mdns {
            let m = mdns::tokio::Behaviour::new(mdns::Config::default(), cfg.local_peer_id)
                .map_err(|e| crate::Error::Build(format!("mdns: {e}")))?;
            Toggle::from(Some(m))
        } else {
            Toggle::from(None)
        };

        // AutoNAT.
        let autonat = autonat::Behaviour::new(cfg.local_peer_id, autonat::Config::default());

        // Identify.
        let identify = identify::Behaviour::new(identify::Config::new(
            IDENTIFY_PROTOCOL.to_string(),
            cfg.local_public_key.clone(),
        ));

        // Ping.
        let ping = ping::Behaviour::new(ping::Config::new());

        // Relay server role.
        let relay = if cfg.enable_relay {
            Toggle::from(Some(relay::Behaviour::new(
                cfg.local_peer_id,
                relay::Config::default(),
            )))
        } else {
            Toggle::from(None)
        };

        // DCUtR.
        let dcutr = if cfg.enable_dcutr {
            Toggle::from(Some(dcutr::Behaviour::new(cfg.local_peer_id)))
        } else {
            Toggle::from(None)
        };

        // Request-response for envelope fetch.
        let req_resp = request_response::Behaviour::with_codec(
            EnvelopeCodec,
            [(
                StreamProtocol::new(ENVELOPE_PROTOCOL),
                request_response::ProtocolSupport::Full,
            )],
            request_response::Config::default(),
        );

        Ok(Self {
            kad,
            gossipsub,
            mdns,
            autonat,
            identify,
            ping,
            relay,
            dcutr,
            req_resp,
        })
    }
}

/// Build a sensible default Gossipsub config: signed messages, IDs from
/// content (so identical envelopes deduplicate), short heartbeat.
pub fn default_gossipsub_config() -> gossipsub::Config {
    use std::time::Duration;
    gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(1))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .message_id_fn(|m: &gossipsub::Message| {
            let h = blake3_hash_message(m);
            gossipsub::MessageId::from(h.to_vec())
        })
        .build()
        .expect("default gossipsub config is statically valid")
}

fn blake3_hash_message(m: &gossipsub::Message) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(m.topic.as_str().as_bytes());
    h.update(&m.data);
    *h.finalize().as_bytes()
}
