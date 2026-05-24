//! Envelope publish / subscribe / fetch over libp2p.
//!
//! These functions mirror the surface shape of
//! `smart_byte_net::Node::publish_envelope` / `subscribe` so callers
//! can swap between the Iroh and libp2p transports without
//! restructuring call sites. They operate on a borrowed
//! [`Libp2pNode`] so the caller controls when the swarm is driven.
//!
//! ## Driving the swarm
//!
//! Because libp2p's `Swarm` is not `Clone`, this crate cannot spawn a
//! hidden background task without taking exclusive ownership. The
//! pattern is therefore:
//!
//! 1. Build a [`Libp2pNode`].
//! 2. Call [`publish_envelope`] / [`subscribe_topic`] /
//!    [`fetch_by_said`] to register intent.
//! 3. Drive the swarm yourself — either by calling
//!    [`crate::event::run`] from a spawned task (preferred for any
//!    concurrent fan-out) or by selecting on
//!    `node.swarm.select_next_some()` directly inside a single-task
//!    program.
//!
//! For the simple "subscribe and read the next message" pattern that
//! tests use, [`next_envelope_on`] polls the swarm until the next
//! gossipsub envelope on a given topic arrives.

use std::time::Duration;

use libp2p::futures::{Stream, StreamExt};
use libp2p::gossipsub::{IdentTopic, MessageId};
use libp2p::request_response::{self, OutboundRequestId};
use libp2p::swarm::SwarmEvent;
use libp2p::{PeerId, kad};
use smart_byte_core::{Envelope, Said};

use crate::behaviour::SmartByteBehaviourEvent;
use crate::codec::{EnvelopeRequest, EnvelopeResponse};
use crate::error::{Error, Result};
use crate::node::Libp2pNode;

/// Publish an envelope on a gossipsub topic. Returns the `MessageId`
/// assigned by gossipsub.
pub async fn publish_envelope(
    node: &mut Libp2pNode,
    topic: &str,
    env: &Envelope,
) -> Result<MessageId> {
    let topic_obj = IdentTopic::new(topic);
    // Subscribe if we have not already; gossipsub will only mesh peers
    // for topics we are subscribed to.
    node.swarm
        .behaviour_mut()
        .gossipsub
        .subscribe(&topic_obj)
        .map_err(|e| Error::Gossipsub(e.to_string()))?;

    let bytes = env.to_cbor().map_err(Error::Envelope)?;
    let id = node
        .swarm
        .behaviour_mut()
        .gossipsub
        .publish(topic_obj, bytes)
        .map_err(|e| Error::Gossipsub(e.to_string()))?;
    Ok(id)
}

/// Register a gossipsub subscription on the given topic. After
/// returning, the node will receive messages on `topic` whenever the
/// swarm is driven (see module-level docs).
pub fn subscribe_topic(node: &mut Libp2pNode, topic: &str) -> Result<bool> {
    let topic_obj = IdentTopic::new(topic);
    node.swarm
        .behaviour_mut()
        .gossipsub
        .subscribe(&topic_obj)
        .map_err(|e| Error::Gossipsub(e.to_string()))
}

/// Subscribe to `topic` and return a [`Stream`] of `(Envelope,
/// PeerId)` pairs.
///
/// The returned stream borrows `node` for its entire lifetime and
/// drives the swarm forward as messages arrive. Use this for simple
/// single-subscriber call sites; reach for [`crate::event::run`] when
/// fanning out to multiple consumers.
pub async fn subscribe<'a>(
    node: &'a mut Libp2pNode,
    topic: &str,
) -> Result<impl Stream<Item = (Envelope, PeerId)> + 'a> {
    subscribe_topic(node, topic)?;
    let topic_owned = topic.to_string();
    let stream = futures::stream::unfold(
        (node, topic_owned),
        |(node, topic_owned)| async move {
            loop {
                let ev = node.swarm.select_next_some().await;
                if let SwarmEvent::Behaviour(SmartByteBehaviourEvent::Gossipsub(
                    libp2p::gossipsub::Event::Message {
                        propagation_source,
                        message,
                        ..
                    },
                )) = ev
                    && message.topic.as_str() == topic_owned
                    && let Ok(env) = Envelope::from_cbor(&message.data)
                {
                    return Some((
                        (env, propagation_source),
                        (node, topic_owned),
                    ));
                }
            }
        },
    );
    Ok(Box::pin(stream))
}

/// Poll the swarm until the next gossipsub envelope arrives on
/// `topic`, or `timeout` elapses.
pub async fn next_envelope_on(
    node: &mut Libp2pNode,
    topic: &str,
    timeout: Duration,
) -> Result<(Envelope, PeerId)> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(Error::Gossipsub(format!(
                "timeout waiting for envelope on topic {topic}"
            )));
        }
        match tokio::time::timeout(remaining, node.swarm.select_next_some()).await {
            Ok(SwarmEvent::Behaviour(SmartByteBehaviourEvent::Gossipsub(
                libp2p::gossipsub::Event::Message {
                    propagation_source,
                    message,
                    ..
                },
            ))) if message.topic.as_str() == topic => {
                let env = Envelope::from_cbor(&message.data)
                    .map_err(Error::Envelope)?;
                return Ok((env, propagation_source));
            }
            Ok(_) => {}
            Err(_) => {
                return Err(Error::Gossipsub(format!(
                    "timeout waiting for envelope on topic {topic}"
                )));
            }
        }
    }
}

/// Fetch an envelope by its [`Said`].
///
/// * If `peer` is supplied, sends a direct request-response query to
///   that peer.
/// * If `peer` is `None`, performs a Kademlia `get_providers` lookup
///   first, then issues a request-response query to one of the
///   returned providers.
pub async fn fetch_by_said(
    node: &mut Libp2pNode,
    said: Said,
    peer: Option<PeerId>,
) -> Result<Envelope> {
    let target_peer = match peer {
        Some(p) => p,
        None => find_provider(node, said).await?,
    };
    direct_fetch(node, target_peer, said).await
}

/// Respond to a pending request-response query that was forwarded out
/// of the event loop. Tests use this to serve envelopes they hold.
pub fn respond_to_fetch(
    node: &mut Libp2pNode,
    channel: request_response::ResponseChannel<EnvelopeResponse>,
    envelope: Option<Envelope>,
) -> Result<()> {
    node.swarm
        .behaviour_mut()
        .req_resp
        .send_response(channel, EnvelopeResponse { envelope })
        .map_err(|_| Error::Gossipsub("response channel closed".to_string()))
}

/// Perform a request-response query against a specific peer for a SAID.
async fn direct_fetch(node: &mut Libp2pNode, peer: PeerId, said: Said) -> Result<Envelope> {
    let req = EnvelopeRequest { said };
    let request_id = node
        .swarm
        .behaviour_mut()
        .req_resp
        .send_request(&peer, req);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(Error::Fetch {
                peer,
                reason: "timed out waiting for response".to_string(),
            });
        }
        match tokio::time::timeout(remaining, node.swarm.select_next_some()).await {
            Ok(ev) => {
                if let Some(envelope) = handle_event_for_response(ev, request_id, peer)? {
                    return Ok(envelope);
                }
            }
            Err(_) => {
                return Err(Error::Fetch {
                    peer,
                    reason: "timed out waiting for response".to_string(),
                });
            }
        }
    }
}

fn handle_event_for_response(
    ev: SwarmEvent<SmartByteBehaviourEvent>,
    expected: OutboundRequestId,
    peer: PeerId,
) -> Result<Option<Envelope>> {
    match ev {
        SwarmEvent::Behaviour(SmartByteBehaviourEvent::ReqResp(
            request_response::Event::Message {
                message:
                    request_response::Message::Response {
                        request_id,
                        response: EnvelopeResponse { envelope },
                    },
                ..
            },
        )) if request_id == expected => match envelope {
            Some(e) => Ok(Some(e)),
            None => Err(Error::Fetch {
                peer,
                reason: "peer does not hold envelope".to_string(),
            }),
        },
        SwarmEvent::Behaviour(SmartByteBehaviourEvent::ReqResp(
            request_response::Event::OutboundFailure {
                request_id, error, ..
            },
        )) if request_id == expected => Err(Error::Fetch {
            peer,
            reason: error.to_string(),
        }),
        _ => Ok(None),
    }
}

/// Locate a provider for `said` by querying Kademlia.
async fn find_provider(node: &mut Libp2pNode, said: Said) -> Result<PeerId> {
    let key = kad::RecordKey::new(&said.as_bytes().to_vec());
    let qid = node
        .swarm
        .behaviour_mut()
        .kad
        .get_providers(key.clone());

    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(Error::Kademlia("provider lookup timed out".to_string()));
        }
        match tokio::time::timeout(remaining, node.swarm.select_next_some()).await {
            Ok(SwarmEvent::Behaviour(SmartByteBehaviourEvent::Kad(
                kad::Event::OutboundQueryProgressed {
                    id,
                    result: kad::QueryResult::GetProviders(Ok(progress)),
                    ..
                },
            ))) if id == qid => match progress {
                kad::GetProvidersOk::FoundProviders { providers, .. } => {
                    if let Some(p) = providers.into_iter().next() {
                        return Ok(p);
                    }
                }
                kad::GetProvidersOk::FinishedWithNoAdditionalRecord { .. } => {
                    return Err(Error::Kademlia("no providers found".to_string()));
                }
            },
            Ok(SwarmEvent::Behaviour(SmartByteBehaviourEvent::Kad(
                kad::Event::OutboundQueryProgressed {
                    id,
                    result: kad::QueryResult::GetProviders(Err(e)),
                    ..
                },
            ))) if id == qid => {
                return Err(Error::Kademlia(format!("{e:?}")));
            }
            Ok(_) => {}
            Err(_) => {
                return Err(Error::Kademlia("provider lookup timed out".to_string()));
            }
        }
    }
}
