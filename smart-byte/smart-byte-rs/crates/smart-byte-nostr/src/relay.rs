//! Relay protocol messages (client + server) and an in-memory matcher.
//!
//! This module models the wire-level JSON-array messages defined by
//! NIP-01 and friends:
//!
//! Client → relay: `["REQ", sub_id, filter1, ...]`, `["EVENT", event]`,
//! `["CLOSE", sub_id]`, `["AUTH", event]` (NIP-42).
//!
//! Relay → client: `["EVENT", sub_id, event]`, `["EOSE", sub_id]`,
//! `["NOTICE", message]`, `["OK", event_id, accepted, message]`,
//! `["AUTH", challenge]` (NIP-42), `["CLOSED", sub_id, reason]`.
//!
//! The actual WebSocket transport is intentionally **not** in this
//! crate — the message types and an in-memory routing helper give us a
//! transport-agnostic surface that integrators can wire to
//! `tokio-tungstenite`, `ws-stream-wasm`, or any other client.

use crate::error::NostrError;
use crate::event::Event;
use crate::filter::Filter;
use serde_json::Value;
use std::collections::HashMap;

/// A client-to-relay message.
#[derive(Debug, Clone)]
pub enum ClientMessage {
    /// Open or update a subscription.
    Req {
        /// Subscription id chosen by the client.
        sub_id: String,
        /// One or more filters (logical OR).
        filters: Vec<Filter>,
    },
    /// Publish an event.
    Event(Event),
    /// Close a subscription.
    Close {
        /// Subscription id to close.
        sub_id: String,
    },
    /// NIP-42 authentication: client sends a signed kind-22242 event.
    Auth(Event),
}

impl ClientMessage {
    /// Serialize to a relay-bound JSON array string.
    pub fn to_json(&self) -> Result<String, NostrError> {
        let value: Value = match self {
            Self::Req { sub_id, filters } => {
                let mut arr = vec![Value::String("REQ".into()), Value::String(sub_id.clone())];
                for f in filters {
                    arr.push(serde_json::to_value(f)?);
                }
                Value::Array(arr)
            }
            Self::Event(e) => Value::Array(vec![Value::String("EVENT".into()), serde_json::to_value(e)?]),
            Self::Close { sub_id } => Value::Array(vec![
                Value::String("CLOSE".into()),
                Value::String(sub_id.clone()),
            ]),
            Self::Auth(e) => Value::Array(vec![
                Value::String("AUTH".into()),
                serde_json::to_value(e)?,
            ]),
        };
        Ok(serde_json::to_string(&value)?)
    }
}

/// A relay-to-client message.
#[derive(Debug, Clone)]
pub enum RelayMessage {
    /// An event matching an open subscription.
    Event {
        /// Subscription id this event belongs to.
        sub_id: String,
        /// The event itself.
        event: Event,
    },
    /// End-of-stored-events: subscription has caught up, future events are live.
    Eose {
        /// Subscription id that just finished its stored backlog.
        sub_id: String,
    },
    /// Free-form notice from the relay.
    Notice(String),
    /// Per-event ack: was the EVENT accepted?
    Ok {
        /// Event id this OK refers to.
        event_id: String,
        /// Whether the relay accepted the event.
        accepted: bool,
        /// Human-readable message (prefixed `duplicate:`, `pow:`, etc.).
        message: String,
    },
    /// NIP-42 authentication challenge.
    Auth {
        /// Random challenge to be signed.
        challenge: String,
    },
    /// Subscription was closed by the relay.
    Closed {
        /// Subscription id that was closed.
        sub_id: String,
        /// Reason string.
        reason: String,
    },
}

impl RelayMessage {
    /// Parse a relay-emitted JSON array string.
    pub fn from_json(s: &str) -> Result<Self, NostrError> {
        let value: Value = serde_json::from_str(s)?;
        let arr = value
            .as_array()
            .ok_or_else(|| NostrError::Relay("not a json array".into()))?;
        if arr.is_empty() {
            return Err(NostrError::Relay("empty message".into()));
        }
        let tag = arr[0]
            .as_str()
            .ok_or_else(|| NostrError::Relay("first element not a string".into()))?;
        match tag {
            "EVENT" => {
                if arr.len() != 3 {
                    return Err(NostrError::Relay("EVENT: expected 3 elements".into()));
                }
                let sub_id = arr[1]
                    .as_str()
                    .ok_or_else(|| NostrError::Relay("EVENT sub_id not a string".into()))?
                    .to_string();
                let event: Event = serde_json::from_value(arr[2].clone())?;
                Ok(Self::Event { sub_id, event })
            }
            "EOSE" => {
                let sub_id = arr
                    .get(1)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| NostrError::Relay("EOSE sub_id not a string".into()))?
                    .to_string();
                Ok(Self::Eose { sub_id })
            }
            "NOTICE" => {
                let m = arr
                    .get(1)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| NostrError::Relay("NOTICE missing message".into()))?
                    .to_string();
                Ok(Self::Notice(m))
            }
            "OK" => {
                if arr.len() < 4 {
                    return Err(NostrError::Relay("OK: expected 4 elements".into()));
                }
                let event_id = arr[1]
                    .as_str()
                    .ok_or_else(|| NostrError::Relay("OK event_id not a string".into()))?
                    .to_string();
                let accepted = arr[2]
                    .as_bool()
                    .ok_or_else(|| NostrError::Relay("OK accepted not a bool".into()))?;
                let message = arr[3]
                    .as_str()
                    .ok_or_else(|| NostrError::Relay("OK message not a string".into()))?
                    .to_string();
                Ok(Self::Ok {
                    event_id,
                    accepted,
                    message,
                })
            }
            "AUTH" => {
                let c = arr
                    .get(1)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| NostrError::Relay("AUTH challenge missing".into()))?
                    .to_string();
                Ok(Self::Auth { challenge: c })
            }
            "CLOSED" => {
                if arr.len() < 3 {
                    return Err(NostrError::Relay("CLOSED: expected 3 elements".into()));
                }
                Ok(Self::Closed {
                    sub_id: arr[1]
                        .as_str()
                        .ok_or_else(|| NostrError::Relay("CLOSED sub_id not a string".into()))?
                        .to_string(),
                    reason: arr[2]
                        .as_str()
                        .ok_or_else(|| NostrError::Relay("CLOSED reason not a string".into()))?
                        .to_string(),
                })
            }
            other => Err(NostrError::Relay(format!("unknown tag {other}"))),
        }
    }
}

/// In-memory relay-side router: tracks subscriptions and routes incoming
/// EVENTs out to matching subscribers. Useful for tests and for embedding
/// a minimal relay inside an application.
#[derive(Debug, Default)]
pub struct RelayRouter {
    subs: HashMap<String, Vec<Filter>>,
}

impl RelayRouter {
    /// New empty router.
    pub fn new() -> Self {
        Self::default()
    }

    /// Open or replace a subscription.
    pub fn open(&mut self, sub_id: impl Into<String>, filters: Vec<Filter>) {
        self.subs.insert(sub_id.into(), filters);
    }

    /// Close a subscription.
    pub fn close(&mut self, sub_id: &str) {
        self.subs.remove(sub_id);
    }

    /// Return the subscription ids that match `event`.
    pub fn matching_subs(&self, event: &Event) -> Vec<String> {
        self.subs
            .iter()
            .filter(|(_, fs)| fs.iter().any(|f| f.matches(event)))
            .map(|(k, _)| k.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::UnsignedEvent;
    use crate::keys::NostrSecretKey;

    #[test]
    fn client_message_serialization() {
        let m = ClientMessage::Close {
            sub_id: "sub1".into(),
        };
        let s = m.to_json().expect("ser");
        assert_eq!(s, "[\"CLOSE\",\"sub1\"]");
    }

    #[test]
    fn relay_message_parse_event() {
        let sk = NostrSecretKey::generate();
        let ev = UnsignedEvent::new(sk.public_key(), 1, "hi", 1_700_000_000)
            .sign(&sk)
            .expect("sign");
        let payload = serde_json::json!(["EVENT", "sub1", &ev]).to_string();
        match RelayMessage::from_json(&payload).expect("parse") {
            RelayMessage::Event { sub_id, event } => {
                assert_eq!(sub_id, "sub1");
                assert_eq!(event.id, ev.id);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn router_routes_by_kind() {
        let sk = NostrSecretKey::generate();
        let ev = UnsignedEvent::new(sk.public_key(), 1, "hi", 1_700_000_000)
            .sign(&sk)
            .expect("sign");
        let mut r = RelayRouter::new();
        r.open("a", vec![Filter::new().with_kind(1)]);
        r.open("b", vec![Filter::new().with_kind(2)]);
        let subs = r.matching_subs(&ev);
        assert_eq!(subs, vec!["a".to_string()]);
    }
}
