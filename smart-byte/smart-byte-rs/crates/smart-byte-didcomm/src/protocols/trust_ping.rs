//! Aries RFC 0048: Trust-Ping Protocol v2 (DIDComm v2 port).
//!
//! Protocol URI: `https://didcomm.org/trust-ping/2.0`.
//! Messages: `ping`, `ping-response`.

use serde::{Deserialize, Serialize};

use crate::message::DidcommMessage;
use crate::protocol::{Protocol, ProtocolMessage};

/// Base URI for this protocol.
pub const PROTOCOL_URI: &str = "https://didcomm.org/trust-ping";
/// Protocol version.
pub const VERSION: &str = "2.0";

/// Singleton handle.
pub struct TrustPing;
impl Protocol for TrustPing {
    fn protocol_uri(&self) -> &str {
        PROTOCOL_URI
    }
    fn version(&self) -> &str {
        VERSION
    }
}

/// Body of a `ping` message.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PingBody {
    /// If `true`, the responder MUST send a `ping-response`.
    #[serde(default)]
    pub response_requested: bool,
    /// Free-form comment.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub comment: Option<String>,
}

/// Body of a `ping-response`. RFC 0048 v2 specifies an empty body —
/// the binding is through `thid` to the original ping.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PingResponseBody {
    /// Optional comment.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub comment: Option<String>,
}

/// Typed enum of trust-ping v2 message types.
#[derive(Debug, Clone)]
pub enum TrustPingKind {
    /// `ping`.
    Ping(PingBody),
    /// `ping-response`.
    PingResponse(PingResponseBody),
}

impl ProtocolMessage for TrustPingKind {
    fn from_message(msg: &DidcommMessage) -> Option<Self> {
        let base = format!("{PROTOCOL_URI}/{VERSION}");
        if msg.type_ == format!("{base}/ping") {
            let body: PingBody =
                serde_json::from_value(msg.body.clone()).unwrap_or_default();
            return Some(TrustPingKind::Ping(body));
        }
        if msg.type_ == format!("{base}/ping-response") {
            let body: PingResponseBody =
                serde_json::from_value(msg.body.clone()).unwrap_or_default();
            return Some(TrustPingKind::PingResponse(body));
        }
        None
    }

    fn to_message(&self) -> DidcommMessage {
        let (suffix, body) = match self {
            TrustPingKind::Ping(b) => (
                "ping",
                serde_json::to_value(b).expect("ping body serialisable"),
            ),
            TrustPingKind::PingResponse(b) => (
                "ping-response",
                serde_json::to_value(b)
                    .expect("ping-response body serialisable"),
            ),
        };
        DidcommMessage::new(format!("{PROTOCOL_URI}/{VERSION}/{suffix}"))
            .body(body)
    }
}

/// Build a ping-response for a given ping message, threading via `thid`.
pub fn respond_to(ping: &DidcommMessage) -> DidcommMessage {
    let mut resp = TrustPingKind::PingResponse(PingResponseBody::default())
        .to_message();
    resp.thid = Some(ping.thid.clone().unwrap_or_else(|| ping.id.clone()));
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_response_round_trip() {
        let p = TrustPingKind::Ping(PingBody {
            response_requested: true,
            comment: Some("hello".into()),
        })
        .to_message();
        let resp = respond_to(&p);
        match TrustPingKind::from_message(&resp).unwrap() {
            TrustPingKind::PingResponse(_) => {}
            _ => panic!("expected ping-response"),
        }
        assert_eq!(resp.thid.as_deref(), Some(p.id.as_str()));
    }
}
