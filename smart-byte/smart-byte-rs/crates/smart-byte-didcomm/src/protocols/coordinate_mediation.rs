//! Aries RFC 0211: Coordinate-Mediation Protocol (DIDComm v2 port).
//!
//! Protocol URI: `https://didcomm.org/coordinate-mediation/3.0`.
//! Messages: `mediate-request`, `mediate-grant`, `mediate-deny`,
//! `keylist-update`, `keylist-update-response`, `keylist-query`,
//! `keylist`.

use serde::{Deserialize, Serialize};

use crate::message::DidcommMessage;
use crate::protocol::{Protocol, ProtocolMessage};

/// Base URI for this protocol.
pub const PROTOCOL_URI: &str = "https://didcomm.org/coordinate-mediation";
/// Protocol version.
pub const VERSION: &str = "3.0";

/// Singleton handle.
pub struct CoordinateMediation;
impl Protocol for CoordinateMediation {
    fn protocol_uri(&self) -> &str {
        PROTOCOL_URI
    }
    fn version(&self) -> &str {
        VERSION
    }
}

/// Mediation request — empty body.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MediateRequestBody {}

/// Mediation grant — endpoint + routing keys.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MediateGrantBody {
    /// Mediator's DIDComm endpoint URL.
    pub endpoint: String,
    /// Routing keys to use when forwarding through the mediator.
    pub routing_keys: Vec<String>,
}

/// Mediation deny — empty body.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MediateDenyBody {}

/// A single keylist update entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeylistUpdate {
    /// `add` or `remove`.
    pub action: String,
    /// Recipient key (typically a DID URL or did:key form).
    pub recipient_key: String,
}

/// `keylist-update` body.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeylistUpdateBody {
    /// Updates to apply.
    pub updates: Vec<KeylistUpdate>,
}

/// Typed enum of coordinate-mediation v3 message types.
#[derive(Debug, Clone)]
pub enum CoordinateMediationKind {
    /// `mediate-request`.
    MediateRequest(MediateRequestBody),
    /// `mediate-grant`.
    MediateGrant(MediateGrantBody),
    /// `mediate-deny`.
    MediateDeny(MediateDenyBody),
    /// `keylist-update`.
    KeylistUpdate(KeylistUpdateBody),
}

impl ProtocolMessage for CoordinateMediationKind {
    fn from_message(msg: &DidcommMessage) -> Option<Self> {
        let base = format!("{PROTOCOL_URI}/{VERSION}");
        let suffix = msg.type_.strip_prefix(&format!("{base}/"))?;
        match suffix {
            "mediate-request" => Some(Self::MediateRequest(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            "mediate-grant" => Some(Self::MediateGrant(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            "mediate-deny" => Some(Self::MediateDeny(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            "keylist-update" => Some(Self::KeylistUpdate(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            _ => None,
        }
    }

    fn to_message(&self) -> DidcommMessage {
        let (suffix, body) = match self {
            Self::MediateRequest(b) => (
                "mediate-request",
                serde_json::to_value(b).expect("body serialisable"),
            ),
            Self::MediateGrant(b) => (
                "mediate-grant",
                serde_json::to_value(b).expect("body serialisable"),
            ),
            Self::MediateDeny(b) => (
                "mediate-deny",
                serde_json::to_value(b).expect("body serialisable"),
            ),
            Self::KeylistUpdate(b) => (
                "keylist-update",
                serde_json::to_value(b).expect("body serialisable"),
            ),
        };
        DidcommMessage::new(format!("{PROTOCOL_URI}/{VERSION}/{suffix}"))
            .body(body)
    }
}
