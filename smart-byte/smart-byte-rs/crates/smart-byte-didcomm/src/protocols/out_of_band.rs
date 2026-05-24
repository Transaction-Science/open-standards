//! Aries RFC 0434: Out-of-Band Protocol v2 (DIDComm v2 port).
//!
//! Protocol URI: `https://didcomm.org/out-of-band/2.0`.
//! Message: `invitation`.

use serde::{Deserialize, Serialize};

use crate::message::DidcommMessage;
use crate::protocol::{Protocol, ProtocolMessage};

/// Base URI for this protocol.
pub const PROTOCOL_URI: &str = "https://didcomm.org/out-of-band";
/// Protocol version.
pub const VERSION: &str = "2.0";

/// Singleton handle.
pub struct OutOfBand;
impl Protocol for OutOfBand {
    fn protocol_uri(&self) -> &str {
        PROTOCOL_URI
    }
    fn version(&self) -> &str {
        VERSION
    }
}

/// Body of an OOB v2 `invitation`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct InvitationBody {
    /// Optional human-readable label for the inviter.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub label: Option<String>,
    /// Optional standard goal code, e.g. `aries.rel.build`.
    #[serde(rename = "goal_code", skip_serializing_if = "Option::is_none", default)]
    pub goal_code: Option<String>,
    /// Optional human-readable goal.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub goal: Option<String>,
    /// Accepted media types (DIDComm encrypted, signed, plaintext).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accept: Vec<String>,
}

/// Typed enum of OOB v2 message types.
#[derive(Debug, Clone)]
pub enum OutOfBandKind {
    /// `invitation`.
    Invitation(InvitationBody),
}

impl ProtocolMessage for OutOfBandKind {
    fn from_message(msg: &DidcommMessage) -> Option<Self> {
        if msg.type_ == format!("{PROTOCOL_URI}/{VERSION}/invitation") {
            let body: InvitationBody =
                serde_json::from_value(msg.body.clone()).unwrap_or_default();
            return Some(OutOfBandKind::Invitation(body));
        }
        None
    }

    fn to_message(&self) -> DidcommMessage {
        let (suffix, body) = match self {
            OutOfBandKind::Invitation(b) => (
                "invitation",
                serde_json::to_value(b).expect("invitation body serialisable"),
            ),
        };
        DidcommMessage::new(format!("{PROTOCOL_URI}/{VERSION}/{suffix}"))
            .body(body)
    }
}
