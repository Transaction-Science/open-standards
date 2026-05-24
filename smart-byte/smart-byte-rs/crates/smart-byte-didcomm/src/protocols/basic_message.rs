//! Aries RFC 0095: Basic Message Protocol v2 (DIDComm v2 port).
//!
//! Protocol URI: `https://didcomm.org/basicmessage/2.0`.
//! Single message: `message`.

use serde::{Deserialize, Serialize};

use crate::message::DidcommMessage;
use crate::protocol::{Protocol, ProtocolMessage};

/// Base URI for this protocol.
pub const PROTOCOL_URI: &str = "https://didcomm.org/basicmessage";
/// Protocol version.
pub const VERSION: &str = "2.0";

/// Singleton handle for the protocol id pair.
pub struct BasicMessage;
impl Protocol for BasicMessage {
    fn protocol_uri(&self) -> &str {
        PROTOCOL_URI
    }
    fn version(&self) -> &str {
        VERSION
    }
}

/// Body of a basic-message v2 message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BasicMessageBody {
    /// Message text.
    pub content: String,
    /// Optional ISO-8601 timestamp from sender.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sent_time: Option<String>,
}

/// Typed enum of basic-message v2 message types.
#[derive(Debug, Clone)]
pub enum BasicMessageKind {
    /// `message`.
    Message(BasicMessageBody),
}

impl ProtocolMessage for BasicMessageKind {
    fn from_message(msg: &DidcommMessage) -> Option<Self> {
        if msg.type_ == format!("{PROTOCOL_URI}/{VERSION}/message") {
            let body: BasicMessageBody =
                serde_json::from_value(msg.body.clone()).ok()?;
            return Some(BasicMessageKind::Message(body));
        }
        None
    }

    fn to_message(&self) -> DidcommMessage {
        let (suffix, body) = match self {
            BasicMessageKind::Message(b) => (
                "message",
                serde_json::to_value(b).expect("basicmessage body serialisable"),
            ),
        };
        DidcommMessage::new(format!("{PROTOCOL_URI}/{VERSION}/{suffix}"))
            .body(body)
    }
}
