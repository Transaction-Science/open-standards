//! Aries RFC 0685: Message-Pickup Protocol v3 (DIDComm v2).
//!
//! Protocol URI: `https://didcomm.org/messagepickup/3.0`.
//! Messages: `status-request`, `status`, `delivery-request`, `delivery`,
//! `messages-received`, `live-delivery-change`.

use serde::{Deserialize, Serialize};

use crate::message::DidcommMessage;
use crate::protocol::{Protocol, ProtocolMessage};

/// Base URI for this protocol.
pub const PROTOCOL_URI: &str = "https://didcomm.org/messagepickup";
/// Protocol version.
pub const VERSION: &str = "3.0";

/// Singleton handle.
pub struct MessagePickup;
impl Protocol for MessagePickup {
    fn protocol_uri(&self) -> &str {
        PROTOCOL_URI
    }
    fn version(&self) -> &str {
        VERSION
    }
}

/// Body of `status-request`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct StatusRequestBody {
    /// Optional recipient_did filter.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub recipient_did: Option<String>,
}

/// Body of `status`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct StatusBody {
    /// Recipient did this status pertains to.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub recipient_did: Option<String>,
    /// Number of messages queued.
    pub message_count: u64,
    /// Longest delay in seconds among queued messages.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub longest_waited_seconds: Option<u64>,
    /// Newest received time (ISO-8601).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub newest_received_time: Option<String>,
    /// Oldest received time (ISO-8601).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub oldest_received_time: Option<String>,
    /// Aggregate bytes queued.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub total_bytes: Option<u64>,
    /// Whether live delivery is enabled.
    #[serde(default)]
    pub live_delivery: bool,
}

/// Body of `delivery-request`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeliveryRequestBody {
    /// Maximum number of messages to deliver.
    pub limit: u64,
    /// Optional recipient filter.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub recipient_did: Option<String>,
}

/// Body of `messages-received`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessagesReceivedBody {
    /// Message ids that have been received and can be deleted.
    pub message_id_list: Vec<String>,
}

/// Typed enum of message-pickup v3 message types.
#[derive(Debug, Clone)]
pub enum MessagePickupKind {
    /// `status-request`.
    StatusRequest(StatusRequestBody),
    /// `status`.
    Status(StatusBody),
    /// `delivery-request`.
    DeliveryRequest(DeliveryRequestBody),
    /// `messages-received`.
    MessagesReceived(MessagesReceivedBody),
}

impl ProtocolMessage for MessagePickupKind {
    fn from_message(msg: &DidcommMessage) -> Option<Self> {
        let base = format!("{PROTOCOL_URI}/{VERSION}");
        let suffix = msg.type_.strip_prefix(&format!("{base}/"))?;
        match suffix {
            "status-request" => Some(Self::StatusRequest(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            "status" => Some(Self::Status(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            "delivery-request" => Some(Self::DeliveryRequest(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            "messages-received" => Some(Self::MessagesReceived(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            _ => None,
        }
    }

    fn to_message(&self) -> DidcommMessage {
        let (suffix, body) = match self {
            Self::StatusRequest(b) => (
                "status-request",
                serde_json::to_value(b).expect("body serialisable"),
            ),
            Self::Status(b) => (
                "status",
                serde_json::to_value(b).expect("body serialisable"),
            ),
            Self::DeliveryRequest(b) => (
                "delivery-request",
                serde_json::to_value(b).expect("body serialisable"),
            ),
            Self::MessagesReceived(b) => (
                "messages-received",
                serde_json::to_value(b).expect("body serialisable"),
            ),
        };
        DidcommMessage::new(format!("{PROTOCOL_URI}/{VERSION}/{suffix}"))
            .body(body)
    }
}
