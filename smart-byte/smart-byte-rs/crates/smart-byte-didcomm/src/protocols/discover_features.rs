//! Aries RFC 0557: Discover-Features Protocol v2 (DIDComm v2 port).
//!
//! Protocol URI: `https://didcomm.org/discover-features/2.0`.
//! Messages: `queries`, `disclose`.

use serde::{Deserialize, Serialize};

use crate::message::DidcommMessage;
use crate::protocol::{Protocol, ProtocolMessage};

/// Base URI for this protocol.
pub const PROTOCOL_URI: &str = "https://didcomm.org/discover-features";
/// Protocol version.
pub const VERSION: &str = "2.0";

/// Singleton handle.
pub struct DiscoverFeatures;
impl Protocol for DiscoverFeatures {
    fn protocol_uri(&self) -> &str {
        PROTOCOL_URI
    }
    fn version(&self) -> &str {
        VERSION
    }
}

/// A single query in a `queries` message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FeatureQuery {
    /// Feature type, e.g. `protocol`, `goal-code`, `gov-framework`.
    #[serde(rename = "feature-type")]
    pub feature_type: String,
    /// Match pattern (substring or wildcard).
    pub match_: String,
}

/// Body of a `queries` message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct QueriesBody {
    /// One or more feature queries.
    pub queries: Vec<FeatureQuery>,
}

/// A single disclosure in a `disclose` message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Disclosure {
    /// Feature type.
    #[serde(rename = "feature-type")]
    pub feature_type: String,
    /// Concrete feature id, e.g. a protocol URI.
    pub id: String,
    /// Optional supported roles.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roles: Vec<String>,
}

/// Body of a `disclose` message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DiscloseBody {
    /// Set of disclosures matching the queries.
    pub disclosures: Vec<Disclosure>,
}

/// Typed enum of discover-features v2 message types.
#[derive(Debug, Clone)]
pub enum DiscoverFeaturesKind {
    /// `queries`.
    Queries(QueriesBody),
    /// `disclose`.
    Disclose(DiscloseBody),
}

impl ProtocolMessage for DiscoverFeaturesKind {
    fn from_message(msg: &DidcommMessage) -> Option<Self> {
        let base = format!("{PROTOCOL_URI}/{VERSION}");
        if msg.type_ == format!("{base}/queries") {
            let body: QueriesBody =
                serde_json::from_value(msg.body.clone()).ok()?;
            return Some(DiscoverFeaturesKind::Queries(body));
        }
        if msg.type_ == format!("{base}/disclose") {
            let body: DiscloseBody =
                serde_json::from_value(msg.body.clone()).ok()?;
            return Some(DiscoverFeaturesKind::Disclose(body));
        }
        None
    }

    fn to_message(&self) -> DidcommMessage {
        let (suffix, body) = match self {
            DiscoverFeaturesKind::Queries(b) => (
                "queries",
                serde_json::to_value(b).expect("queries body serialisable"),
            ),
            DiscoverFeaturesKind::Disclose(b) => (
                "disclose",
                serde_json::to_value(b).expect("disclose body serialisable"),
            ),
        };
        DidcommMessage::new(format!("{PROTOCOL_URI}/{VERSION}/{suffix}"))
            .body(body)
    }
}
