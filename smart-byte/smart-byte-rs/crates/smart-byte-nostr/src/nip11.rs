//! NIP-11 relay information document.
//!
//! A relay exposes its capabilities at the same URL it serves
//! WebSockets on, but with `Accept: application/nostr+json`. This
//! module models the document; HTTP fetching is left to callers.

use serde::{Deserialize, Serialize};

/// NIP-11 relay information document.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RelayInformation {
    /// Relay operator-chosen name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Contact pubkey hex of the operator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pubkey: Option<String>,
    /// Out-of-band contact string (email, alt URL, etc).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact: Option<String>,
    /// NIP numbers supported by this relay.
    #[serde(default)]
    pub supported_nips: Vec<u32>,
    /// Implementation software identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub software: Option<String>,
    /// Software version string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Server-declared limitations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limitation: Option<RelayLimitation>,
}

/// NIP-11 `limitation` object — relay-declared limits.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RelayLimitation {
    /// Max simultaneous subscriptions per client.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_subscriptions: Option<u32>,
    /// Max filters per REQ.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_filters: Option<u32>,
    /// Max event size in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_event_tags: Option<u32>,
    /// Min PoW required (NIP-13).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_pow_difficulty: Option<u32>,
    /// Whether AUTH (NIP-42) is required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_required: Option<bool>,
    /// Whether payment is required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payment_required: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nip11_roundtrip_json() {
        let info = RelayInformation {
            name: Some("test relay".into()),
            supported_nips: vec![1, 4, 11, 17, 44, 65],
            ..Default::default()
        };
        let s = serde_json::to_string(&info).expect("ser");
        let back: RelayInformation = serde_json::from_str(&s).expect("de");
        assert_eq!(back.name.as_deref(), Some("test relay"));
        assert_eq!(back.supported_nips, vec![1, 4, 11, 17, 44, 65]);
    }
}
