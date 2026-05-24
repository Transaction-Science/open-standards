//! NIP-65 relay list metadata.
//!
//! Kind-10002 events carry a user's "outbox/inbox" relay preferences as
//! `["r", <url>, "read" | "write"]` tags. When the third element is
//! omitted the relay is used for both read and write.

use crate::error::NostrError;
use crate::event::{Event, UnsignedEvent};
use crate::keys::NostrSecretKey;

/// Kind 10002 — NIP-65 relay list metadata.
pub const KIND_RELAY_LIST: u32 = 10002;

/// Marker for which direction a relay is used in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelayUsage {
    /// Used to read events addressed to this user.
    Read,
    /// Used to publish events from this user.
    Write,
    /// Used for both reading and writing.
    ReadWrite,
}

impl RelayUsage {
    /// Parse the third tag element. Missing or empty means [`Self::ReadWrite`].
    pub fn from_marker(marker: Option<&str>) -> Self {
        match marker {
            Some("read") => Self::Read,
            Some("write") => Self::Write,
            _ => Self::ReadWrite,
        }
    }

    /// Render the third tag element ("read", "write", or omitted).
    pub fn marker(self) -> Option<&'static str> {
        match self {
            Self::Read => Some("read"),
            Self::Write => Some("write"),
            Self::ReadWrite => None,
        }
    }
}

/// A relay entry in a NIP-65 list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelayEntry {
    /// Relay WebSocket URL.
    pub url: String,
    /// Read/write usage marker.
    pub usage: RelayUsage,
}

/// Parsed NIP-65 relay list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RelayList {
    /// All `r` entries in the order they appear in the event.
    pub entries: Vec<RelayEntry>,
}

impl RelayList {
    /// Construct an empty list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a relay entry, builder-style.
    pub fn with(mut self, url: impl Into<String>, usage: RelayUsage) -> Self {
        self.entries.push(RelayEntry {
            url: url.into(),
            usage,
        });
        self
    }

    /// Build the NIP-65 tags from this list.
    pub fn to_tags(&self) -> Vec<Vec<String>> {
        self.entries
            .iter()
            .map(|e| {
                let mut t = vec!["r".to_string(), e.url.clone()];
                if let Some(m) = e.usage.marker() {
                    t.push(m.to_string());
                }
                t
            })
            .collect()
    }

    /// Parse a NIP-65 event's tags into a relay list.
    pub fn from_event(event: &Event) -> Result<Self, NostrError> {
        if event.kind != KIND_RELAY_LIST {
            return Err(NostrError::InvalidEvent(format!(
                "expected kind {KIND_RELAY_LIST}, got {}",
                event.kind
            )));
        }
        let mut entries = Vec::new();
        for tag in &event.tags {
            if tag.first().map(|s| s.as_str()) != Some("r") {
                continue;
            }
            let url = tag
                .get(1)
                .ok_or_else(|| NostrError::InvalidEvent("r tag missing url".into()))?
                .clone();
            let marker = tag.get(2).map(|s| s.as_str());
            entries.push(RelayEntry {
                url,
                usage: RelayUsage::from_marker(marker),
            });
        }
        Ok(Self { entries })
    }

    /// Build a signed kind-10002 event for this list.
    pub fn to_event(&self, sk: &NostrSecretKey, now: i64) -> Result<Event, NostrError> {
        let pk = sk.public_key();
        let mut e = UnsignedEvent::new(pk, KIND_RELAY_LIST, "", now);
        e.tags = self.to_tags();
        e.sign(sk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_roundtrip_through_event() {
        let sk = NostrSecretKey::generate();
        let list = RelayList::new()
            .with("wss://relay.one", RelayUsage::ReadWrite)
            .with("wss://relay.two", RelayUsage::Read)
            .with("wss://relay.three", RelayUsage::Write);
        let event = list.to_event(&sk, 1_700_000_000).expect("event");
        event.verify().expect("verify");
        let parsed = RelayList::from_event(&event).expect("parse");
        assert_eq!(parsed, list);
    }
}
