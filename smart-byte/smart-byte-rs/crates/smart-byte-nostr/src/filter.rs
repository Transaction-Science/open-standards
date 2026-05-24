//! NIP-01 subscription filter.
//!
//! Filters are JSON objects with optional `ids`, `authors`, `kinds`,
//! `since`, `until`, `limit`, and tag-prefixed keys (e.g. `#e`, `#p`).

use crate::event::Event;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A REQ filter. Empty fields match anything.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Filter {
    /// Match by full event id (hex).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ids: Vec<String>,
    /// Match by author pubkey (hex).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub authors: Vec<String>,
    /// Match by kind.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kinds: Vec<u32>,
    /// Match created_at >= since.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub since: Option<i64>,
    /// Match created_at <= until.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub until: Option<i64>,
    /// Max events to return.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Per-letter tag filters: `#e -> [...]`, `#p -> [...]`, etc.
    #[serde(flatten)]
    pub tags: BTreeMap<String, Vec<String>>,
}

impl Filter {
    /// Empty filter (matches everything).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an id constraint.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.ids.push(id.into());
        self
    }

    /// Add an author constraint.
    pub fn with_author(mut self, pubkey: impl Into<String>) -> Self {
        self.authors.push(pubkey.into());
        self
    }

    /// Add a kind constraint.
    pub fn with_kind(mut self, kind: u32) -> Self {
        self.kinds.push(kind);
        self
    }

    /// Set the `since` constraint.
    pub fn with_since(mut self, since: i64) -> Self {
        self.since = Some(since);
        self
    }

    /// Set the `until` constraint.
    pub fn with_until(mut self, until: i64) -> Self {
        self.until = Some(until);
        self
    }

    /// Set the `limit`.
    pub fn with_limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Add a tag-letter constraint (single-letter tags only, per spec).
    pub fn with_tag(mut self, letter: char, value: impl Into<String>) -> Self {
        let key = format!("#{letter}");
        self.tags.entry(key).or_default().push(value.into());
        self
    }

    /// Check whether `event` matches this filter.
    pub fn matches(&self, event: &Event) -> bool {
        if !self.ids.is_empty() && !self.ids.iter().any(|i| i == &event.id) {
            return false;
        }
        if !self.authors.is_empty() && !self.authors.iter().any(|a| a == &event.pubkey) {
            return false;
        }
        if !self.kinds.is_empty() && !self.kinds.contains(&event.kind) {
            return false;
        }
        if let Some(s) = self.since
            && event.created_at < s
        {
            return false;
        }
        if let Some(u) = self.until
            && event.created_at > u
        {
            return false;
        }
        // Tag filters: every #X key must have at least one tag in the event
        // whose first element == X (without the leading '#') and whose
        // second element is in the filter's value list.
        for (key, values) in &self.tags {
            let Some(letter) = key.strip_prefix('#') else {
                continue;
            };
            let any = event.tags.iter().any(|t| {
                t.first().map(|s| s.as_str()) == Some(letter)
                    && t.get(1).is_some_and(|v| values.iter().any(|x| x == v))
            });
            if !any {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::UnsignedEvent;
    use crate::keys::NostrSecretKey;

    #[test]
    fn match_by_kind_and_tag() {
        let sk = NostrSecretKey::generate();
        let pk_hex = sk.public_key().to_hex();
        let event = UnsignedEvent::new(sk.public_key(), 1, "hi", 1_700_000_000)
            .with_tag(vec!["e".into(), "deadbeef".into()])
            .sign(&sk)
            .expect("sign");

        let f = Filter::new()
            .with_kind(1)
            .with_author(pk_hex)
            .with_tag('e', "deadbeef");
        assert!(f.matches(&event));

        let f_bad = Filter::new().with_kind(2);
        assert!(!f_bad.matches(&event));
    }
}
