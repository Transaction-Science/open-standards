//! `OrderedCollection` and `OrderedCollectionPage` (ActivityStreams §4.2).
//!
//! Inboxes, outboxes, followers, following, and Mastodon's `featured`
//! all surface as collections. We expose two shapes:
//!
//! * [`OrderedCollection`] — the top-level handle, carrying `totalItems`
//!   and `first` / `last` pointers.
//! * [`OrderedCollectionPage`] — a single page of items.

use crate::error::{ActivityPubError, Result};
use crate::vocabulary::AS2_CONTEXT;
use serde::{Deserialize, Serialize};

/// Top-level OrderedCollection document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OrderedCollection {
    /// JSON-LD context.
    #[serde(rename = "@context")]
    pub context: serde_json::Value,
    /// Stable IRI for this collection.
    pub id: String,
    /// AS2 `type` — always `"OrderedCollection"`.
    #[serde(rename = "type")]
    pub type_field: String,
    /// Total number of items in the collection across all pages.
    #[serde(rename = "totalItems")]
    pub total_items: u64,
    /// IRI of the first page, if paged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first: Option<String>,
    /// IRI of the last page, if paged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last: Option<String>,
    /// Inline items — present when the collection is small enough to
    /// avoid paging.
    #[serde(rename = "orderedItems", default, skip_serializing_if = "Vec::is_empty")]
    pub ordered_items: Vec<serde_json::Value>,
}

impl OrderedCollection {
    /// Construct an empty OrderedCollection.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            context: serde_json::Value::String(AS2_CONTEXT.into()),
            id: id.into(),
            type_field: "OrderedCollection".to_string(),
            total_items: 0,
            first: None,
            last: None,
            ordered_items: Vec::new(),
        }
    }

    /// Set the first-page pointer.
    pub fn with_first(mut self, iri: impl Into<String>) -> Self {
        self.first = Some(iri.into());
        self
    }

    /// Set the last-page pointer.
    pub fn with_last(mut self, iri: impl Into<String>) -> Self {
        self.last = Some(iri.into());
        self
    }

    /// Inline the items rather than paging (used when total fits).
    pub fn with_items(mut self, items: Vec<serde_json::Value>) -> Self {
        self.total_items = items.len() as u64;
        self.ordered_items = items;
        self
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self> {
        let c: OrderedCollection = serde_json::from_str(s)?;
        if c.type_field != "OrderedCollection" {
            return Err(ActivityPubError::Vocabulary(format!(
                "expected OrderedCollection, got {}",
                c.type_field
            )));
        }
        Ok(c)
    }
}

/// A single page of an OrderedCollection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OrderedCollectionPage {
    /// JSON-LD context.
    #[serde(rename = "@context")]
    pub context: serde_json::Value,
    /// Stable IRI for this page.
    pub id: String,
    /// AS2 `type` — always `"OrderedCollectionPage"`.
    #[serde(rename = "type")]
    pub type_field: String,
    /// IRI of the parent collection.
    #[serde(rename = "partOf")]
    pub part_of: String,
    /// Items on this page.
    #[serde(rename = "orderedItems", default, skip_serializing_if = "Vec::is_empty")]
    pub ordered_items: Vec<serde_json::Value>,
    /// Pointer to the next page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
    /// Pointer to the previous page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prev: Option<String>,
}

impl OrderedCollectionPage {
    /// Construct a fresh page.
    pub fn new(id: impl Into<String>, part_of: impl Into<String>) -> Self {
        Self {
            context: serde_json::Value::String(AS2_CONTEXT.into()),
            id: id.into(),
            type_field: "OrderedCollectionPage".to_string(),
            part_of: part_of.into(),
            ordered_items: Vec::new(),
            next: None,
            prev: None,
        }
    }

    /// Set the inline items.
    pub fn with_items(mut self, items: Vec<serde_json::Value>) -> Self {
        self.ordered_items = items;
        self
    }

    /// Set the `next` page pointer.
    pub fn with_next(mut self, iri: impl Into<String>) -> Self {
        self.next = Some(iri.into());
        self
    }

    /// Set the `prev` page pointer.
    pub fn with_prev(mut self, iri: impl Into<String>) -> Self {
        self.prev = Some(iri.into());
        self
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self> {
        let p: OrderedCollectionPage = serde_json::from_str(s)?;
        if p.type_field != "OrderedCollectionPage" {
            return Err(ActivityPubError::Vocabulary(format!(
                "expected OrderedCollectionPage, got {}",
                p.type_field
            )));
        }
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_roundtrip() -> Result<()> {
        let c = OrderedCollection::new("https://a.test/users/alice/outbox")
            .with_first("https://a.test/users/alice/outbox?page=1")
            .with_items(vec![serde_json::json!("https://a.test/activities/1")]);
        let json = c.to_json()?;
        let d = OrderedCollection::from_json(&json)?;
        assert_eq!(c, d);
        assert_eq!(d.total_items, 1);
        Ok(())
    }

    #[test]
    fn page_roundtrip() -> Result<()> {
        let p = OrderedCollectionPage::new(
            "https://a.test/users/alice/outbox?page=1",
            "https://a.test/users/alice/outbox",
        )
        .with_next("https://a.test/users/alice/outbox?page=2");
        let json = p.to_json()?;
        let q = OrderedCollectionPage::from_json(&json)?;
        assert_eq!(p, q);
        Ok(())
    }
}
