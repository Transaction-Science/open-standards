//! AT URI ↔ Smart Byte AID bridge.
//!
//! An AT URI has the shape `at://<authority>/<collection>/<rkey>` where
//! `<authority>` is either a DID or a handle and `<collection>/<rkey>`
//! is the MST key.
//!
//! A Smart Byte **AID** (AT Identifier) in this crate is a typed
//! wrapper around the canonical string form
//! `<authority>:<collection>:<rkey>`. It is a stable, URL-safe label
//! suitable for embedding in [`smart_byte_core::Provenance`] fields or
//! in envelope cargo. AIDs are not SAIDs — they identify *records*, not
//! *content* — but the bridge gives us a deterministic mapping in both
//! directions.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use smart_byte_core::Said;

use crate::error::AtprotoError;

/// AT Protocol URI scheme.
pub const AT_SCHEME: &str = "at://";

/// Parsed AT URI.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AtUri {
    /// Authority (DID or handle).
    pub authority: String,
    /// Collection NSID, e.g. `"app.bsky.feed.post"`. `None` for
    /// repo-level URIs like `at://did:plc:abc`.
    pub collection: Option<String>,
    /// Record key, e.g. `"3kjp..."`. `None` if not present.
    pub rkey: Option<String>,
}

impl AtUri {
    /// Construct an AT URI from its parts.
    pub fn new(
        authority: impl Into<String>,
        collection: Option<String>,
        rkey: Option<String>,
    ) -> Self {
        Self {
            authority: authority.into(),
            collection,
            rkey,
        }
    }

    /// Repo-level URI (`at://<authority>`).
    pub fn repo(authority: impl Into<String>) -> Self {
        Self {
            authority: authority.into(),
            collection: None,
            rkey: None,
        }
    }

    /// Whether the URI carries a full `<collection>/<rkey>` pair.
    pub fn is_record(&self) -> bool {
        self.collection.is_some() && self.rkey.is_some()
    }

    /// Convert to a Smart Byte [`Aid`].
    pub fn to_aid(&self) -> Aid {
        Aid {
            authority: self.authority.clone(),
            collection: self.collection.clone(),
            rkey: self.rkey.clone(),
        }
    }

    /// Compute a deterministic [`Said`] commitment to this AT URI. Two
    /// AT URIs with the same canonical form will hash to the same SAID.
    pub fn to_said(&self) -> Said {
        Said::hash(self.to_string().as_bytes())
    }
}

impl fmt::Display for AtUri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(AT_SCHEME)?;
        f.write_str(&self.authority)?;
        if let Some(c) = &self.collection {
            write!(f, "/{c}")?;
            if let Some(r) = &self.rkey {
                write!(f, "/{r}")?;
            }
        }
        Ok(())
    }
}

impl FromStr for AtUri {
    type Err = AtprotoError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let rest = s.strip_prefix(AT_SCHEME).ok_or_else(|| {
            AtprotoError::InvalidAtUri(format!("missing at:// prefix: {s}"))
        })?;
        if rest.is_empty() {
            return Err(AtprotoError::InvalidAtUri("empty authority".into()));
        }
        let mut parts = rest.splitn(3, '/');
        let authority = parts
            .next()
            .ok_or_else(|| {
                AtprotoError::InvalidAtUri("missing authority".into())
            })?
            .to_string();
        if authority.is_empty() {
            return Err(AtprotoError::InvalidAtUri(
                "authority is empty".into(),
            ));
        }
        let collection = parts.next().map(String::from);
        let rkey = parts.next().map(String::from);
        // Reject trailing slashes (collection without rkey is fine but
        // an empty rkey is not).
        if let Some(c) = &collection {
            if c.is_empty() {
                return Err(AtprotoError::InvalidAtUri(
                    "empty collection".into(),
                ));
            }
        }
        if let Some(r) = &rkey {
            if r.is_empty() {
                return Err(AtprotoError::InvalidAtUri("empty rkey".into()));
            }
        }
        Ok(Self {
            authority,
            collection,
            rkey,
        })
    }
}

impl Serialize for AtUri {
    fn serialize<S: serde::Serializer>(
        &self,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for AtUri {
    fn deserialize<D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        raw.parse().map_err(serde::de::Error::custom)
    }
}

/// Smart Byte AT Identifier — typed bridge label.
///
/// Canonical text form: `<authority>|<collection>|<rkey>`, with empty
/// components when only an authority (or authority + collection) is
/// present. The `|` separator avoids ambiguity with the colons inside
/// DID authorities (`did:plc:…`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Aid {
    /// Authority (DID or handle).
    pub authority: String,
    /// Collection NSID, if any.
    pub collection: Option<String>,
    /// Record key, if any.
    pub rkey: Option<String>,
}

impl Aid {
    /// Construct an [`AtUri`] back from this AID.
    pub fn to_at_uri(&self) -> AtUri {
        AtUri {
            authority: self.authority.clone(),
            collection: self.collection.clone(),
            rkey: self.rkey.clone(),
        }
    }

    /// Canonical AID textual form (`authority|collection|rkey`). The
    /// `|` separator was chosen because DID authorities are full of
    /// colons (`did:plc:…`) but never contain a pipe.
    pub fn canonical(&self) -> String {
        format!(
            "{}|{}|{}",
            self.authority,
            self.collection.as_deref().unwrap_or(""),
            self.rkey.as_deref().unwrap_or(""),
        )
    }

    /// Parse an AID from its canonical text form.
    pub fn parse(s: &str) -> Result<Self, AtprotoError> {
        let mut parts = s.splitn(3, '|');
        let authority = parts
            .next()
            .ok_or_else(|| {
                AtprotoError::InvalidAtUri("aid missing authority".into())
            })?
            .to_string();
        if authority.is_empty() {
            return Err(AtprotoError::InvalidAtUri(
                "aid authority empty".into(),
            ));
        }
        let collection = parts.next().map(String::from).filter(|s| !s.is_empty());
        let rkey = parts.next().map(String::from).filter(|s| !s.is_empty());
        Ok(Self {
            authority,
            collection,
            rkey,
        })
    }
}

impl fmt::Display for Aid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.canonical())
    }
}

impl FromStr for Aid {
    type Err = AtprotoError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Aid::parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_uri_parse_record() {
        let s = "at://did:plc:abc/app.bsky.feed.post/3kjp";
        let u: AtUri = s.parse().unwrap();
        assert_eq!(u.authority, "did:plc:abc");
        assert_eq!(u.collection.as_deref(), Some("app.bsky.feed.post"));
        assert_eq!(u.rkey.as_deref(), Some("3kjp"));
        assert!(u.is_record());
        assert_eq!(u.to_string(), s);
    }

    #[test]
    fn at_uri_parse_repo_only() {
        let s = "at://did:plc:abc";
        let u: AtUri = s.parse().unwrap();
        assert_eq!(u.authority, "did:plc:abc");
        assert!(u.collection.is_none());
        assert!(!u.is_record());
    }

    #[test]
    fn at_uri_to_aid_roundtrip() {
        let s = "at://did:plc:abc/app.bsky.feed.post/3kjp";
        let u: AtUri = s.parse().unwrap();
        let aid = u.to_aid();
        let back = aid.to_at_uri();
        assert_eq!(back, u);
    }

    #[test]
    fn aid_canonical_form() {
        let aid = Aid {
            authority: "did:plc:abc".into(),
            collection: Some("app.bsky.feed.post".into()),
            rkey: Some("3kjp".into()),
        };
        assert_eq!(aid.canonical(), "did:plc:abc|app.bsky.feed.post|3kjp");
        let parsed: Aid = aid.canonical().parse().unwrap();
        assert_eq!(parsed, aid);
    }
}
