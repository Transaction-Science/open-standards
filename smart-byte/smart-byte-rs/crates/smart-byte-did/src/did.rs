//! DID syntax: `did:<method>:<method-specific-id>` and DID URLs.
//!
//! DID Core 1.0, § 3.1 (DID Syntax) defines the ABNF; we implement a
//! pragmatic parser that handles the common shape used by all major
//! methods (`did:key`, `did:web`, `did:peer`, `did:jwk`, `did:ion`).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::DidError;

/// The DID method portion of a DID. Known methods are enumerated; any
/// other syntactically-valid method falls into [`DidMethod::Custom`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DidMethod {
    /// `did:key` — self-certifying single-key DID.
    Key,
    /// `did:web` — DID document fetched from an HTTPS origin.
    Web,
    /// `did:peer` — inline-encoded DID document (Aries / DIDComm).
    Peer,
    /// `did:jwk` — base64url-encoded JWK as the method-specific id.
    Jwk,
    /// `did:ion` — Sidetree-based DID method (Bitcoin anchor).
    Ion,
    /// Any other registered DID method name.
    Custom(String),
}

impl DidMethod {
    /// Lower-case method name as used in the DID textual form.
    pub fn as_str(&self) -> &str {
        match self {
            DidMethod::Key => "key",
            DidMethod::Web => "web",
            DidMethod::Peer => "peer",
            DidMethod::Jwk => "jwk",
            DidMethod::Ion => "ion",
            DidMethod::Custom(s) => s,
        }
    }

    fn from_name(name: &str) -> Self {
        match name {
            "key" => DidMethod::Key,
            "web" => DidMethod::Web,
            "peer" => DidMethod::Peer,
            "jwk" => DidMethod::Jwk,
            "ion" => DidMethod::Ion,
            other => DidMethod::Custom(other.to_string()),
        }
    }
}

impl fmt::Display for DidMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A parsed DID: a method plus the method-specific identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Did {
    /// The DID method (e.g. [`DidMethod::Key`]).
    pub method: DidMethod,
    /// The method-specific identifier (the substring after
    /// `did:<method>:`). Stored verbatim — no normalisation is applied.
    pub method_specific_id: String,
}

impl Did {
    /// Construct a [`Did`] from its parts. Does not validate the
    /// method-specific id beyond non-emptiness.
    pub fn new(
        method: DidMethod,
        method_specific_id: impl Into<String>,
    ) -> Result<Self, DidError> {
        let id = method_specific_id.into();
        if id.is_empty() {
            return Err(DidError::InvalidIdentifier(
                "method-specific id is empty".into(),
            ));
        }
        Ok(Did {
            method,
            method_specific_id: id,
        })
    }
}

impl fmt::Display for Did {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "did:{}:{}", self.method, self.method_specific_id)
    }
}

impl FromStr for Did {
    type Err = DidError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let rest = s.strip_prefix("did:").ok_or_else(|| {
            DidError::InvalidIdentifier(format!("missing did: prefix in {s}"))
        })?;
        let (method, msid) = rest.split_once(':').ok_or_else(|| {
            DidError::InvalidIdentifier(format!(
                "missing method-specific id in {s}"
            ))
        })?;
        if method.is_empty() {
            return Err(DidError::InvalidIdentifier(
                "empty method name".into(),
            ));
        }
        if !method
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        {
            return Err(DidError::InvalidIdentifier(format!(
                "method name must be lower-case ASCII alphanumeric: {method}"
            )));
        }
        Did::new(DidMethod::from_name(method), msid)
    }
}

impl Serialize for Did {
    fn serialize<S: serde::Serializer>(
        &self,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Did {
    fn deserialize<D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        raw.parse().map_err(serde::de::Error::custom)
    }
}

/// A DID URL: a DID plus an optional path, query and fragment.
///
/// DID Core 1.0 § 3.2 defines DID URLs as the dereferenceable form of a
/// DID. The fragment commonly identifies a specific verification method
/// or service endpoint inside the DID document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DidUrl {
    /// The underlying DID.
    pub did: Did,
    /// Optional path component (everything after the DID and before `?`/`#`).
    pub path: Option<String>,
    /// Optional query component.
    pub query: Option<String>,
    /// Optional fragment component.
    pub fragment: Option<String>,
}

impl DidUrl {
    /// Parse a DID URL from its textual form.
    pub fn parse(s: &str) -> Result<Self, DidError> {
        // Pull off fragment first, then query, then path; the remainder
        // must parse as a DID.
        let (head, fragment) = match s.split_once('#') {
            Some((h, f)) => (h, Some(f.to_string())),
            None => (s, None),
        };
        let (head, query) = match head.split_once('?') {
            Some((h, q)) => (h, Some(q.to_string())),
            None => (head, None),
        };
        // The path begins at the first `/` *after* the DID's
        // `did:method:id` portion. The method-specific id is allowed to
        // contain `:` but not `/`.
        let rest = head.strip_prefix("did:").ok_or_else(|| {
            DidError::InvalidDidUrl(format!("missing did: prefix in {s}"))
        })?;
        let (method, after_method) = rest.split_once(':').ok_or_else(|| {
            DidError::InvalidDidUrl(format!(
                "missing method-specific id in {s}"
            ))
        })?;
        let (msid, path) = match after_method.split_once('/') {
            Some((id, p)) => (id, Some(format!("/{p}"))),
            None => (after_method, None),
        };
        let did = Did::new(DidMethod::from_name(method), msid)?;
        Ok(DidUrl {
            did,
            path,
            query,
            fragment,
        })
    }
}

impl fmt::Display for DidUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.did)?;
        if let Some(p) = &self.path {
            f.write_str(p)?;
        }
        if let Some(q) = &self.query {
            write!(f, "?{q}")?;
        }
        if let Some(fr) = &self.fragment {
            write!(f, "#{fr}")?;
        }
        Ok(())
    }
}

impl FromStr for DidUrl {
    type Err = DidError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        DidUrl::parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_did_key() {
        let s = "did:key:z6MkpTHR8VNsBxYAAWHut2Geadd9jSwuBV8xRoAnwWsdvktH";
        let d: Did = s.parse().unwrap();
        assert_eq!(d.method, DidMethod::Key);
        assert_eq!(d.to_string(), s);
    }

    #[test]
    fn parse_did_web_with_port() {
        let s = "did:web:example.com%3A8443:users:alice";
        let d: Did = s.parse().unwrap();
        assert_eq!(d.method, DidMethod::Web);
        assert_eq!(d.method_specific_id, "example.com%3A8443:users:alice");
    }

    #[test]
    fn parse_custom_method() {
        let d: Did = "did:foo:bar".parse().unwrap();
        assert!(matches!(d.method, DidMethod::Custom(ref m) if m == "foo"));
    }

    #[test]
    fn rejects_missing_prefix() {
        assert!("key:zABC".parse::<Did>().is_err());
    }

    #[test]
    fn rejects_empty_msid() {
        assert!("did:key:".parse::<Did>().is_err());
    }

    #[test]
    fn parse_did_url_with_fragment() {
        let s = "did:key:z6MkpTHR8VNsBxYAAWHut2Geadd9jSwuBV8xRoAnwWsdvktH#keys-1";
        let u: DidUrl = s.parse().unwrap();
        assert_eq!(u.fragment.as_deref(), Some("keys-1"));
        assert_eq!(u.path, None);
        assert_eq!(u.to_string(), s);
    }

    #[test]
    fn parse_did_url_with_path_and_query() {
        let s = "did:web:example.com:alice/profile?versionId=2#key-1";
        let u: DidUrl = s.parse().unwrap();
        assert_eq!(u.path.as_deref(), Some("/profile"));
        assert_eq!(u.query.as_deref(), Some("versionId=2"));
        assert_eq!(u.fragment.as_deref(), Some("key-1"));
    }
}
