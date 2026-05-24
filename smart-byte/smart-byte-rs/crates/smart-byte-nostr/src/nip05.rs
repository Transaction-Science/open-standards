//! NIP-05 DNS-based human-readable identifiers.
//!
//! A NIP-05 identifier looks like `name@domain`. The server hosts a JSON
//! document at `https://<domain>/.well-known/nostr.json?name=<name>`
//! mapping `name` to the user's hex pubkey, optionally with relay hints.
//!
//! This module models the document shape and a pure verification
//! function. Actual HTTP fetching is intentionally NOT in this crate —
//! callers wire in `reqwest` or any other HTTP client and pass the
//! response body in.

use crate::error::NostrError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// NIP-05 well-known document.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Nip05Document {
    /// `name -> pubkey hex` map.
    #[serde(default)]
    pub names: BTreeMap<String, String>,
    /// Optional `pubkey hex -> relay urls` map.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relays: Option<BTreeMap<String, Vec<String>>>,
}

/// Parsed `name@domain` identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nip05Address {
    /// Local part (the `name`). The reserved value `"_"` denotes the
    /// bare domain.
    pub name: String,
    /// Domain (RFC1035 hostname).
    pub domain: String,
}

impl Nip05Address {
    /// Parse `name@domain`. If the input has no `@`, the local part is
    /// inferred as `"_"` and the whole input becomes the domain.
    pub fn parse(s: &str) -> Result<Self, NostrError> {
        if let Some((name, domain)) = s.split_once('@') {
            if name.is_empty() || domain.is_empty() {
                return Err(NostrError::Nip05("empty name or domain".into()));
            }
            Ok(Self {
                name: name.to_string(),
                domain: domain.to_string(),
            })
        } else if s.is_empty() {
            Err(NostrError::Nip05("empty identifier".into()))
        } else {
            Ok(Self {
                name: "_".to_string(),
                domain: s.to_string(),
            })
        }
    }

    /// The URL the resolver should fetch for this identifier.
    pub fn well_known_url(&self) -> String {
        format!(
            "https://{}/.well-known/nostr.json?name={}",
            self.domain, self.name
        )
    }
}

/// Resolve a NIP-05 identifier against a fetched document.
///
/// Returns the hex pubkey associated with the address' `name`, or an
/// error if the document does not list this name or lists a different
/// pubkey than expected.
pub fn resolve(addr: &Nip05Address, doc: &Nip05Document) -> Result<String, NostrError> {
    doc.names
        .get(&addr.name)
        .cloned()
        .ok_or_else(|| NostrError::Nip05(format!("name '{}' not in document", addr.name)))
}

/// Verify that `expected_pubkey_hex` matches what the document maps
/// `addr.name` to.
pub fn verify(
    addr: &Nip05Address,
    doc: &Nip05Document,
    expected_pubkey_hex: &str,
) -> Result<(), NostrError> {
    let found = resolve(addr, doc)?;
    if found.eq_ignore_ascii_case(expected_pubkey_hex) {
        Ok(())
    } else {
        Err(NostrError::Nip05(format!(
            "pubkey mismatch: doc='{found}' expected='{expected_pubkey_hex}'"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_with_and_without_at() {
        let a = Nip05Address::parse("alice@example.com").expect("parse");
        assert_eq!(a.name, "alice");
        assert_eq!(a.domain, "example.com");
        let b = Nip05Address::parse("example.com").expect("parse");
        assert_eq!(b.name, "_");
        assert_eq!(b.domain, "example.com");
    }

    #[test]
    fn resolve_and_verify() {
        let mut doc = Nip05Document::default();
        doc.names.insert("alice".into(), "deadbeef".into());
        let addr = Nip05Address::parse("alice@example.com").expect("parse");
        verify(&addr, &doc, "deadbeef").expect("verify ok");
        assert!(verify(&addr, &doc, "cafebabe").is_err());
    }
}
