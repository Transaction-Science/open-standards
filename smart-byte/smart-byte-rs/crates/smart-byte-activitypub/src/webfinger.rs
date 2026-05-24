//! Webfinger (RFC 7033) — user discovery for ActivityPub.
//!
//! A federation peer that wants to find `acct:alice@example.test` does:
//!
//! ```text
//! GET https://example.test/.well-known/webfinger?resource=acct:alice@example.test
//! ```
//!
//! The server replies with a [`Jrd`] document containing one or more
//! [`Link`] entries. The interesting one carries `rel =
//! "self"` + `type = "application/activity+json"` and a `href` pointing
//! at the actor IRI.
//!
//! This module models the document and provides a pure parser /
//! constructor. HTTP fetching is out of scope — callers wire it in.

use crate::error::{ActivityPubError, Result};
use crate::vocabulary::ACTIVITY_JSON;
use serde::{Deserialize, Serialize};

/// A single link entry in a Webfinger JRD.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Link {
    /// The relation, e.g. `"self"`, `"http://webfinger.net/rel/profile-page"`.
    pub rel: String,
    /// MIME type of the target representation.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub type_field: Option<String>,
    /// Target URI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub href: Option<String>,
    /// URI template for the link.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
}

/// A Webfinger JRD (JSON Resource Descriptor) — RFC 7033 §4.4.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Jrd {
    /// The `acct:` URI the document is describing.
    pub subject: String,
    /// Alternate identifiers for the subject.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    /// Type-specific properties.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub properties: std::collections::BTreeMap<String, serde_json::Value>,
    /// Discovery links — at least one `rel="self"` should be present.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<Link>,
}

impl Jrd {
    /// Build a JRD for the given `acct:` resource pointing at the
    /// actor IRI.
    pub fn for_actor(acct: impl Into<String>, actor_iri: impl Into<String>) -> Self {
        let acct = acct.into();
        let actor_iri = actor_iri.into();
        Self {
            subject: acct.clone(),
            aliases: vec![actor_iri.clone()],
            properties: std::collections::BTreeMap::new(),
            links: vec![
                Link {
                    rel: "self".to_string(),
                    type_field: Some(ACTIVITY_JSON.to_string()),
                    href: Some(actor_iri),
                    template: None,
                },
                Link {
                    rel: "http://webfinger.net/rel/profile-page".to_string(),
                    type_field: Some("text/html".to_string()),
                    href: None,
                    template: None,
                },
            ],
        }
    }

    /// Return the first link whose `rel` equals the supplied string.
    pub fn link_by_rel(&self, rel: &str) -> Option<&Link> {
        self.links.iter().find(|l| l.rel == rel)
    }

    /// Resolve the actor IRI by looking for `rel="self"` +
    /// `type="application/activity+json"`. Falls back to the first
    /// `rel="self"` if no typed entry exists.
    pub fn actor_iri(&self) -> Option<&str> {
        let typed = self.links.iter().find(|l| {
            l.rel == "self"
                && l.type_field
                    .as_deref()
                    .map(|t| t == ACTIVITY_JSON)
                    .unwrap_or(false)
        });
        let chosen = typed.or_else(|| self.link_by_rel("self"));
        chosen.and_then(|l| l.href.as_deref())
    }

    /// Serialise to canonical JSON (JRD).
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Parse a JRD, enforcing that the subject is non-empty.
    pub fn from_json(s: &str) -> Result<Self> {
        let jrd: Jrd = serde_json::from_str(s)?;
        if jrd.subject.is_empty() {
            return Err(ActivityPubError::Webfinger("jrd missing subject".into()));
        }
        Ok(jrd)
    }
}

/// Parse a Webfinger `resource` query value, returning `(user, host)`.
///
/// Accepts `acct:alice@example.test` and `alice@example.test`.
pub fn parse_acct(resource: &str) -> Result<(String, String)> {
    let body = resource.strip_prefix("acct:").unwrap_or(resource);
    let (user, host) = body
        .split_once('@')
        .ok_or_else(|| ActivityPubError::Webfinger(format!("missing @ in {resource}")))?;
    if user.is_empty() || host.is_empty() {
        return Err(ActivityPubError::Webfinger(format!(
            "empty user or host in {resource}"
        )));
    }
    Ok((user.to_string(), host.to_string()))
}

/// Build the canonical Webfinger query URL for the given `acct` and
/// host. Useful for clients fetching a remote actor.
pub fn discovery_url(host: &str, acct: &str) -> String {
    let resource = if acct.starts_with("acct:") {
        acct.to_string()
    } else {
        format!("acct:{acct}")
    };
    format!(
        "https://{host}/.well-known/webfinger?resource={}",
        percent_encode(&resource)
    )
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        // Conservative unreserved set per RFC 3986.
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push_str(&format!("{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jrd_roundtrip() -> Result<()> {
        let j = Jrd::for_actor("acct:alice@example.test", "https://example.test/users/alice");
        let s = j.to_json()?;
        let k = Jrd::from_json(&s)?;
        assert_eq!(j, k);
        assert_eq!(k.actor_iri(), Some("https://example.test/users/alice"));
        Ok(())
    }

    #[test]
    fn parse_acct_works() -> Result<()> {
        let (u, h) = parse_acct("acct:alice@example.test")?;
        assert_eq!(u, "alice");
        assert_eq!(h, "example.test");
        let (u2, h2) = parse_acct("bob@b.test")?;
        assert_eq!(u2, "bob");
        assert_eq!(h2, "b.test");
        Ok(())
    }

    #[test]
    fn parse_acct_rejects_malformed() {
        assert!(parse_acct("noseparator").is_err());
        assert!(parse_acct("acct:@host").is_err());
        assert!(parse_acct("acct:user@").is_err());
    }

    #[test]
    fn discovery_url_shape() {
        let u = discovery_url("example.test", "alice@example.test");
        assert_eq!(
            u,
            "https://example.test/.well-known/webfinger?resource=acct%3Aalice%40example.test"
        );
    }
}
