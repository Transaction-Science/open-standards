//! HMAC-over-value encrypted indexes (DIF EDV v0.10 § 4.4).
//!
//! The vault is a passive ciphertext store — it cannot see plaintext
//! attribute names or values, but it must still answer equality queries
//! such as "give me every document tagged `type=Note`".
//!
//! The trick is to never send the vault the raw `type` and `Note`
//! strings. Instead the client holds an HMAC-SHA-256 key, computes
//! `name_tag = HMAC(key, "type")` and `value_tag = HMAC(key, "Note")`, and
//! attaches `(name_tag, value_tag)` to the encrypted document. The vault
//! indexes the tags. Later, an authorised client that holds the same key
//! recomputes the tags and asks the vault for documents matching those
//! tags. The vault learns equality but never the underlying values.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::error::EdvError;
use crate::spec::{EncryptedDocument, IndexedAttribute, IndexedEntry};

type HmacSha256 = Hmac<Sha256>;

/// A client-side HMAC key used to derive blinded index tags.
#[derive(Debug, Clone)]
pub struct IndexKey {
    /// Key id (DID URL), surfaced in `EncryptedDocument::indexed[*].hmac`.
    pub kid: String,
    /// 32-byte HMAC-SHA-256 key.
    pub key: [u8; 32],
}

impl IndexKey {
    /// Compute the base64url HMAC-SHA-256 tag over `value`.
    pub fn tag(&self, value: &[u8]) -> Result<String, EdvError> {
        let mut mac = <HmacSha256 as Mac>::new_from_slice(&self.key)
            .map_err(|e| EdvError::Crypto(format!("hmac key: {e}")))?;
        mac.update(value);
        let bytes = mac.finalize().into_bytes();
        Ok(URL_SAFE_NO_PAD.encode(bytes))
    }

    /// Build an [`IndexedAttribute`] for a single `(name, value)` pair.
    pub fn attribute(
        &self,
        name: &str,
        value: &str,
        unique: bool,
    ) -> Result<IndexedAttribute, EdvError> {
        Ok(IndexedAttribute {
            name: self.tag(name.as_bytes())?,
            value: self.tag(value.as_bytes())?,
            unique,
        })
    }

    /// Build an [`IndexedEntry`] from a list of `(name, value)` pairs,
    /// ready to attach to [`EncryptedDocument::indexed`].
    pub fn entry(
        &self,
        attributes: &[(&str, &str)],
    ) -> Result<IndexedEntry, EdvError> {
        let mut out = Vec::with_capacity(attributes.len());
        for (n, v) in attributes {
            out.push(self.attribute(n, v, false)?);
        }
        Ok(IndexedEntry {
            hmac: crate::spec::Hmac::new(self.kid.clone()),
            attributes: out,
        })
    }
}

/// A boolean equality query over an encrypted index.
///
/// All `(name, value)` pairs must match (AND semantics).
#[derive(Debug, Clone, Default)]
pub struct Query {
    /// Required `(name_tag, value_tag)` pairs.
    pub equals: Vec<(String, String)>,
}

impl Query {
    /// Create an empty query.
    pub fn new() -> Self {
        Self {
            equals: Vec::new(),
        }
    }

    /// Add an equality clause, deriving the tags from `key`.
    pub fn equal(
        mut self,
        key: &IndexKey,
        name: &str,
        value: &str,
    ) -> Result<Self, EdvError> {
        self.equals
            .push((key.tag(name.as_bytes())?, key.tag(value.as_bytes())?));
        Ok(self)
    }

    /// Evaluate the query against a single document.
    pub fn matches(&self, doc: &EncryptedDocument) -> bool {
        if self.equals.is_empty() {
            return true;
        }
        let all_tags: Vec<&IndexedAttribute> = doc
            .indexed
            .iter()
            .flat_map(|e| e.attributes.iter())
            .collect();
        self.equals.iter().all(|(n, v)| {
            all_tags.iter().any(|a| &a.name == n && &a.value == v)
        })
    }
}

/// Evaluate `query` against every document in `docs`, returning matches.
pub fn search<'a>(
    docs: &'a [EncryptedDocument],
    query: &Query,
) -> Vec<&'a EncryptedDocument> {
    docs.iter().filter(|d| query.matches(d)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, entry: IndexedEntry) -> EncryptedDocument {
        EncryptedDocument {
            id: id.into(),
            sequence: 0,
            jwe: serde_json::json!({}),
            indexed: vec![entry],
            stream: None,
        }
    }

    #[test]
    fn tag_is_deterministic() {
        let k = IndexKey {
            kid: "k1".into(),
            key: [7u8; 32],
        };
        let a = k.tag(b"value").expect("tag a");
        let b = k.tag(b"value").expect("tag b");
        assert_eq!(a, b);
    }

    #[test]
    fn different_values_different_tags() {
        let k = IndexKey {
            kid: "k1".into(),
            key: [7u8; 32],
        };
        let a = k.tag(b"alice").expect("a");
        let b = k.tag(b"bob").expect("b");
        assert_ne!(a, b);
    }

    #[test]
    fn equality_query_matches() {
        let k = IndexKey {
            kid: "k1".into(),
            key: [7u8; 32],
        };
        let entry = k
            .entry(&[("type", "Note"), ("tag", "private")])
            .expect("entry");
        let doc = make_doc("urn:doc:1", entry);
        let q = Query::new()
            .equal(&k, "type", "Note")
            .expect("equal");
        assert!(q.matches(&doc));
    }

    #[test]
    fn nonmatch_returns_false() {
        let k = IndexKey {
            kid: "k1".into(),
            key: [7u8; 32],
        };
        let entry = k
            .entry(&[("type", "Note")])
            .expect("entry");
        let doc = make_doc("urn:doc:1", entry);
        let q = Query::new()
            .equal(&k, "type", "Photo")
            .expect("equal");
        assert!(!q.matches(&doc));
    }

    #[test]
    fn search_filters_corpus() {
        let k = IndexKey {
            kid: "k1".into(),
            key: [7u8; 32],
        };
        let a = make_doc(
            "urn:a",
            k.entry(&[("type", "Note")]).expect("a"),
        );
        let b = make_doc(
            "urn:b",
            k.entry(&[("type", "Photo")]).expect("b"),
        );
        let q = Query::new()
            .equal(&k, "type", "Note")
            .expect("q");
        let corpus = [a, b];
        let hits = search(&corpus, &q);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "urn:a");
    }
}
