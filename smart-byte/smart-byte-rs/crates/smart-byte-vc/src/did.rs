//! Decentralized Identifiers (DIDs, W3C Rec).
//!
//! This module implements the small subset of the DID syntax that
//! Smart Byte needs to identify issuers, holders, and verification
//! methods. Full DID Resolution / DID Document handling is intentionally
//! out of scope for this crate; `did:key` is implemented natively
//! because it round-trips Ed25519 public keys without requiring a
//! resolver.

use std::fmt;
use std::str::FromStr;

use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};

use crate::error::VcError;

/// W3C DID syntax errors.
#[derive(Debug, thiserror::Error)]
pub enum DidError {
    /// String did not begin with the `did:` scheme.
    #[error("missing did: scheme")]
    MissingScheme,
    /// The method-name segment was empty.
    #[error("empty did method")]
    EmptyMethod,
    /// The method-specific-id segment was empty.
    #[error("empty method-specific id")]
    EmptyId,
    /// `did:key` decoding failed.
    #[error("did:key decode error: {0}")]
    DidKey(String),
}

/// A parsed DID. The grammar is `did:METHOD:METHOD_SPECIFIC_ID[/PATH][#FRAGMENT]`.
///
/// Round-trips losslessly via [`Display`] and [`FromStr`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Did {
    method: String,
    method_specific_id: String,
    path: Option<String>,
    fragment: Option<String>,
}

impl Did {
    /// Construct a DID from its parts. Panics-free; the caller is
    /// responsible for syntactic validity beyond emptiness, which is
    /// checked here.
    pub fn new(
        method: impl Into<String>,
        method_specific_id: impl Into<String>,
    ) -> Result<Self, DidError> {
        let method = method.into();
        let id = method_specific_id.into();
        if method.is_empty() {
            return Err(DidError::EmptyMethod);
        }
        if id.is_empty() {
            return Err(DidError::EmptyId);
        }
        Ok(Self {
            method,
            method_specific_id: id,
            path: None,
            fragment: None,
        })
    }

    /// Return the DID method name (e.g. `"key"` for `did:key:...`).
    pub fn method(&self) -> &str {
        &self.method
    }

    /// Return the method-specific id (everything after `did:METHOD:`,
    /// up to the first `/` or `#`).
    pub fn method_specific_id(&self) -> &str {
        &self.method_specific_id
    }

    /// Return the path component, if present.
    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// Return the fragment, if present.
    pub fn fragment(&self) -> Option<&str> {
        self.fragment.as_deref()
    }

    /// Attach a fragment (used to address a verification method).
    pub fn with_fragment(mut self, fragment: impl Into<String>) -> Self {
        self.fragment = Some(fragment.into());
        self
    }
}

impl fmt::Display for Did {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "did:{}:{}", self.method, self.method_specific_id)?;
        if let Some(p) = &self.path {
            write!(f, "/{}", p)?;
        }
        if let Some(frag) = &self.fragment {
            write!(f, "#{}", frag)?;
        }
        Ok(())
    }
}

impl FromStr for Did {
    type Err = DidError;

    fn from_str(s: &str) -> Result<Self, DidError> {
        let rest = s.strip_prefix("did:").ok_or(DidError::MissingScheme)?;
        // method
        let (method, rest) = rest
            .split_once(':')
            .ok_or(DidError::EmptyId)?;
        if method.is_empty() {
            return Err(DidError::EmptyMethod);
        }
        // fragment
        let (rest, fragment) = match rest.split_once('#') {
            Some((a, b)) => (a, Some(b.to_string())),
            None => (rest, None),
        };
        // path
        let (id, path) = match rest.split_once('/') {
            Some((a, b)) => (a, Some(b.to_string())),
            None => (rest, None),
        };
        if id.is_empty() {
            return Err(DidError::EmptyId);
        }
        Ok(Self {
            method: method.to_string(),
            method_specific_id: id.to_string(),
            path,
            fragment,
        })
    }
}

impl TryFrom<String> for Did {
    type Error = DidError;
    fn try_from(s: String) -> Result<Self, DidError> {
        s.parse()
    }
}

impl From<Did> for String {
    fn from(d: Did) -> String {
        d.to_string()
    }
}

/// `did:key` ergonomics.
///
/// The `did:key` method encodes a public key as a multibase-multicodec
/// string. For Ed25519 the multicodec prefix is the varint `0xed 0x01`.
pub struct DidKey;

const ED25519_MULTICODEC: [u8; 2] = [0xed, 0x01];

impl DidKey {
    /// Build a `did:key:z…` DID from an Ed25519 verifying key.
    pub fn from_ed25519(key: &VerifyingKey) -> Did {
        let mut buf = Vec::with_capacity(2 + 32);
        buf.extend_from_slice(&ED25519_MULTICODEC);
        buf.extend_from_slice(key.as_bytes());
        let mb = multibase::encode(multibase::Base::Base58Btc, buf);
        // Construction is infallible: `mb` is non-empty.
        Did::new("key", mb).expect("did:key id is non-empty")
    }

    /// Decode the Ed25519 verifying key from a `did:key:z…` DID.
    pub fn to_ed25519(did: &Did) -> Result<VerifyingKey, VcError> {
        if did.method() != "key" {
            return Err(VcError::Did(format!(
                "expected did:key, got did:{}",
                did.method()
            )));
        }
        let id = did.method_specific_id();
        let (_, bytes) = multibase::decode(id)
            .map_err(|e| VcError::Did(format!("multibase decode: {e}")))?;
        if bytes.len() != 34 || bytes[0..2] != ED25519_MULTICODEC {
            return Err(VcError::Did("not an Ed25519 did:key".into()));
        }
        let mut k = [0u8; 32];
        k.copy_from_slice(&bytes[2..34]);
        VerifyingKey::from_bytes(&k)
            .map_err(|e| VcError::Did(format!("bad ed25519 key: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[test]
    fn parses_simple_did() {
        let d: Did = "did:example:abc123".parse().unwrap();
        assert_eq!(d.method(), "example");
        assert_eq!(d.method_specific_id(), "abc123");
        assert_eq!(d.to_string(), "did:example:abc123");
    }

    #[test]
    fn parses_did_with_fragment() {
        let d: Did = "did:example:abc#keys-1".parse().unwrap();
        assert_eq!(d.fragment(), Some("keys-1"));
        assert_eq!(d.to_string(), "did:example:abc#keys-1");
    }

    #[test]
    fn rejects_missing_scheme() {
        assert!("example:abc".parse::<Did>().is_err());
    }

    #[test]
    fn did_key_roundtrip() {
        let sk = SigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let did = DidKey::from_ed25519(&vk);
        let back = DidKey::to_ed25519(&did).unwrap();
        assert_eq!(back.as_bytes(), vk.as_bytes());
        assert!(did.to_string().starts_with("did:key:z"));
    }
}
