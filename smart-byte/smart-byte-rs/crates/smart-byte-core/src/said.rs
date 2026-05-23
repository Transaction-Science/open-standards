//! Self-Addressing IDentifiers.
//!
//! A [`Said`] is a 32-byte BLAKE3 digest, displayed as bytewise base32
//! (RFC 4648, lower-case, no padding). The substrate uses SAIDs for
//! envelope ids and for principal identities.

use std::fmt;

use data_encoding::BASE32_NOPAD;
use serde::{Deserialize, Serialize};

/// Errors returned when parsing a [`Said`] from text.
#[derive(Debug, thiserror::Error)]
pub enum SaidError {
    #[error("said decode failed: {0}")]
    Decode(String),
    #[error("said must be 32 bytes; got {0}")]
    WrongLength(usize),
}

/// 32-byte self-addressing identifier.
///
/// Equality is byte-equality. The textual form is upper-case base32
/// without padding so it can be embedded in URLs and copied by humans.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Said(pub [u8; 32]);

impl Said {
    /// Construct a SAID by hashing arbitrary bytes with BLAKE3.
    pub fn hash(bytes: &[u8]) -> Self {
        let h = blake3::hash(bytes);
        Self(*h.as_bytes())
    }

    /// Borrow the underlying 32 bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Encode as upper-case base32 without padding.
    pub fn to_base32(&self) -> String {
        BASE32_NOPAD.encode(&self.0)
    }

    /// Decode from base32 (case-insensitive, no padding).
    pub fn from_base32(s: &str) -> Result<Self, SaidError> {
        let upper = s.to_ascii_uppercase();
        let bytes = BASE32_NOPAD
            .decode(upper.as_bytes())
            .map_err(|e| SaidError::Decode(e.to_string()))?;
        if bytes.len() != 32 {
            return Err(SaidError::WrongLength(bytes.len()));
        }
        let mut out = [0u8; 32];
        out.copy_from_slice(&bytes);
        Ok(Self(out))
    }
}

impl fmt::Display for Said {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_base32())
    }
}

impl fmt::Debug for Said {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Said({})", self.to_base32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable() {
        let a = Said::hash(b"hello");
        let b = Said::hash(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn base32_roundtrip() {
        let said = Said::hash(b"smart-byte");
        let text = said.to_base32();
        let parsed = Said::from_base32(&text).expect("roundtrip");
        assert_eq!(said, parsed);
    }

    #[test]
    fn rejects_short_base32() {
        let err = Said::from_base32("AAAA").unwrap_err();
        assert!(matches!(err, SaidError::WrongLength(_)));
    }
}
