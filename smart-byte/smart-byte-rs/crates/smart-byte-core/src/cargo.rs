//! Envelope cargo: the payload an envelope carries.
//!
//! Cargo is intentionally a small closed set in v1.0. New cargo types
//! land as additional variants in this enum once their canonical
//! encoding has been settled in the spec.

use serde::{Deserialize, Serialize};

/// Discriminated union of supported cargo payloads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum Cargo {
    /// Opaque bytes. The application interprets these.
    Bytes(Vec<u8>),
    /// USD claim, denominated in minor units (cents).
    Usd { minor: i64 },
    /// Joule claim, denominated in microjoules.
    JouleClaim { microjoules: u64 },
    /// Application-defined cargo, identified by a URI namespace.
    Custom { type_uri: String, body: Vec<u8> },
}

impl Cargo {
    /// Returns a short tag identifying the cargo kind. Useful for
    /// logging and CLI output.
    pub fn kind(&self) -> &'static str {
        match self {
            Cargo::Bytes(_) => "bytes",
            Cargo::Usd { .. } => "usd",
            Cargo::JouleClaim { .. } => "joule_claim",
            Cargo::Custom { .. } => "custom",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_are_distinct() {
        assert_eq!(Cargo::Bytes(vec![1, 2, 3]).kind(), "bytes");
        assert_eq!(Cargo::Usd { minor: 100 }.kind(), "usd");
        assert_eq!(
            Cargo::JouleClaim { microjoules: 42 }.kind(),
            "joule_claim"
        );
        assert_eq!(
            Cargo::Custom {
                type_uri: "urn:example:foo".into(),
                body: vec![]
            }
            .kind(),
            "custom"
        );
    }

    #[test]
    fn cbor_roundtrip() {
        let c = Cargo::Usd { minor: 10_000 };
        let bytes = serde_cbor::to_vec(&c).unwrap();
        let back: Cargo = serde_cbor::from_slice(&bytes).unwrap();
        assert_eq!(c, back);
    }
}
