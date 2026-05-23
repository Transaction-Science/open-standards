//! VC Issuer.
//!
//! Per W3C VCDM 2.0, an issuer may be a bare IRI or an object whose
//! `id` is the issuer IRI plus optional descriptive metadata. We model
//! both shapes losslessly so round-trips preserve test-vector form.

use iref::IriBuf;
use serde::{Deserialize, Serialize};

/// `issuer` field of a VC.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Issuer {
    /// Bare IRI form.
    Uri(IriBuf),
    /// Object form with optional descriptive metadata.
    Object {
        /// Issuer IRI.
        id: IriBuf,
        /// Optional human-readable name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Optional image IRI (e.g. logo).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image: Option<IriBuf>,
    },
}

impl Issuer {
    /// Return the issuer IRI regardless of which form is used.
    pub fn id(&self) -> &IriBuf {
        match self {
            Issuer::Uri(u) => u,
            Issuer::Object { id, .. } => id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uri_roundtrip() {
        let json = "\"did:example:issuer\"";
        let iss: Issuer = serde_json::from_str(json).unwrap();
        assert_eq!(iss.id().as_str(), "did:example:issuer");
        let back = serde_json::to_string(&iss).unwrap();
        assert_eq!(back, json);
    }

    #[test]
    fn object_form() {
        let json = "{\"id\":\"did:example:issuer\",\"name\":\"Acme\"}";
        let iss: Issuer = serde_json::from_str(json).unwrap();
        match iss {
            Issuer::Object { name, .. } => assert_eq!(name.as_deref(), Some("Acme")),
            _ => panic!("expected object form"),
        }
    }
}
