//! The Authentic Chained Data Container body.
//!
//! An ACDC is a labeled bundle of SAIDed sections:
//!
//! | Field | Meaning                                                   |
//! |-------|-----------------------------------------------------------|
//! | `v`   | Version string (e.g. [`crate::VERSION_STRING`]).          |
//! | `d`   | SAID of the whole credential.                             |
//! | `i`   | Issuer AID (KERI controller).                             |
//! | `ri`  | Registry SAID (TEL anchor) — optional for untargeted.     |
//! | `s`   | Schema section SAID.                                      |
//! | `a`   | Attribute section: either an inline object or a SAID.     |
//! | `e`   | Edge section: links to other ACDCs by SAID.               |
//! | `r`   | Rule section: machine-readable terms of use.              |
//!
//! Two flavours are supported per spec §3.4:
//!
//! * **Targeted**: `a` contains an `i` (subject AID). The credential
//!   binds a claim about a specific party.
//! * **Untargeted**: `a` has no `i`; the credential is a bearer-style
//!   claim (e.g. a schema attestation).
//!
//! The SAID of an ACDC is derived as in KERI: place
//! [`crate::SAID_PLACEHOLDER`] into `d`, JCS-encode, BLAKE3-256 the
//! bytes, store the digest back. Every section that carries its own
//! `d` field uses the same procedure.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use smart_byte_core::Said;

use crate::error::{AcdcError, Result};

/// The schema section of an ACDC.
///
/// In the wire form `s` may be either an inline schema object or a
/// pointer (just the schema SAID). We store both arms explicitly so
/// callers can build either shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SchemaSection {
    /// Inline schema. The map's `$id` is the schema SAID.
    Inline(serde_json::Map<String, Value>),
    /// Just a SAID reference to a schema known out-of-band.
    Reference(Said),
}

/// The registry section. An ACDC anchored to a TEL carries the
/// registry's SAID here (`ri`). Untargeted/uncontrolled ACDCs omit it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RegistrySection(pub Said);

/// The attribute section. Either inline (claims about the subject) or a
/// SAID reference into a separate disclosure store (used by selective
/// disclosure when the inline attributes have been replaced with a
/// digest).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeSection {
    /// Inline attribute map. The map's `d` field (if present) is the
    /// section's SAID.
    Inline(serde_json::Map<String, Value>),
    /// Compact form: just the section SAID. Used by selective
    /// disclosure to elide undisclosed attributes.
    Compact(Said),
}

impl AttributeSection {
    /// Borrow the inline map, if this is the inline form.
    pub fn as_inline(&self) -> Option<&serde_json::Map<String, Value>> {
        match self {
            Self::Inline(m) => Some(m),
            Self::Compact(_) => None,
        }
    }

    /// Subject AID for a targeted ACDC, if present. The subject is the
    /// inline attribute map's `i` field.
    pub fn subject_aid(&self) -> Option<&str> {
        self.as_inline()
            .and_then(|m| m.get("i"))
            .and_then(|v| v.as_str())
    }
}

/// The edge section is a JSON object mapping local edge labels to edge
/// records. We store it as an opaque JSON map and let [`crate::edge`]
/// parse the structure when graph traversal is needed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(transparent)]
pub struct EdgeSection(pub serde_json::Map<String, Value>);

/// An Authentic Chained Data Container.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Acdc {
    /// Version string. Defaults to [`crate::VERSION_STRING`].
    pub v: String,
    /// Self-addressing identifier of the whole credential.
    pub d: Said,
    /// Issuer AID (KERI controller AID).
    pub i: String,
    /// Registry SAID for the credential's TEL, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ri: Option<Said>,
    /// Schema section.
    pub s: SchemaSection,
    /// Attribute section.
    pub a: AttributeSection,
    /// Edge section (links to other ACDCs).
    #[serde(default, skip_serializing_if = "is_empty_edges")]
    pub e: EdgeSection,
    /// Rule section.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r: Option<serde_json::Map<String, Value>>,
}

fn is_empty_edges(e: &EdgeSection) -> bool {
    e.0.is_empty()
}

impl Acdc {
    /// True when the attribute section carries a subject AID (`a.i`).
    pub fn is_targeted(&self) -> bool {
        self.a.subject_aid().is_some()
    }

    /// Encode this ACDC as canonical JSON per RFC 8785 (JCS).
    pub fn to_jcs(&self) -> Result<Vec<u8>> {
        serde_jcs::to_vec(self).map_err(|e| AcdcError::Jcs(e.to_string()))
    }

    /// Recompute the SAID over the credential body with `d` replaced by
    /// the spec placeholder, and verify it matches `self.d`.
    pub fn verify_said(&self) -> Result<()> {
        let computed = self.compute_said()?;
        if computed != self.d {
            return Err(AcdcError::SaidMismatch {
                asserted: self.d,
                computed,
            });
        }
        Ok(())
    }

    /// Compute the SAID over the body with `d` set to the placeholder.
    pub fn compute_said(&self) -> Result<Said> {
        let mut tmp = self.clone();
        tmp.d = placeholder_said();
        let bytes = tmp.to_jcs()?;
        Ok(Said::hash(&bytes))
    }

    /// Parse from canonical JSON.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let acdc: Self = serde_json::from_slice(bytes)?;
        Ok(acdc)
    }
}

/// Build an [`Acdc`] from parts and seal it with its SAID.
#[derive(Default, Clone, Debug)]
pub struct AcdcBuilder {
    issuer: Option<String>,
    registry: Option<Said>,
    schema: Option<SchemaSection>,
    attributes: Option<AttributeSection>,
    edges: EdgeSection,
    rules: Option<serde_json::Map<String, Value>>,
}

impl AcdcBuilder {
    /// New empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the issuer AID.
    pub fn issuer(mut self, aid: impl Into<String>) -> Self {
        self.issuer = Some(aid.into());
        self
    }

    /// Anchor to a registry SAID.
    pub fn registry(mut self, said: Said) -> Self {
        self.registry = Some(said);
        self
    }

    /// Set the schema section.
    pub fn schema(mut self, schema: SchemaSection) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Set the attribute section.
    pub fn attributes(mut self, attrs: AttributeSection) -> Self {
        self.attributes = Some(attrs);
        self
    }

    /// Replace the edge section.
    pub fn edges(mut self, edges: EdgeSection) -> Self {
        self.edges = edges;
        self
    }

    /// Add one edge label -> JSON record.
    pub fn edge(mut self, label: impl Into<String>, edge: Value) -> Self {
        self.edges.0.insert(label.into(), edge);
        self
    }

    /// Set the rule section.
    pub fn rules(mut self, rules: serde_json::Map<String, Value>) -> Self {
        self.rules = Some(rules);
        self
    }

    /// Finish building. Computes the SAID and writes it into `d`.
    pub fn build(self) -> Result<Acdc> {
        let issuer = self.issuer.ok_or(AcdcError::MissingField("i"))?;
        let schema = self.schema.ok_or(AcdcError::MissingField("s"))?;
        let attributes = self.attributes.ok_or(AcdcError::MissingField("a"))?;

        let mut acdc = Acdc {
            v: crate::VERSION_STRING.to_string(),
            d: placeholder_said(),
            i: issuer,
            ri: self.registry,
            s: schema,
            a: attributes,
            e: self.edges,
            r: self.rules,
        };
        let said = acdc.compute_said()?;
        acdc.d = said;
        Ok(acdc)
    }
}

/// SAID placeholder used during derivation: 32 zero bytes. Because
/// [`crate::SAID_PLACEHOLDER`] is the *textual* placeholder used in
/// raw-JSON schemes, the binary equivalent inside our typed model is
/// simply a zero digest distinguishable from any real BLAKE3 output.
pub(crate) fn placeholder_said() -> Said {
    Said([0u8; 32])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schema() -> SchemaSection {
        let mut m = serde_json::Map::new();
        m.insert("$id".into(), Value::String("schema-1".into()));
        m.insert("title".into(), Value::String("Sample".into()));
        SchemaSection::Inline(m)
    }

    fn sample_attrs(targeted: bool) -> AttributeSection {
        let mut m = serde_json::Map::new();
        m.insert("name".into(), Value::String("Alice".into()));
        if targeted {
            m.insert("i".into(), Value::String("Bsubject".into()));
        }
        AttributeSection::Inline(m)
    }

    #[test]
    fn build_and_verify_targeted() {
        let acdc = AcdcBuilder::new()
            .issuer("Bissuer")
            .schema(sample_schema())
            .attributes(sample_attrs(true))
            .build()
            .expect("build");
        assert!(acdc.is_targeted());
        acdc.verify_said().expect("verify");
    }

    #[test]
    fn build_and_verify_untargeted() {
        let acdc = AcdcBuilder::new()
            .issuer("Bissuer")
            .schema(sample_schema())
            .attributes(sample_attrs(false))
            .build()
            .expect("build");
        assert!(!acdc.is_targeted());
        acdc.verify_said().expect("verify");
    }

    #[test]
    fn tampered_credential_fails_verify() {
        let mut acdc = AcdcBuilder::new()
            .issuer("Bissuer")
            .schema(sample_schema())
            .attributes(sample_attrs(true))
            .build()
            .expect("build");
        // Mutate the attributes after sealing.
        if let AttributeSection::Inline(m) = &mut acdc.a {
            m.insert("extra".into(), Value::String("tampered".into()));
        }
        assert!(matches!(
            acdc.verify_said(),
            Err(AcdcError::SaidMismatch { .. })
        ));
    }
}
