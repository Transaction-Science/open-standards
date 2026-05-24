//! Verifiable Presentation builder.
//!
//! A [`VerifiablePresentation`] bundles:
//!
//! * a holder-side identifier (DID URL),
//! * a list of disclosed (cleartext) attributes,
//! * a list of [`PredicateEntry`] proofs over hidden attributes,
//!   tagged by scheme and predicate kind,
//! * an optional holder-link commitment (from
//!   [`crate::anoncreds::LinkSecret`]).
//!
//! Each [`PredicateEntry`] carries the proof bytes plus a metadata
//! header (`scheme`, `kind`, `commitment`) so a verifier can route to
//! the correct backend without parsing the proof bytes first.

use serde::{Deserialize, Serialize};

use crate::error::ZkError;

/// A single predicate proof inside a [`VerifiablePresentation`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredicateEntry {
    /// Attribute the predicate is over (e.g. `"age"`).
    pub attribute: String,
    /// Scheme tag, e.g. `"bulletproofs"`, `"groth16-stub"`,
    /// `"plonk-stub"`.
    pub scheme: String,
    /// Predicate kind, e.g. `"range"`, `"inequality"`,
    /// `"set-membership"`.
    pub kind: String,
    /// Public commitment the proof binds to (e.g. compressed Pedersen
    /// point for Bulletproofs predicates), as raw bytes.
    #[serde(with = "serde_bytes")]
    pub commitment: Vec<u8>,
    /// Opaque proof bytes — interpretation defined by `scheme`.
    #[serde(with = "serde_bytes")]
    pub proof: Vec<u8>,
}

/// Disclosed attribute echoed in cleartext.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DisclosedAttribute {
    /// Attribute name.
    pub name: String,
    /// Attribute value (UTF-8 encoded).
    pub value: String,
}

/// W3C-style Verifiable Presentation envelope.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifiablePresentation {
    /// Holder identifier (e.g. DID URL).
    pub holder: String,
    /// Disclosed attributes carried in cleartext.
    pub disclosed: Vec<DisclosedAttribute>,
    /// Predicate proofs over hidden attributes.
    pub predicates: Vec<PredicateEntry>,
    /// Optional link-secret commitment binding multiple presentations
    /// to the same holder. `None` for one-shot presentations.
    #[serde(with = "serde_bytes")]
    pub link_commitment: Option<[u8; 32]>,
}

/// Fluent builder for a [`VerifiablePresentation`].
#[derive(Debug)]
pub struct VerifiablePresentationBuilder {
    holder: String,
    disclosed: Vec<DisclosedAttribute>,
    predicates: Vec<PredicateEntry>,
    link_commitment: Option<[u8; 32]>,
}

impl VerifiablePresentationBuilder {
    /// Start a new builder bound to a holder DID URL.
    pub fn new(holder: impl Into<String>) -> Self {
        Self {
            holder: holder.into(),
            disclosed: Vec::new(),
            predicates: Vec::new(),
            link_commitment: None,
        }
    }

    /// Disclose a single attribute in cleartext.
    pub fn disclose(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.disclosed.push(DisclosedAttribute {
            name: name.into(),
            value: value.into(),
        });
        self
    }

    /// Attach a predicate proof.
    pub fn add_predicate(mut self, entry: PredicateEntry) -> Self {
        self.predicates.push(entry);
        self
    }

    /// Attach a link-secret commitment.
    pub fn link_commitment(mut self, commitment: [u8; 32]) -> Self {
        self.link_commitment = Some(commitment);
        self
    }

    /// Finalise the envelope.
    pub fn build(self) -> Result<VerifiablePresentation, ZkError> {
        if self.holder.is_empty() {
            return Err(ZkError::Encoding("holder must be non-empty".to_string()));
        }
        Ok(VerifiablePresentation {
            holder: self.holder,
            disclosed: self.disclosed,
            predicates: self.predicates,
            link_commitment: self.link_commitment,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_assembles_envelope() {
        let vp = VerifiablePresentationBuilder::new("did:example:alice")
            .disclose("given_name", "Alice")
            .add_predicate(PredicateEntry {
                attribute: "age".into(),
                scheme: "bulletproofs".into(),
                kind: "inequality".into(),
                commitment: vec![0u8; 32],
                proof: vec![1, 2, 3],
            })
            .link_commitment([9u8; 32])
            .build()
            .expect("build");
        assert_eq!(vp.holder, "did:example:alice");
        assert_eq!(vp.disclosed.len(), 1);
        assert_eq!(vp.predicates.len(), 1);
        assert!(vp.link_commitment.is_some());
    }

    #[test]
    fn empty_holder_rejected() {
        let r = VerifiablePresentationBuilder::new("").build();
        assert!(r.is_err());
    }
}
