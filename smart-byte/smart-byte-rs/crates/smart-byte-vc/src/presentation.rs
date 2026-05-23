//! W3C Verifiable Presentation.
//!
//! A presentation bundles one or more credentials with a holder-bound
//! proof (typically using `proofPurpose: authentication`).

use iref::IriBuf;
use serde::{Deserialize, Serialize};

use crate::credential::{VC_CONTEXT_V2, VerifiableCredential};
use crate::did::Did;
use crate::error::VcError;
use crate::proof::Proof;

/// Verifiable Presentation per VCDM 2.0 §6.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiablePresentation {
    /// `@context`.
    #[serde(rename = "@context")]
    pub context: Vec<IriBuf>,
    /// Optional presentation id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<IriBuf>,
    /// `type`. MUST include `VerifiablePresentation`.
    #[serde(rename = "type")]
    pub type_: Vec<String>,
    /// Embedded credentials (0..n).
    #[serde(
        default,
        rename = "verifiableCredential",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub verifiable_credential: Vec<VerifiableCredential>,
    /// Optional holder DID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holder: Option<Did>,
    /// Holder-bound proofs (0..n).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub proof: Vec<Proof>,
}

impl VerifiablePresentation {
    /// New empty presentation with the v2 context and default type.
    pub fn new() -> Result<Self, VcError> {
        let ctx: IriBuf = VC_CONTEXT_V2
            .parse()
            .map_err(|e: iref::InvalidIri<_>| VcError::Iri(e.to_string()))?;
        Ok(Self {
            context: vec![ctx],
            id: None,
            type_: vec!["VerifiablePresentation".to_string()],
            verifiable_credential: Vec::new(),
            holder: None,
            proof: Vec::new(),
        })
    }

    /// Attach a credential.
    pub fn with_credential(mut self, vc: VerifiableCredential) -> Self {
        self.verifiable_credential.push(vc);
        self
    }

    /// Set the holder DID.
    pub fn with_holder(mut self, did: Did) -> Self {
        self.holder = Some(did);
        self
    }

    /// Validate structural requirements.
    pub fn validate_shape(&self) -> Result<(), VcError> {
        match self.context.first() {
            Some(c) if c.as_str() == VC_CONTEXT_V2 => Ok(()),
            _ => Err(VcError::Credential(format!(
                "@context must begin with {VC_CONTEXT_V2}"
            ))),
        }?;
        if !self.type_.iter().any(|t| t == "VerifiablePresentation") {
            return Err(VcError::Credential(
                "type must include VerifiablePresentation".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credential::{CredentialSubject, VcBuilder};
    use crate::issuer::Issuer;

    #[test]
    fn new_presentation_has_defaults() {
        let vp = VerifiablePresentation::new().unwrap();
        assert!(vp.type_.contains(&"VerifiablePresentation".to_string()));
        vp.validate_shape().unwrap();
    }

    #[test]
    fn presentation_with_credential() {
        let subj = CredentialSubject {
            id: Some("did:example:alice".parse().unwrap()),
            claims: serde_json::Map::new(),
        };
        let vc = VcBuilder::new()
            .issuer(Issuer::Uri("did:example:issuer".parse().unwrap()))
            .subject(subj)
            .build()
            .unwrap();
        let holder: Did = "did:example:alice".parse().unwrap();
        let vp = VerifiablePresentation::new()
            .unwrap()
            .with_holder(holder)
            .with_credential(vc);
        assert_eq!(vp.verifiable_credential.len(), 1);
    }
}
