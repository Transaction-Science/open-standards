//! W3C Verifiable Credential Data Model 2.0 in-memory form.
//!
//! The shape follows §4 of the VCDM 2.0 Recommendation. Field naming is
//! `serde`-renamed to match the JSON wire form (`@context`, `type`,
//! `validFrom`, `credentialSubject`, …).

use chrono::{DateTime, Utc};
use iref::IriBuf;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::VcError;
use crate::issuer::Issuer;
use crate::proof::Proof;

/// The mandatory v2 context.
pub const VC_CONTEXT_V2: &str = "https://www.w3.org/ns/credentials/v2";

/// A claim about a subject. The W3C model allows arbitrary properties
/// alongside the optional `id`, so we store the body as a `serde_json::Value`
/// to preserve round-trip semantics with real test vectors.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialSubject {
    /// Subject IRI, if known. May be absent for bearer-style credentials.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<IriBuf>,
    /// Remaining claim properties as raw JSON.
    #[serde(flatten)]
    pub claims: serde_json::Map<String, Value>,
}

/// `credentialStatus` entry — references a status mechanism (commonly
/// Bitstring Status List 2021).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialStatus {
    /// Status entry IRI.
    pub id: IriBuf,
    /// Type tag, e.g. `BitstringStatusListEntry`.
    #[serde(rename = "type")]
    pub type_: String,
    /// Index into the bitstring.
    #[serde(default, rename = "statusListIndex", skip_serializing_if = "Option::is_none")]
    pub status_list_index: Option<String>,
    /// IRI of the status list credential.
    #[serde(default, rename = "statusListCredential", skip_serializing_if = "Option::is_none")]
    pub status_list_credential: Option<IriBuf>,
    /// Purpose: `revocation`, `suspension`, etc.
    #[serde(default, rename = "statusPurpose", skip_serializing_if = "Option::is_none")]
    pub status_purpose: Option<String>,
}

/// `termsOfUse` entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TermsOfUse {
    /// Optional IRI of this terms entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<IriBuf>,
    /// Type tag.
    #[serde(rename = "type")]
    pub type_: String,
    /// Remaining policy properties.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// `evidence` entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evidence {
    /// Optional IRI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<IriBuf>,
    /// Type tag.
    #[serde(rename = "type")]
    pub type_: Vec<String>,
    /// Remaining evidence properties.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// A W3C VCDM 2.0 Verifiable Credential.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiableCredential {
    /// `@context`. MUST begin with the v2 context IRI.
    #[serde(rename = "@context")]
    pub context: Vec<IriBuf>,
    /// Optional credential identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<IriBuf>,
    /// `type`. MUST include `VerifiableCredential`.
    #[serde(rename = "type")]
    pub type_: Vec<String>,
    /// Issuer.
    pub issuer: Issuer,
    /// Validity start.
    #[serde(default, rename = "validFrom", skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<DateTime<Utc>>,
    /// Validity end.
    #[serde(default, rename = "validUntil", skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<DateTime<Utc>>,
    /// One or more credential subjects.
    #[serde(rename = "credentialSubject")]
    pub credential_subject: Vec<CredentialSubject>,
    /// Optional status entry.
    #[serde(default, rename = "credentialStatus", skip_serializing_if = "Option::is_none")]
    pub credential_status: Option<CredentialStatus>,
    /// `termsOfUse` (0..n).
    #[serde(default, rename = "termsOfUse", skip_serializing_if = "Vec::is_empty")]
    pub terms_of_use: Vec<TermsOfUse>,
    /// `evidence` (0..n).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<Evidence>,
    /// Embedded proofs (0..n). External proofs (JWT) carry the VC payload
    /// outside this struct.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub proof: Vec<Proof>,
}

impl VerifiableCredential {
    /// Encode this credential as canonical JSON per RFC 8785 (JCS).
    pub fn to_jcs(&self) -> Result<Vec<u8>, VcError> {
        serde_jcs::to_vec(self).map_err(|e| VcError::Jcs(e.to_string()))
    }

    /// Encode the credential, with `proof` removed, as canonical JSON.
    /// This is the form signed by Data Integrity proofs.
    pub fn to_jcs_without_proof(&self) -> Result<Vec<u8>, VcError> {
        let mut tmp = self.clone();
        tmp.proof.clear();
        tmp.to_jcs()
    }

    /// Validate the structural requirements of VCDM 2.0:
    /// * `@context[0]` is the v2 IRI.
    /// * `type` contains `VerifiableCredential`.
    /// * `credentialSubject` is non-empty.
    pub fn validate_shape(&self) -> Result<(), VcError> {
        match self.context.first() {
            Some(c) if c.as_str() == VC_CONTEXT_V2 => {}
            _ => {
                return Err(VcError::Credential(format!(
                    "@context must begin with {VC_CONTEXT_V2}"
                )));
            }
        }
        if !self.type_.iter().any(|t| t == "VerifiableCredential") {
            return Err(VcError::Credential(
                "type must include VerifiableCredential".into(),
            ));
        }
        if self.credential_subject.is_empty() {
            return Err(VcError::Credential(
                "credentialSubject must not be empty".into(),
            ));
        }
        Ok(())
    }
}

/// Fluent builder for [`VerifiableCredential`]. Defaults `@context` to
/// the v2 IRI and `type` to `["VerifiableCredential"]`; callers may
/// extend both.
#[derive(Clone, Debug)]
pub struct VcBuilder {
    context: Vec<IriBuf>,
    id: Option<IriBuf>,
    type_: Vec<String>,
    issuer: Option<Issuer>,
    valid_from: Option<DateTime<Utc>>,
    valid_until: Option<DateTime<Utc>>,
    credential_subject: Vec<CredentialSubject>,
    credential_status: Option<CredentialStatus>,
    terms_of_use: Vec<TermsOfUse>,
    evidence: Vec<Evidence>,
}

impl Default for VcBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VcBuilder {
    /// Start a new builder pre-populated with the VCDM 2.0 defaults.
    pub fn new() -> Self {
        let ctx: IriBuf = VC_CONTEXT_V2
            .parse()
            .expect("VC v2 context is a valid IRI");
        Self {
            context: vec![ctx],
            id: None,
            type_: vec!["VerifiableCredential".to_string()],
            issuer: None,
            valid_from: None,
            valid_until: None,
            credential_subject: Vec::new(),
            credential_status: None,
            terms_of_use: Vec::new(),
            evidence: Vec::new(),
        }
    }

    /// Append a context IRI.
    pub fn context(mut self, iri: IriBuf) -> Self {
        self.context.push(iri);
        self
    }

    /// Set the optional credential id.
    pub fn id(mut self, iri: IriBuf) -> Self {
        self.id = Some(iri);
        self
    }

    /// Append a type tag.
    pub fn type_tag(mut self, t: impl Into<String>) -> Self {
        self.type_.push(t.into());
        self
    }

    /// Set the issuer.
    pub fn issuer(mut self, issuer: Issuer) -> Self {
        self.issuer = Some(issuer);
        self
    }

    /// Set `validFrom`.
    pub fn valid_from(mut self, t: DateTime<Utc>) -> Self {
        self.valid_from = Some(t);
        self
    }

    /// Set `validUntil`.
    pub fn valid_until(mut self, t: DateTime<Utc>) -> Self {
        self.valid_until = Some(t);
        self
    }

    /// Append a credential subject.
    pub fn subject(mut self, s: CredentialSubject) -> Self {
        self.credential_subject.push(s);
        self
    }

    /// Set credential status.
    pub fn status(mut self, s: CredentialStatus) -> Self {
        self.credential_status = Some(s);
        self
    }

    /// Append a terms-of-use entry.
    pub fn terms_of_use(mut self, t: TermsOfUse) -> Self {
        self.terms_of_use.push(t);
        self
    }

    /// Append an evidence entry.
    pub fn evidence(mut self, e: Evidence) -> Self {
        self.evidence.push(e);
        self
    }

    /// Finish the build, validating structural requirements.
    pub fn build(self) -> Result<VerifiableCredential, VcError> {
        let issuer = self
            .issuer
            .ok_or_else(|| VcError::Credential("issuer is required".into()))?;
        let vc = VerifiableCredential {
            context: self.context,
            id: self.id,
            type_: self.type_,
            issuer,
            valid_from: self.valid_from,
            valid_until: self.valid_until,
            credential_subject: self.credential_subject,
            credential_status: self.credential_status,
            terms_of_use: self.terms_of_use,
            evidence: self.evidence,
            proof: Vec::new(),
        };
        vc.validate_shape()?;
        Ok(vc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn issuer_did() -> Issuer {
        Issuer::Uri("did:example:issuer".parse().unwrap())
    }

    #[test]
    fn build_minimum_vc() {
        let subj = CredentialSubject {
            id: Some("did:example:alice".parse().unwrap()),
            claims: serde_json::Map::new(),
        };
        let vc = VcBuilder::new()
            .issuer(issuer_did())
            .valid_from(chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap())
            .subject(subj)
            .build()
            .unwrap();
        assert_eq!(vc.context[0].as_str(), VC_CONTEXT_V2);
        assert!(vc.type_.contains(&"VerifiableCredential".to_string()));
    }

    #[test]
    fn rejects_empty_subject() {
        let err = VcBuilder::new().issuer(issuer_did()).build().unwrap_err();
        assert!(matches!(err, VcError::Credential(_)));
    }

    #[test]
    fn jcs_is_stable() {
        let subj = CredentialSubject {
            id: Some("did:example:alice".parse().unwrap()),
            claims: serde_json::Map::new(),
        };
        let vc = VcBuilder::new()
            .issuer(issuer_did())
            .subject(subj)
            .build()
            .unwrap();
        let a = vc.to_jcs().unwrap();
        let b = vc.to_jcs().unwrap();
        assert_eq!(a, b);
    }
}
