//! Presentation Definition + Presentation Submission (DIF PE 2.0 +
//! OID4VP draft 23 carrier semantics).
//!
//! OID4VP supports two query modes:
//!
//! * **DCQL** ([`crate::dcql`]) — new in draft 23 and preferred for new
//!   deployments.
//! * **Presentation Exchange (PE) 2.0** — legacy but still widely
//!   deployed. The verifier publishes a `presentation_definition` and
//!   the wallet returns a `presentation_submission` alongside the
//!   `vp_token`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::OidcError;

/// One input descriptor inside a presentation definition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputDescriptor {
    /// Descriptor id, referenced by the submission.
    pub id: String,
    /// Optional human-readable name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional human-readable purpose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
    /// Format constraints (which credential formats are acceptable).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    /// JSON Schema / JSONPath constraints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraints: Option<Constraints>,
}

/// PE constraints body. We keep `fields` typed and let everything else
/// round-trip as raw JSON.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraints {
    /// `fields[]` — each entry asserts the existence (and optionally
    /// the value) of a claim at a JSONPath.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<Field>,
    /// `limit_disclosure` (`required` or `preferred`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit_disclosure: Option<String>,
}

/// PE `field` constraint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Field {
    /// JSONPath expressions to inspect (any-of).
    pub path: Vec<String>,
    /// Optional id used by the submission.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Optional human-readable purpose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
    /// Optional JSON-Schema filter to apply to the resolved value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Value>,
    /// Whether the field is optional (default false).
    #[serde(default, skip_serializing_if = "is_false")]
    pub optional: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Top-level Presentation Definition (PE 2.0 §4).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresentationDefinition {
    /// Definition id.
    pub id: String,
    /// Optional human-readable name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional human-readable purpose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
    /// Format constraints at the top level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    /// Input descriptors.
    pub input_descriptors: Vec<InputDescriptor>,
    /// Optional submission requirements.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub submission_requirements: Vec<Value>,
}

impl PresentationDefinition {
    /// Validate that the definition has at least one input descriptor
    /// and that descriptor ids are unique.
    pub fn validate(&self) -> Result<(), OidcError> {
        if self.id.is_empty() {
            return Err(OidcError::Presentation("id required".into()));
        }
        if self.input_descriptors.is_empty() {
            return Err(OidcError::Presentation(
                "at least one input_descriptor required".into(),
            ));
        }
        let mut ids = std::collections::HashSet::new();
        for d in &self.input_descriptors {
            if !ids.insert(&d.id) {
                return Err(OidcError::Presentation(format!(
                    "duplicate input_descriptor id: {}",
                    d.id
                )));
            }
        }
        Ok(())
    }
}

/// One descriptor inside a submission.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmissionDescriptor {
    /// Matches the corresponding `InputDescriptor.id`.
    pub id: String,
    /// Credential format (`jwt_vp`, `ldp_vp`, `vc+sd-jwt`, `mso_mdoc`).
    pub format: String,
    /// JSONPath into the `vp_token` payload where the credential
    /// resides.
    pub path: String,
    /// Optional nested descriptor (when `path` resolves to a
    /// `vp_token` array element that itself wraps a credential).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_nested: Option<Box<SubmissionDescriptor>>,
}

/// PE 2.0 Submission body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresentationSubmission {
    /// Submission id (typically a UUID).
    pub id: String,
    /// Matches the requested `presentation_definition.id`.
    pub definition_id: String,
    /// One descriptor per input the wallet is fulfilling.
    pub descriptor_map: Vec<SubmissionDescriptor>,
}

impl PresentationSubmission {
    /// Validate that every descriptor in this submission maps to an
    /// input descriptor in the supplied definition.
    pub fn validate_against(
        &self,
        def: &PresentationDefinition,
    ) -> Result<(), OidcError> {
        if self.definition_id != def.id {
            return Err(OidcError::Presentation(format!(
                "definition_id mismatch: {} != {}",
                self.definition_id, def.id
            )));
        }
        for sd in &self.descriptor_map {
            if !def.input_descriptors.iter().any(|d| d.id == sd.id) {
                return Err(OidcError::Presentation(format!(
                    "submission descriptor references unknown input id: {}",
                    sd.id
                )));
            }
        }
        Ok(())
    }
}

/// Build a minimal presentation definition with one input descriptor
/// that asserts existence at the given JSONPath.
pub fn simple_definition(
    id: impl Into<String>,
    descriptor_id: impl Into<String>,
    path: impl Into<String>,
) -> PresentationDefinition {
    PresentationDefinition {
        id: id.into(),
        name: None,
        purpose: None,
        format: None,
        input_descriptors: vec![InputDescriptor {
            id: descriptor_id.into(),
            name: None,
            purpose: None,
            format: None,
            constraints: Some(Constraints {
                fields: vec![Field {
                    path: vec![path.into()],
                    id: None,
                    purpose: None,
                    filter: None,
                    optional: false,
                }],
                limit_disclosure: Some("required".into()),
            }),
        }],
        submission_requirements: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_validates() {
        let def = simple_definition(
            "pdef-1",
            "id-degree",
            "$.credentialSubject.degree",
        );
        def.validate().unwrap();
    }

    #[test]
    fn duplicate_ids_rejected() {
        let mut def = simple_definition(
            "pdef-1",
            "id-degree",
            "$.credentialSubject.degree",
        );
        def.input_descriptors
            .push(def.input_descriptors[0].clone());
        assert!(def.validate().is_err());
    }

    #[test]
    fn submission_validates_against_definition() {
        let def = simple_definition(
            "pdef-1",
            "id-degree",
            "$.credentialSubject.degree",
        );
        let sub = PresentationSubmission {
            id: "sub-1".into(),
            definition_id: "pdef-1".into(),
            descriptor_map: vec![SubmissionDescriptor {
                id: "id-degree".into(),
                format: "jwt_vp".into(),
                path: "$.verifiableCredential[0]".into(),
                path_nested: None,
            }],
        };
        sub.validate_against(&def).unwrap();
    }

    #[test]
    fn submission_unknown_descriptor_errors() {
        let def = simple_definition(
            "pdef-1",
            "id-degree",
            "$.credentialSubject.degree",
        );
        let sub = PresentationSubmission {
            id: "sub-1".into(),
            definition_id: "pdef-1".into(),
            descriptor_map: vec![SubmissionDescriptor {
                id: "missing".into(),
                format: "jwt_vp".into(),
                path: "$.x".into(),
                path_nested: None,
            }],
        };
        assert!(sub.validate_against(&def).is_err());
    }
}
