//! Digital Credentials Query Language (DCQL) — new in OID4VP draft 23.
//!
//! DCQL replaces the older DIF Presentation Exchange (still supported
//! via [`crate::presentation`]) with a JSON query language designed for
//! W3C VCs, SD-JWT VCs, and ISO mdoc. A DCQL query consists of one or
//! more *credential queries*, each of which:
//!
//! 1. Selects a credential by format (`vc+sd-jwt`, `mso_mdoc`,
//!    `dc+sd-jwt`, …) and optional metadata constraints (`vct`,
//!    `doctype`, type tags).
//! 2. Lists one or more *claim queries*, each addressing a specific
//!    claim path; claim values may be constrained to an enumerated set.
//! 3. Optionally lists *claim set alternatives* (`claim_sets`) so the
//!    verifier can demand "either set A or set B".
//!
//! This module implements the wire form plus a small matcher that
//! evaluates a DCQL query against a JSON candidate (used by wallets to
//! pick credentials and by verifiers to validate submitted ones).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::OidcError;

/// One element of a claim path. The DCQL spec allows strings (object
/// keys), unsigned integers (array indices), and the JSON `null`
/// (wildcard / "all elements").
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PathSegment {
    /// Object key.
    Key(String),
    /// Array index.
    Index(u64),
    /// Wildcard for every element / property.
    Wildcard(Option<()>),
}

/// One claim query — a JSON pointer expressed as a list of
/// [`PathSegment`]s plus optional enumerated values.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimQuery {
    /// Optional claim identifier (used by `claim_sets`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Path: JSON-pointer-like list of segments rooted at the credential.
    pub path: Vec<PathSegment>,
    /// Optional enumerated values; the claim's actual value must match
    /// one of these.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<Value>,
}

impl ClaimQuery {
    /// Construct a path-only claim query (no value constraint).
    pub fn path<I: IntoIterator<Item = PathSegment>>(path: I) -> Self {
        Self {
            id: None,
            path: path.into_iter().collect(),
            values: Vec::new(),
        }
    }

    /// Attach a claim identifier.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Constrain to enumerated values.
    pub fn with_values<I: IntoIterator<Item = Value>>(mut self, vs: I) -> Self {
        self.values = vs.into_iter().collect();
        self
    }
}

/// Per-credential meta-constraints. Keys are union of all formats; only
/// the relevant subset is populated for any given query.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialMeta {
    /// SD-JWT VC type identifiers (any-of).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vct_values: Vec<String>,
    /// mdoc doctype (single value).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doctype_value: Option<String>,
    /// W3C VC type tag sets (any-of).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub type_values: Vec<Vec<String>>,
}

/// One credential query.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialQuery {
    /// Identifier used by `credential_sets` and by the submission.
    pub id: String,
    /// Credential format identifier.
    pub format: String,
    /// Whether the credential MUST be returned (default: `true`).
    #[serde(default = "default_true")]
    pub require: bool,
    /// Format-specific constraints.
    #[serde(default, skip_serializing_if = "is_default_meta")]
    pub meta: CredentialMeta,
    /// Claim queries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub claims: Vec<ClaimQuery>,
    /// Alternative `claim_sets`: each inner Vec is a list of claim ids;
    /// the credential matches if ALL claims in at least one set match.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub claim_sets: Vec<Vec<String>>,
}

fn default_true() -> bool {
    true
}

fn is_default_meta(m: &CredentialMeta) -> bool {
    m.vct_values.is_empty()
        && m.doctype_value.is_none()
        && m.type_values.is_empty()
}

/// `credential_sets` alternative inside a top-level DCQL query.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialSet {
    /// Lists of credential query ids; each inner Vec is one acceptable
    /// combination, the verifier accepts the submission if at least one
    /// combination is satisfied.
    pub options: Vec<Vec<String>>,
    /// Whether the set itself is required.
    #[serde(default = "default_true")]
    pub required: bool,
}

/// Top-level DCQL query (the body of `dcql_query` inside an OID4VP
/// authorization request).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DcqlQuery {
    /// One or more credential queries.
    pub credentials: Vec<CredentialQuery>,
    /// Optional credential sets (alternative combinations).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub credential_sets: Vec<CredentialSet>,
}

impl DcqlQuery {
    /// Sanity-check identifiers + non-emptiness.
    pub fn validate(&self) -> Result<(), OidcError> {
        if self.credentials.is_empty() {
            return Err(OidcError::Dcql("credentials must be non-empty".into()));
        }
        let mut ids = std::collections::HashSet::new();
        for c in &self.credentials {
            if !ids.insert(&c.id) {
                return Err(OidcError::Dcql(format!(
                    "duplicate credential id: {}",
                    c.id
                )));
            }
            for cs in &c.claim_sets {
                for cid in cs {
                    if !c.claims.iter().any(|cl| cl.id.as_deref() == Some(cid))
                    {
                        return Err(OidcError::Dcql(format!(
                            "claim_sets references unknown claim id: {cid}"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Evaluate a [`ClaimQuery`] against a JSON document. Returns `Some`
/// with the matched value (or first-matched element for wildcards), or
/// `None` if not found / if values constraint not satisfied.
pub fn evaluate_claim<'a>(
    doc: &'a Value,
    query: &ClaimQuery,
) -> Option<&'a Value> {
    let v = walk_path(doc, &query.path)?;
    if !query.values.is_empty() && !query.values.iter().any(|e| e == v) {
        return None;
    }
    Some(v)
}

fn walk_path<'a>(doc: &'a Value, path: &[PathSegment]) -> Option<&'a Value> {
    let mut current = doc;
    for seg in path {
        current = match (current, seg) {
            (Value::Object(m), PathSegment::Key(k)) => m.get(k)?,
            (Value::Array(a), PathSegment::Index(i)) => {
                a.get(*i as usize)?
            }
            (Value::Array(a), PathSegment::Wildcard(_)) => {
                // First-match wildcard.
                a.first()?
            }
            _ => return None,
        };
    }
    Some(current)
}

/// Evaluate a [`CredentialQuery`]'s claims against a candidate JSON
/// document. Returns the list of claim ids that matched. If
/// `claim_sets` is present, at least one set must be fully satisfied.
pub fn evaluate_credential(
    doc: &Value,
    query: &CredentialQuery,
) -> Result<Vec<String>, OidcError> {
    let mut matched = Vec::new();
    for claim in &query.claims {
        if evaluate_claim(doc, claim).is_some() {
            if let Some(id) = &claim.id {
                matched.push(id.clone());
            }
        } else if query.claim_sets.is_empty() {
            // No claim_sets ⇒ every claim is required.
            return Err(OidcError::Dcql(format!(
                "required claim did not match: {:?}",
                claim.path
            )));
        }
    }
    if !query.claim_sets.is_empty() {
        let any_satisfied = query.claim_sets.iter().any(|set| {
            set.iter().all(|cid| matched.iter().any(|m| m == cid))
        });
        if !any_satisfied {
            return Err(OidcError::Dcql(
                "no claim_set was fully satisfied".into(),
            ));
        }
    }
    Ok(matched)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iso_credential() -> Value {
        serde_json::json!({
            "vct": "https://example.com/UniversityDegree",
            "credentialSubject": {
                "id": "did:example:alice",
                "degree": {
                    "type": "BachelorDegree",
                    "name": "Bachelor of Science"
                }
            }
        })
    }

    #[test]
    fn path_walk_simple() {
        let cred = iso_credential();
        let q = ClaimQuery::path(vec![
            PathSegment::Key("credentialSubject".into()),
            PathSegment::Key("degree".into()),
            PathSegment::Key("name".into()),
        ]);
        let m = evaluate_claim(&cred, &q).unwrap();
        assert_eq!(m, &Value::from("Bachelor of Science"));
    }

    #[test]
    fn path_walk_missing() {
        let cred = iso_credential();
        let q = ClaimQuery::path(vec![PathSegment::Key("nope".into())]);
        assert!(evaluate_claim(&cred, &q).is_none());
    }

    #[test]
    fn values_constraint_matches() {
        let cred = iso_credential();
        let q = ClaimQuery::path(vec![PathSegment::Key("vct".into())])
            .with_values(vec![Value::from(
                "https://example.com/UniversityDegree",
            )]);
        assert!(evaluate_claim(&cred, &q).is_some());
    }

    #[test]
    fn values_constraint_rejects() {
        let cred = iso_credential();
        let q = ClaimQuery::path(vec![PathSegment::Key("vct".into())])
            .with_values(vec![Value::from("https://other.example/X")]);
        assert!(evaluate_claim(&cred, &q).is_none());
    }

    #[test]
    fn credential_query_all_claims_required() {
        let cred = iso_credential();
        let q = CredentialQuery {
            id: "deg".into(),
            format: "vc+sd-jwt".into(),
            require: true,
            meta: CredentialMeta::default(),
            claims: vec![
                ClaimQuery::path(vec![PathSegment::Key("vct".into())])
                    .with_id("c-vct"),
            ],
            claim_sets: vec![],
        };
        let matched = evaluate_credential(&cred, &q).unwrap();
        assert_eq!(matched, vec!["c-vct".to_string()]);
    }

    #[test]
    fn claim_sets_one_must_satisfy() {
        let cred = iso_credential();
        let q = CredentialQuery {
            id: "deg".into(),
            format: "vc+sd-jwt".into(),
            require: true,
            meta: CredentialMeta::default(),
            claims: vec![
                ClaimQuery::path(vec![PathSegment::Key("vct".into())])
                    .with_id("c-vct"),
                ClaimQuery::path(vec![PathSegment::Key("nope".into())])
                    .with_id("c-nope"),
            ],
            claim_sets: vec![
                vec!["c-vct".into()],
                vec!["c-nope".into()],
            ],
        };
        let matched = evaluate_credential(&cred, &q).unwrap();
        assert!(matched.contains(&"c-vct".to_string()));
    }

    #[test]
    fn claim_sets_none_satisfied_errors() {
        let cred = iso_credential();
        let q = CredentialQuery {
            id: "deg".into(),
            format: "vc+sd-jwt".into(),
            require: true,
            meta: CredentialMeta::default(),
            claims: vec![
                ClaimQuery::path(vec![PathSegment::Key("nope".into())])
                    .with_id("c-nope"),
            ],
            claim_sets: vec![vec!["c-nope".into()]],
        };
        assert!(evaluate_credential(&cred, &q).is_err());
    }

    #[test]
    fn validates_query_ids_unique() {
        let q = DcqlQuery {
            credentials: vec![
                CredentialQuery {
                    id: "a".into(),
                    format: "vc+sd-jwt".into(),
                    require: true,
                    meta: CredentialMeta::default(),
                    claims: vec![],
                    claim_sets: vec![],
                },
                CredentialQuery {
                    id: "a".into(),
                    format: "vc+sd-jwt".into(),
                    require: true,
                    meta: CredentialMeta::default(),
                    claims: vec![],
                    claim_sets: vec![],
                },
            ],
            credential_sets: vec![],
        };
        assert!(q.validate().is_err());
    }
}
