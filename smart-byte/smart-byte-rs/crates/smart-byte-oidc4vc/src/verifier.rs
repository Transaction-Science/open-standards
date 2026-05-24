//! Verifier — the OID4VP "Verifier" (Relying Party) side.
//!
//! A verifier builds an authorization request that the wallet consumes
//! and returns either:
//!
//! * `dcql_query` + `vp_token` (DCQL flow, OID4VP draft 23 default), or
//! * `presentation_definition` + `presentation_submission` + `vp_token`
//!   (PE 2.0 legacy flow).
//!
//! The authorization request itself can be transported as a URL deep
//! link (`openid4vp://`) or as a JWT-Secured Authorization Request
//! (JAR, RFC 9101) referenced via `request_uri`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::dcql::DcqlQuery;
use crate::error::OidcError;
use crate::presentation::{PresentationDefinition, PresentationSubmission};

/// `response_mode` values commonly used with OID4VP.
pub const RESPONSE_MODE_DIRECT_POST: &str = "direct_post";
/// `direct_post.jwt` — direct post with response in a JWE.
pub const RESPONSE_MODE_DIRECT_POST_JWT: &str = "direct_post.jwt";
/// `fragment` — for `openid4vp://` redirect URIs.
pub const RESPONSE_MODE_FRAGMENT: &str = "fragment";

/// `client_id_scheme` values from OID4VP draft 23.
pub const CLIENT_ID_SCHEME_DID: &str = "did";
/// `redirect_uri` client_id scheme.
pub const CLIENT_ID_SCHEME_REDIRECT_URI: &str = "redirect_uri";
/// `x509_san_dns` client_id scheme.
pub const CLIENT_ID_SCHEME_X509_SAN_DNS: &str = "x509_san_dns";
/// `verifier_attestation` client_id scheme.
pub const CLIENT_ID_SCHEME_VERIFIER_ATTESTATION: &str =
    "verifier_attestation";

/// OID4VP authorization request body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VpAuthRequest {
    /// `response_type`: `vp_token` or `vp_token id_token`.
    pub response_type: String,
    /// `client_id`.
    pub client_id: String,
    /// `client_id_scheme` (draft 23).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id_scheme: Option<String>,
    /// `response_uri` (where the wallet posts the response, with
    /// `response_mode = direct_post[.jwt]`) or `redirect_uri`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_uri: Option<String>,
    /// Legacy / fragment-flow redirect URI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redirect_uri: Option<String>,
    /// `response_mode`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_mode: Option<String>,
    /// `nonce` — required, echoed by the wallet in the `vp_token`.
    pub nonce: String,
    /// `state`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    /// DCQL query body (draft 23).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dcql_query: Option<DcqlQuery>,
    /// PE 2.0 presentation definition (legacy).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation_definition: Option<PresentationDefinition>,
    /// `presentation_definition_uri` (by-reference variant).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation_definition_uri: Option<String>,
}

impl VpAuthRequest {
    /// Validate basic shape: exactly one of `dcql_query` /
    /// `presentation_definition` / `presentation_definition_uri` MUST
    /// be present, plus `nonce` MUST be non-empty.
    pub fn validate(&self) -> Result<(), OidcError> {
        if self.nonce.is_empty() {
            return Err(OidcError::Presentation("nonce required".into()));
        }
        let mut count = 0;
        if self.dcql_query.is_some() {
            count += 1;
        }
        if self.presentation_definition.is_some() {
            count += 1;
        }
        if self.presentation_definition_uri.is_some() {
            count += 1;
        }
        if count != 1 {
            return Err(OidcError::Presentation(format!(
                "exactly one of dcql_query / presentation_definition / \
                 presentation_definition_uri required (got {count})"
            )));
        }
        if let Some(q) = &self.dcql_query {
            q.validate()?;
        }
        if let Some(p) = &self.presentation_definition {
            p.validate()?;
        }
        Ok(())
    }
}

/// OID4VP authorization response body. Returned by the wallet in the
/// chosen `response_mode`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VpAuthResponse {
    /// `vp_token` — a single VP string or a JSON object/array.
    pub vp_token: Value,
    /// `presentation_submission` — PE 2.0 only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation_submission: Option<PresentationSubmission>,
    /// `state` (echo).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
}

/// In-memory verifier. Holds the request the verifier issued so that
/// incoming wallet responses can be validated.
#[derive(Clone, Debug)]
pub struct Verifier {
    /// The request the verifier sent.
    pub request: VpAuthRequest,
}

impl Verifier {
    /// Construct, validating the request.
    pub fn new(request: VpAuthRequest) -> Result<Self, OidcError> {
        request.validate()?;
        Ok(Self { request })
    }

    /// Build a DCQL-mode verifier from a [`DcqlQuery`] and minimal
    /// request parameters.
    pub fn from_dcql(
        client_id: impl Into<String>,
        nonce: impl Into<String>,
        response_uri: impl Into<String>,
        query: DcqlQuery,
    ) -> Result<Self, OidcError> {
        let req = VpAuthRequest {
            response_type: "vp_token".into(),
            client_id: client_id.into(),
            client_id_scheme: Some(CLIENT_ID_SCHEME_REDIRECT_URI.into()),
            response_uri: Some(response_uri.into()),
            redirect_uri: None,
            response_mode: Some(RESPONSE_MODE_DIRECT_POST.into()),
            nonce: nonce.into(),
            state: None,
            dcql_query: Some(query),
            presentation_definition: None,
            presentation_definition_uri: None,
        };
        Self::new(req)
    }

    /// Build a PE-2.0 verifier from a definition.
    pub fn from_presentation_definition(
        client_id: impl Into<String>,
        nonce: impl Into<String>,
        response_uri: impl Into<String>,
        def: PresentationDefinition,
    ) -> Result<Self, OidcError> {
        let req = VpAuthRequest {
            response_type: "vp_token".into(),
            client_id: client_id.into(),
            client_id_scheme: Some(CLIENT_ID_SCHEME_REDIRECT_URI.into()),
            response_uri: Some(response_uri.into()),
            redirect_uri: None,
            response_mode: Some(RESPONSE_MODE_DIRECT_POST.into()),
            nonce: nonce.into(),
            state: None,
            dcql_query: None,
            presentation_definition: Some(def),
            presentation_definition_uri: None,
        };
        Self::new(req)
    }

    /// Validate a wallet response shape against the issued request.
    /// Cryptographic verification of the VP token itself is delegated
    /// to [`smart_byte_vc`].
    pub fn validate_response(
        &self,
        response: &VpAuthResponse,
    ) -> Result<(), OidcError> {
        // PE 2.0 responses MUST include a presentation_submission
        // matching the request's definition.
        if let Some(def) = &self.request.presentation_definition {
            let sub = response.presentation_submission.as_ref().ok_or_else(
                || {
                    OidcError::Presentation(
                        "presentation_submission required for PE 2.0 flow"
                            .into(),
                    )
                },
            )?;
            sub.validate_against(def)?;
        }
        // DCQL responses MUST NOT carry presentation_submission.
        if self.request.dcql_query.is_some()
            && response.presentation_submission.is_some()
        {
            return Err(OidcError::Presentation(
                "DCQL response must not include presentation_submission".into(),
            ));
        }
        // `state` echo (if requested) must round-trip.
        if let Some(expected) = &self.request.state {
            match &response.state {
                Some(got) if got == expected => {}
                _ => {
                    return Err(OidcError::Presentation(
                        "state echo missing or mismatched".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dcql::{
        ClaimQuery, CredentialMeta, CredentialQuery, DcqlQuery, PathSegment,
    };
    use crate::presentation::{simple_definition, SubmissionDescriptor};

    fn dcql_query() -> DcqlQuery {
        DcqlQuery {
            credentials: vec![CredentialQuery {
                id: "deg".into(),
                format: "vc+sd-jwt".into(),
                require: true,
                meta: CredentialMeta::default(),
                claims: vec![ClaimQuery::path(vec![PathSegment::Key(
                    "vct".into(),
                )])],
                claim_sets: vec![],
            }],
            credential_sets: vec![],
        }
    }

    #[test]
    fn dcql_verifier_validates() {
        let v = Verifier::from_dcql(
            "https://rp.example.com",
            "n-1",
            "https://rp.example.com/cb",
            dcql_query(),
        )
        .unwrap();
        assert!(v.request.dcql_query.is_some());
    }

    #[test]
    fn pe_verifier_validates() {
        let def = simple_definition("p1", "d1", "$.x");
        let v = Verifier::from_presentation_definition(
            "https://rp.example.com",
            "n-1",
            "https://rp.example.com/cb",
            def,
        )
        .unwrap();
        assert!(v.request.presentation_definition.is_some());
    }

    #[test]
    fn rejects_both_query_modes() {
        let req = VpAuthRequest {
            response_type: "vp_token".into(),
            client_id: "rp".into(),
            client_id_scheme: None,
            response_uri: None,
            redirect_uri: None,
            response_mode: None,
            nonce: "n".into(),
            state: None,
            dcql_query: Some(dcql_query()),
            presentation_definition: Some(simple_definition(
                "p1", "d1", "$.x",
            )),
            presentation_definition_uri: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn dcql_response_rejects_submission() {
        let v = Verifier::from_dcql(
            "rp",
            "n",
            "https://rp.example.com/cb",
            dcql_query(),
        )
        .unwrap();
        let resp = VpAuthResponse {
            vp_token: Value::from("ey.x"),
            presentation_submission: Some(PresentationSubmission {
                id: "s".into(),
                definition_id: "p1".into(),
                descriptor_map: vec![],
            }),
            state: None,
        };
        assert!(v.validate_response(&resp).is_err());
    }

    #[test]
    fn pe_response_requires_submission() {
        let def = simple_definition("p1", "d1", "$.x");
        let v = Verifier::from_presentation_definition(
            "rp",
            "n",
            "https://rp.example.com/cb",
            def,
        )
        .unwrap();
        let bad = VpAuthResponse {
            vp_token: Value::from("ey.x"),
            presentation_submission: None,
            state: None,
        };
        assert!(v.validate_response(&bad).is_err());
        let good_sub = PresentationSubmission {
            id: "s".into(),
            definition_id: "p1".into(),
            descriptor_map: vec![SubmissionDescriptor {
                id: "d1".into(),
                format: "jwt_vp".into(),
                path: "$".into(),
                path_nested: None,
            }],
        };
        let good = VpAuthResponse {
            vp_token: Value::from("ey.x"),
            presentation_submission: Some(good_sub),
            state: None,
        };
        v.validate_response(&good).unwrap();
    }
}
