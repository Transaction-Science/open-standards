//! Credential Issuer + Authorization Server metadata (OID4VCI draft 13).
//!
//! Two metadata documents are advertised by an OID4VCI deployment:
//!
//! * **Credential Issuer Metadata** — served at
//!   `<credential_issuer>/.well-known/openid-credential-issuer`.
//!   Lists supported credential configurations (format, types,
//!   cryptosuites), the `credential_endpoint`, optional
//!   `notification_endpoint`, optional `nonce_endpoint`, and supported
//!   batch sizes.
//! * **Authorization Server Metadata** — served at
//!   `<authorization_server>/.well-known/oauth-authorization-server`.
//!   Lists the `token_endpoint`, supported grant types (including
//!   `urn:ietf:params:oauth:grant-type:pre-authorized_code`), and DPoP
//!   signing algorithms.
//!
//! This module models both in `serde`-portable form plus a
//! [`CredentialIssuer`] state object that maps configuration IDs to
//! issuance handlers.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::OidcError;

/// One entry in `credential_configurations_supported`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialConfiguration {
    /// Credential format: `vc+sd-jwt`, `mso_mdoc`, `jwt_vc_json`,
    /// `ldp_vc`, `dc+sd-jwt`, …
    pub format: String,
    /// Optional human-readable scope string used in `scope` requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Cryptosuite or `alg` values acceptable for issuer signing.
    #[serde(
        default,
        rename = "credential_signing_alg_values_supported",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub credential_signing_alg_values_supported: Vec<String>,
    /// Proof types the wallet may use to bind the credential to a key.
    #[serde(
        default,
        rename = "proof_types_supported",
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub proof_types_supported: HashMap<String, ProofTypeMetadata>,
    /// Optional `vct` (SD-JWT VC) value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vct: Option<String>,
    /// Optional `doctype` (mdoc).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doctype: Option<String>,
    /// Optional W3C VC type tags.
    #[serde(
        default,
        rename = "credential_definition",
        skip_serializing_if = "Option::is_none"
    )]
    pub credential_definition: Option<CredentialDefinition>,
}

/// Per-proof-type metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofTypeMetadata {
    /// Acceptable `alg` values (e.g. `EdDSA`, `ES256`).
    #[serde(default, rename = "proof_signing_alg_values_supported")]
    pub proof_signing_alg_values_supported: Vec<String>,
}

/// W3C `credential_definition` block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialDefinition {
    /// `@context` IRIs.
    #[serde(rename = "@context", default, skip_serializing_if = "Vec::is_empty")]
    pub context: Vec<String>,
    /// Type tags.
    #[serde(rename = "type", default, skip_serializing_if = "Vec::is_empty")]
    pub type_: Vec<String>,
}

/// `openid-credential-issuer` metadata document.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IssuerMetadata {
    /// Issuer identifier (`iss` / `credential_issuer`).
    pub credential_issuer: String,
    /// Authorisation server endpoints.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub authorization_servers: Vec<String>,
    /// `credential_endpoint`.
    pub credential_endpoint: String,
    /// Optional `nonce_endpoint` (OID4VCI 13 nonce endpoint).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nonce_endpoint: Option<String>,
    /// Optional `notification_endpoint`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notification_endpoint: Option<String>,
    /// Optional `deferred_credential_endpoint`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred_credential_endpoint: Option<String>,
    /// Optional batch endpoint (some drafts).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_credential_endpoint: Option<String>,
    /// Credential configurations supported, keyed by configuration id.
    #[serde(default)]
    pub credential_configurations_supported:
        HashMap<String, CredentialConfiguration>,
    /// Optional display metadata.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub display: Vec<DisplayInfo>,
}

/// Issuer display block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisplayInfo {
    /// Human-readable name.
    pub name: String,
    /// BCP-47 locale tag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locale: Option<String>,
}

impl IssuerMetadata {
    /// Validate that `credential_issuer` and `credential_endpoint` are
    /// present and well-formed URLs.
    pub fn validate(&self) -> Result<(), OidcError> {
        if self.credential_issuer.is_empty() {
            return Err(OidcError::IssuerMetadata(
                "credential_issuer required".into(),
            ));
        }
        url::Url::parse(&self.credential_issuer)
            .map_err(|e| OidcError::IssuerMetadata(e.to_string()))?;
        if self.credential_endpoint.is_empty() {
            return Err(OidcError::IssuerMetadata(
                "credential_endpoint required".into(),
            ));
        }
        url::Url::parse(&self.credential_endpoint)
            .map_err(|e| OidcError::IssuerMetadata(e.to_string()))?;
        Ok(())
    }
}

/// `oauth-authorization-server` metadata document (RFC 8414 + OID4VCI
/// 13 extensions).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizationServerMetadata {
    /// `issuer` (the AS identifier).
    pub issuer: String,
    /// `token_endpoint`.
    pub token_endpoint: String,
    /// Optional `authorization_endpoint` (only required for
    /// authorisation-code flow).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization_endpoint: Option<String>,
    /// Optional `pushed_authorization_request_endpoint` (PAR).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pushed_authorization_request_endpoint: Option<String>,
    /// Grant types supported, e.g. `authorization_code`,
    /// `urn:ietf:params:oauth:grant-type:pre-authorized_code`.
    #[serde(default)]
    pub grant_types_supported: Vec<String>,
    /// DPoP signing algorithms.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dpop_signing_alg_values_supported: Vec<String>,
    /// Code challenge methods (`S256`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub code_challenge_methods_supported: Vec<String>,
    /// OID4VCI 13 — whether tx_code is supported on pre-authorized flow.
    #[serde(default, rename = "pre-authorized_grant_anonymous_access_supported")]
    pub pre_authorized_grant_anonymous_access_supported: bool,
}

impl AuthorizationServerMetadata {
    /// Standard pre-authorized-code grant URI.
    pub const PREAUTH_GRANT: &'static str =
        "urn:ietf:params:oauth:grant-type:pre-authorized_code";

    /// Validate URL fields and required grants.
    pub fn validate(&self) -> Result<(), OidcError> {
        if self.issuer.is_empty() {
            return Err(OidcError::AsMetadata("issuer required".into()));
        }
        url::Url::parse(&self.issuer)
            .map_err(|e| OidcError::AsMetadata(e.to_string()))?;
        url::Url::parse(&self.token_endpoint)
            .map_err(|e| OidcError::AsMetadata(e.to_string()))?;
        Ok(())
    }

    /// True if this AS advertises the pre-authorized-code grant.
    pub fn supports_pre_authorized(&self) -> bool {
        self.grant_types_supported
            .iter()
            .any(|g| g == Self::PREAUTH_GRANT)
    }

    /// True if the AS advertises the authorization-code grant.
    pub fn supports_authorization_code(&self) -> bool {
        self.grant_types_supported
            .iter()
            .any(|g| g == "authorization_code")
    }
}

/// In-memory Credential Issuer state. Issuance handlers are keyed by
/// configuration id. The state object is intentionally protocol-only
/// — wiring it to a transport is the deployer's job.
#[derive(Clone, Debug)]
pub struct CredentialIssuer {
    /// Public metadata document.
    pub metadata: IssuerMetadata,
    /// Authorization server metadata for this issuer.
    pub as_metadata: AuthorizationServerMetadata,
}

impl CredentialIssuer {
    /// Construct, validating both metadata documents.
    pub fn new(
        metadata: IssuerMetadata,
        as_metadata: AuthorizationServerMetadata,
    ) -> Result<Self, OidcError> {
        metadata.validate()?;
        as_metadata.validate()?;
        Ok(Self {
            metadata,
            as_metadata,
        })
    }

    /// Look up a credential configuration by id.
    pub fn configuration(
        &self,
        id: &str,
    ) -> Result<&CredentialConfiguration, OidcError> {
        self.metadata
            .credential_configurations_supported
            .get(id)
            .ok_or_else(|| {
                OidcError::IssuerMetadata(format!(
                    "unknown credential_configuration_id: {id}"
                ))
            })
    }

    /// Canonical well-known metadata URL.
    pub fn well_known_metadata_url(&self) -> Result<String, OidcError> {
        let base = url::Url::parse(&self.metadata.credential_issuer)?;
        let joined = base.join(".well-known/openid-credential-issuer")?;
        Ok(joined.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> IssuerMetadata {
        let mut configs = HashMap::new();
        configs.insert(
            "UniversityDegree_SD_JWT".to_string(),
            CredentialConfiguration {
                format: "vc+sd-jwt".into(),
                scope: Some("UniversityDegree".into()),
                credential_signing_alg_values_supported: vec!["EdDSA".into()],
                proof_types_supported: HashMap::new(),
                vct: Some("https://example.com/UniversityDegree".into()),
                doctype: None,
                credential_definition: None,
            },
        );
        IssuerMetadata {
            credential_issuer: "https://issuer.example.com".into(),
            authorization_servers: vec!["https://issuer.example.com".into()],
            credential_endpoint: "https://issuer.example.com/credential".into(),
            nonce_endpoint: Some("https://issuer.example.com/nonce".into()),
            notification_endpoint: Some(
                "https://issuer.example.com/notify".into(),
            ),
            deferred_credential_endpoint: None,
            batch_credential_endpoint: None,
            credential_configurations_supported: configs,
            display: vec![DisplayInfo {
                name: "Example Issuer".into(),
                locale: Some("en-US".into()),
            }],
        }
    }

    fn sample_as_metadata() -> AuthorizationServerMetadata {
        AuthorizationServerMetadata {
            issuer: "https://issuer.example.com".into(),
            token_endpoint: "https://issuer.example.com/token".into(),
            authorization_endpoint: None,
            pushed_authorization_request_endpoint: None,
            grant_types_supported: vec![
                AuthorizationServerMetadata::PREAUTH_GRANT.into(),
            ],
            dpop_signing_alg_values_supported: vec!["EdDSA".into()],
            code_challenge_methods_supported: vec!["S256".into()],
            pre_authorized_grant_anonymous_access_supported: true,
        }
    }

    #[test]
    fn validates_metadata() {
        let issuer = CredentialIssuer::new(
            sample_metadata(),
            sample_as_metadata(),
        )
        .expect("metadata should validate");
        assert!(issuer.as_metadata.supports_pre_authorized());
        assert!(!issuer.as_metadata.supports_authorization_code());
    }

    #[test]
    fn rejects_invalid_url() {
        let mut m = sample_metadata();
        m.credential_issuer = "not a url".into();
        assert!(m.validate().is_err());
    }

    #[test]
    fn configuration_lookup() {
        let issuer = CredentialIssuer::new(
            sample_metadata(),
            sample_as_metadata(),
        )
        .unwrap();
        assert!(issuer.configuration("UniversityDegree_SD_JWT").is_ok());
        assert!(issuer.configuration("missing").is_err());
    }

    #[test]
    fn well_known_url() {
        let issuer = CredentialIssuer::new(
            sample_metadata(),
            sample_as_metadata(),
        )
        .unwrap();
        let url = issuer.well_known_metadata_url().unwrap();
        assert!(url.contains("/.well-known/openid-credential-issuer"));
    }
}
