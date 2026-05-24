//! Credential Offer (OID4VCI draft 13 §4).
//!
//! A *credential offer* is a JSON object that the issuer hands to the
//! wallet out-of-band (QR code, deep link, …). It points the wallet at
//! a [`crate::issuer::IssuerMetadata`] and contains one of two grants:
//!
//! * `urn:ietf:params:oauth:grant-type:pre-authorized_code` — the
//!   wallet calls the token endpoint directly with the embedded
//!   `pre-authorized_code` plus an optional `tx_code`.
//! * `authorization_code` — the wallet performs a normal browser
//!   authorisation-code flow rooted at the issuer's AS.
//!
//! The wire form is a JSON document; deployments commonly transport it
//! as `openid-credential-offer://?credential_offer=<urlencoded JSON>`
//! or `?credential_offer_uri=<URL>`. Both forms are represented here.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::OidcError;

/// Pre-authorized-code grant body inside a credential offer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreAuthorizedGrant {
    /// The pre-authorized code itself.
    #[serde(rename = "pre-authorized_code")]
    pub pre_authorized_code: String,
    /// Optional transaction-code requirement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tx_code: Option<TxCodeSpec>,
    /// Optional interval (seconds) the wallet should poll for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<u32>,
    /// Optional authorization-server identifier (URL).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization_server: Option<String>,
}

/// Specification of a transaction code (PIN-like) the wallet must
/// present at the token endpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TxCodeSpec {
    /// Input mode: `numeric` or `text`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_mode: Option<String>,
    /// Length in characters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<u32>,
    /// Human-readable description shown to the user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Authorisation-code grant body inside a credential offer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizationCodeGrant {
    /// Opaque issuer-state value to be echoed back on the AS request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub issuer_state: Option<String>,
    /// Optional authorization-server identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization_server: Option<String>,
}

/// Container for all grants offered. At least one MUST be present.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfferGrants {
    /// Pre-authorized-code grant, if offered.
    #[serde(
        default,
        rename = "urn:ietf:params:oauth:grant-type:pre-authorized_code",
        skip_serializing_if = "Option::is_none"
    )]
    pub pre_authorized: Option<PreAuthorizedGrant>,
    /// Authorization-code grant, if offered.
    #[serde(
        default,
        rename = "authorization_code",
        skip_serializing_if = "Option::is_none"
    )]
    pub authorization_code: Option<AuthorizationCodeGrant>,
}

/// `credential_offer` JSON document.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialOffer {
    /// The credential issuer identifier (URL).
    pub credential_issuer: String,
    /// List of credential configuration ids the issuer is offering.
    pub credential_configuration_ids: Vec<String>,
    /// Grants offered (at least one required).
    #[serde(default, skip_serializing_if = "is_default_grants")]
    pub grants: OfferGrants,
}

fn is_default_grants(g: &OfferGrants) -> bool {
    g.pre_authorized.is_none() && g.authorization_code.is_none()
}

impl CredentialOffer {
    /// Build a pre-authorized-code offer.
    pub fn pre_authorized(
        credential_issuer: impl Into<String>,
        configuration_ids: Vec<String>,
        pre_authorized_code: impl Into<String>,
    ) -> Self {
        Self {
            credential_issuer: credential_issuer.into(),
            credential_configuration_ids: configuration_ids,
            grants: OfferGrants {
                pre_authorized: Some(PreAuthorizedGrant {
                    pre_authorized_code: pre_authorized_code.into(),
                    tx_code: None,
                    interval: None,
                    authorization_server: None,
                }),
                authorization_code: None,
            },
        }
    }

    /// Build an authorisation-code offer.
    pub fn authorization_code(
        credential_issuer: impl Into<String>,
        configuration_ids: Vec<String>,
        issuer_state: Option<String>,
    ) -> Self {
        Self {
            credential_issuer: credential_issuer.into(),
            credential_configuration_ids: configuration_ids,
            grants: OfferGrants {
                pre_authorized: None,
                authorization_code: Some(AuthorizationCodeGrant {
                    issuer_state,
                    authorization_server: None,
                }),
            },
        }
    }

    /// Validate that at least one grant is present and URLs parse.
    pub fn validate(&self) -> Result<(), OidcError> {
        url::Url::parse(&self.credential_issuer)
            .map_err(|e| OidcError::Offer(e.to_string()))?;
        if self.credential_configuration_ids.is_empty() {
            return Err(OidcError::Offer(
                "credential_configuration_ids must be non-empty".into(),
            ));
        }
        if self.grants.pre_authorized.is_none()
            && self.grants.authorization_code.is_none()
        {
            return Err(OidcError::Offer("at least one grant required".into()));
        }
        Ok(())
    }

    /// Serialise to canonical JSON.
    pub fn to_json(&self) -> Result<String, OidcError> {
        serde_json::to_string(self).map_err(Into::into)
    }

    /// Encode as a `openid-credential-offer://` deep-link with the
    /// credential offer embedded as a `credential_offer` query
    /// parameter.
    pub fn to_deep_link(&self) -> Result<String, OidcError> {
        let json = self.to_json()?;
        // Build the URL by hand because percent-encoding the embedded
        // JSON is the canonical form per draft 13 §4.1.
        let encoded: String = byte_percent_encode(json.as_bytes());
        Ok(format!(
            "openid-credential-offer://?credential_offer={encoded}"
        ))
    }

    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self, OidcError> {
        let offer: CredentialOffer = serde_json::from_str(s)?;
        offer.validate()?;
        Ok(offer)
    }

    /// Parse from a deep link (`openid-credential-offer://?...`).
    pub fn from_deep_link(link: &str) -> Result<Self, OidcError> {
        let url = url::Url::parse(link)
            .map_err(|e| OidcError::Offer(e.to_string()))?;
        let params: HashMap<_, _> = url.query_pairs().into_owned().collect();
        if let Some(json) = params.get("credential_offer") {
            Self::from_json(json)
        } else if let Some(uri) = params.get("credential_offer_uri") {
            Err(OidcError::Offer(format!(
                "deep link is by-reference (credential_offer_uri={uri}); \
                 caller must fetch JSON"
            )))
        } else {
            Err(OidcError::Offer(
                "deep link missing credential_offer or credential_offer_uri"
                    .into(),
            ))
        }
    }
}

/// Minimal RFC 3986 percent-encoder for the unreserved set. We avoid
/// the `percent-encoding` crate to keep dependencies tight.
fn byte_percent_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 3);
    for &b in bytes {
        let unreserved = b.is_ascii_alphanumeric()
            || matches!(b, b'-' | b'_' | b'.' | b'~');
        if unreserved {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(hex_digit(b >> 4));
            out.push(hex_digit(b & 0x0f));
        }
    }
    out
}

fn hex_digit(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'A' + (n - 10)) as char,
        _ => '0',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_auth_offer_roundtrip() {
        let offer = CredentialOffer::pre_authorized(
            "https://issuer.example.com",
            vec!["UniversityDegree_SD_JWT".into()],
            "SplxlOBeZQQYbYS6WxSbIA",
        );
        let json = offer.to_json().unwrap();
        let back = CredentialOffer::from_json(&json).unwrap();
        assert_eq!(back, offer);
        assert!(back.grants.pre_authorized.is_some());
    }

    #[test]
    fn auth_code_offer_roundtrip() {
        let offer = CredentialOffer::authorization_code(
            "https://issuer.example.com",
            vec!["UniversityDegree_SD_JWT".into()],
            Some("state-abc".into()),
        );
        let json = offer.to_json().unwrap();
        let back = CredentialOffer::from_json(&json).unwrap();
        assert_eq!(back, offer);
    }

    #[test]
    fn rejects_empty_configurations() {
        let offer = CredentialOffer {
            credential_issuer: "https://issuer.example.com".into(),
            credential_configuration_ids: vec![],
            grants: OfferGrants::default(),
        };
        assert!(offer.validate().is_err());
    }

    #[test]
    fn rejects_no_grants() {
        let offer = CredentialOffer {
            credential_issuer: "https://issuer.example.com".into(),
            credential_configuration_ids: vec!["c1".into()],
            grants: OfferGrants::default(),
        };
        assert!(offer.validate().is_err());
    }

    #[test]
    fn deep_link_roundtrip() {
        let offer = CredentialOffer::pre_authorized(
            "https://issuer.example.com",
            vec!["UniversityDegree_SD_JWT".into()],
            "code-1",
        );
        let link = offer.to_deep_link().unwrap();
        assert!(link.starts_with("openid-credential-offer://"));
        let back = CredentialOffer::from_deep_link(&link).unwrap();
        assert_eq!(back, offer);
    }

    #[test]
    fn deep_link_by_reference_errors() {
        let link = "openid-credential-offer://?credential_offer_uri=\
                    https%3A%2F%2Fissuer.example.com%2Foffer%2F123";
        let err = CredentialOffer::from_deep_link(link).unwrap_err();
        match err {
            OidcError::Offer(_) => {}
            _ => panic!("expected Offer error"),
        }
    }
}
