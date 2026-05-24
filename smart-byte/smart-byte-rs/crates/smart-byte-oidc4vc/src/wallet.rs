//! Wallet client — consumes a [`CredentialOffer`] and walks the
//! issuance flow.
//!
//! The wallet's responsibilities, per OID4VCI draft 13:
//!
//! 1. Parse a [`CredentialOffer`] from a deep link or QR code.
//! 2. Resolve issuer metadata + authorisation-server metadata.
//! 3. Exchange the offer for an access token at the token endpoint
//!    ([`crate::token::TokenRequest`]).
//! 4. Build a proof-of-possession JWT that binds the credential to a
//!    wallet-controlled key (the `c_nonce` returned by the token
//!    endpoint is included in the `nonce` claim).
//! 5. Call the credential endpoint with the proof
//!    ([`crate::credential::CredentialRequest`]).
//! 6. Optionally call the notification endpoint
//!    ([`crate::notification::NotificationRequest`]).
//!
//! This module is the **stateful client** that ties those steps
//! together. It is intentionally transport-agnostic — the caller wires
//! the HTTP layer.

use chrono::{DateTime, Utc};

use crate::credential::{CredentialRequest, credential_request_jwt};
use crate::error::OidcError;
use crate::issuer::{AuthorizationServerMetadata, IssuerMetadata};
use crate::notification::{NotificationEvent, NotificationRequest};
use crate::offer::CredentialOffer;
use crate::token::{DpopClaims, TokenRequest, TokenResponse};

/// Stateful wallet client.
#[derive(Clone, Debug)]
pub struct WalletClient {
    /// Resolved credential offer.
    pub offer: CredentialOffer,
    /// Resolved issuer metadata.
    pub issuer_metadata: IssuerMetadata,
    /// Resolved authorisation-server metadata.
    pub as_metadata: AuthorizationServerMetadata,
    /// Issued access token (set after [`Self::record_token_response`]).
    pub access_token: Option<String>,
    /// Token type (`Bearer` or `DPoP`).
    pub token_type: Option<String>,
    /// Latest `c_nonce` for credential-endpoint proofs.
    pub c_nonce: Option<String>,
}

impl WalletClient {
    /// Construct from an offer + resolved metadata.
    pub fn new(
        offer: CredentialOffer,
        issuer_metadata: IssuerMetadata,
        as_metadata: AuthorizationServerMetadata,
    ) -> Result<Self, OidcError> {
        offer.validate()?;
        issuer_metadata.validate()?;
        as_metadata.validate()?;
        if offer.credential_issuer != issuer_metadata.credential_issuer {
            return Err(OidcError::Offer(format!(
                "offer credential_issuer {} != metadata credential_issuer {}",
                offer.credential_issuer, issuer_metadata.credential_issuer
            )));
        }
        Ok(Self {
            offer,
            issuer_metadata,
            as_metadata,
            access_token: None,
            token_type: None,
            c_nonce: None,
        })
    }

    /// Build the token request from the offer's pre-authorized grant.
    pub fn build_pre_authorized_token_request(
        &self,
        tx_code: Option<String>,
    ) -> Result<TokenRequest, OidcError> {
        let grant = self.offer.grants.pre_authorized.as_ref().ok_or_else(
            || OidcError::Offer("offer has no pre-authorized grant".into()),
        )?;
        if !self.as_metadata.supports_pre_authorized() {
            return Err(OidcError::AsMetadata(
                "AS does not advertise pre-authorized-code grant".into(),
            ));
        }
        Ok(TokenRequest::pre_authorized(
            grant.pre_authorized_code.clone(),
            tx_code,
        ))
    }

    /// Build a DPoP claim set for a request to the token endpoint.
    pub fn build_token_dpop(
        &self,
        now: DateTime<Utc>,
        jti: impl Into<String>,
    ) -> DpopClaims {
        DpopClaims::new("POST", &self.as_metadata.token_endpoint, now, jti)
    }

    /// Build a DPoP claim set for a credential-endpoint request.
    pub fn build_credential_dpop(
        &self,
        now: DateTime<Utc>,
        jti: impl Into<String>,
    ) -> Result<DpopClaims, OidcError> {
        let at = self
            .access_token
            .as_deref()
            .ok_or_else(|| OidcError::Dpop("no access_token recorded".into()))?;
        Ok(DpopClaims::new(
            "POST",
            &self.issuer_metadata.credential_endpoint,
            now,
            jti,
        )
        .with_access_token(at))
    }

    /// Record the success response from the token endpoint.
    pub fn record_token_response(&mut self, resp: &TokenResponse) {
        self.access_token = Some(resp.access_token.clone());
        self.token_type = Some(resp.token_type.clone());
        self.c_nonce = resp.c_nonce.clone();
    }

    /// Build a credential request for the *first* configuration in the
    /// offer using a JWT proof. The JWT itself is provided by the caller
    /// (it depends on the wallet's key material and signing layer).
    pub fn build_credential_request(
        &self,
        proof_jwt: impl Into<String>,
    ) -> Result<CredentialRequest, OidcError> {
        let cfg_id = self
            .offer
            .credential_configuration_ids
            .first()
            .ok_or_else(|| {
                OidcError::Offer(
                    "offer has no credential_configuration_ids".into(),
                )
            })?;
        let cfg = self
            .issuer_metadata
            .credential_configurations_supported
            .get(cfg_id)
            .ok_or_else(|| {
                OidcError::IssuerMetadata(format!(
                    "issuer metadata missing configuration {cfg_id}"
                ))
            })?;
        let _ = cfg; // configuration known; placeholder for future fmt-aware proofs.
        Ok(credential_request_jwt(cfg_id, proof_jwt))
    }

    /// Build a `credential_accepted` notification.
    pub fn build_notification(
        &self,
        notification_id: impl Into<String>,
        event: NotificationEvent,
    ) -> NotificationRequest {
        NotificationRequest::new(notification_id, event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::issuer::{
        AuthorizationServerMetadata, CredentialConfiguration, IssuerMetadata,
    };
    use std::collections::HashMap;

    fn metadata() -> IssuerMetadata {
        let mut configs = HashMap::new();
        configs.insert(
            "UD".to_string(),
            CredentialConfiguration {
                format: "vc+sd-jwt".into(),
                scope: None,
                credential_signing_alg_values_supported: vec!["EdDSA".into()],
                proof_types_supported: HashMap::new(),
                vct: None,
                doctype: None,
                credential_definition: None,
            },
        );
        IssuerMetadata {
            credential_issuer: "https://issuer.example.com".into(),
            authorization_servers: vec!["https://issuer.example.com".into()],
            credential_endpoint: "https://issuer.example.com/credential".into(),
            nonce_endpoint: None,
            notification_endpoint: Some(
                "https://issuer.example.com/notify".into(),
            ),
            deferred_credential_endpoint: None,
            batch_credential_endpoint: None,
            credential_configurations_supported: configs,
            display: vec![],
        }
    }

    fn as_meta() -> AuthorizationServerMetadata {
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

    fn offer() -> CredentialOffer {
        CredentialOffer::pre_authorized(
            "https://issuer.example.com",
            vec!["UD".into()],
            "code-1",
        )
    }

    #[test]
    fn pre_auth_request_built() {
        let w = WalletClient::new(offer(), metadata(), as_meta()).unwrap();
        let req = w.build_pre_authorized_token_request(None).unwrap();
        assert_eq!(req.pre_authorized_code.as_deref(), Some("code-1"));
    }

    #[test]
    fn credential_request_built() {
        let w = WalletClient::new(offer(), metadata(), as_meta()).unwrap();
        let req = w.build_credential_request("ey.proof.sig").unwrap();
        assert_eq!(
            req.credential_configuration_id.as_deref(),
            Some("UD")
        );
    }

    #[test]
    fn credential_dpop_requires_access_token() {
        let w = WalletClient::new(offer(), metadata(), as_meta()).unwrap();
        let now = Utc::now();
        assert!(w.build_credential_dpop(now, "j").is_err());
    }

    #[test]
    fn issuer_url_mismatch_rejected() {
        let mut o = offer();
        o.credential_issuer = "https://other.example.com".into();
        assert!(WalletClient::new(o, metadata(), as_meta()).is_err());
    }
}
