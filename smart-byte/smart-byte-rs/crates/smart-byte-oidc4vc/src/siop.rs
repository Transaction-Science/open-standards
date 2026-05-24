//! Self-Issued OpenID Provider v2 (SIOPv2) — OpenID Foundation draft 13.
//!
//! SIOPv2 lets a wallet act as its own OpenID Provider, signing an
//! `id_token` that authenticates the holder to a relying party without
//! a third-party IdP. The relying party publishes a
//! [`SiopAuthRequest`] indicating it wants `openid` scope and a
//! `subject_syntax_type` (typically `did:key` or `did:jwk`); the
//! wallet returns a [`SiopIdToken`] whose `sub` is the holder DID and
//! whose `sub_jwk` is the holder's verification key.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::OidcError;

/// SIOPv2 supported `subject_syntax_types_supported` values.
pub const SUBJECT_SYNTAX_DID_KEY: &str = "did:key";
/// `did:jwk` subject syntax.
pub const SUBJECT_SYNTAX_DID_JWK: &str = "did:jwk";
/// `urn:ietf:params:oauth:jwk-thumbprint` (per OIDF SIOPv2 draft).
pub const SUBJECT_SYNTAX_JWK_THUMBPRINT: &str =
    "urn:ietf:params:oauth:jwk-thumbprint";

/// Self-Issued authorization request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SiopAuthRequest {
    /// Always `id_token` for SIOPv2.
    pub response_type: String,
    /// RP `client_id`.
    pub client_id: String,
    /// Redirect URI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redirect_uri: Option<String>,
    /// `openid` (required) plus optional additional scopes.
    pub scope: String,
    /// Required random nonce echoed back in the id_token.
    pub nonce: String,
    /// Acceptable subject syntax types.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subject_syntax_types_supported: Vec<String>,
    /// Required `id_token_signing_alg_values_supported`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub id_token_signing_alg_values_supported: Vec<String>,
    /// Optional `state`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    /// Optional `response_mode`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_mode: Option<String>,
}

impl SiopAuthRequest {
    /// Sanity-check `response_type` is `id_token` and `scope` contains
    /// `openid`.
    pub fn validate(&self) -> Result<(), OidcError> {
        if self.response_type != "id_token" {
            return Err(OidcError::Siop(format!(
                "response_type must be id_token, got {}",
                self.response_type
            )));
        }
        if !self.scope.split_whitespace().any(|s| s == "openid") {
            return Err(OidcError::Siop("scope must include openid".into()));
        }
        if self.nonce.is_empty() {
            return Err(OidcError::Siop("nonce is required".into()));
        }
        Ok(())
    }
}

/// SIOPv2 ID token claim set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SiopIdToken {
    /// `iss` — fixed for SIOPv2: `https://self-issued.me/v2`.
    pub iss: String,
    /// `sub` — the holder DID (or JWK thumbprint).
    pub sub: String,
    /// `aud` — the RP's `client_id`.
    pub aud: String,
    /// `nonce` — echo of the request nonce.
    pub nonce: String,
    /// `iat` — issued at (Unix seconds).
    pub iat: i64,
    /// `exp` — expiry (Unix seconds).
    pub exp: i64,
    /// Optional `sub_jwk` — holder verification key as JWK.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sub_jwk: Option<Value>,
    /// Optional `state`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
}

/// Canonical SIOPv2 issuer identifier.
pub const SIOP_V2_ISS: &str = "https://self-issued.me/v2";

impl SiopIdToken {
    /// Build an ID token bound to a holder subject + RP client_id +
    /// nonce.
    pub fn new(
        sub: impl Into<String>,
        aud: impl Into<String>,
        nonce: impl Into<String>,
        iat: i64,
        ttl_seconds: i64,
    ) -> Self {
        Self {
            iss: SIOP_V2_ISS.into(),
            sub: sub.into(),
            aud: aud.into(),
            nonce: nonce.into(),
            iat,
            exp: iat + ttl_seconds,
            sub_jwk: None,
            state: None,
        }
    }

    /// Attach a `sub_jwk`.
    pub fn with_sub_jwk(mut self, jwk: Value) -> Self {
        self.sub_jwk = Some(jwk);
        self
    }

    /// Validate the ID token against an expected `aud` (client_id),
    /// expected `nonce`, and clock value.
    pub fn validate(
        &self,
        expected_aud: &str,
        expected_nonce: &str,
        now_unix: i64,
    ) -> Result<(), OidcError> {
        if self.iss != SIOP_V2_ISS {
            return Err(OidcError::Siop(format!(
                "iss must be {SIOP_V2_ISS}, got {}",
                self.iss
            )));
        }
        if self.aud != expected_aud {
            return Err(OidcError::Siop(format!(
                "aud mismatch: {} != {expected_aud}",
                self.aud
            )));
        }
        if self.nonce != expected_nonce {
            return Err(OidcError::Siop("nonce mismatch".into()));
        }
        if now_unix < self.iat {
            return Err(OidcError::Siop("token issued in the future".into()));
        }
        if now_unix > self.exp {
            return Err(OidcError::Siop("token expired".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> SiopAuthRequest {
        SiopAuthRequest {
            response_type: "id_token".into(),
            client_id: "https://rp.example.com".into(),
            redirect_uri: Some("https://rp.example.com/cb".into()),
            scope: "openid".into(),
            nonce: "n-1".into(),
            subject_syntax_types_supported: vec![
                SUBJECT_SYNTAX_DID_KEY.into(),
                SUBJECT_SYNTAX_DID_JWK.into(),
            ],
            id_token_signing_alg_values_supported: vec!["EdDSA".into()],
            state: None,
            response_mode: None,
        }
    }

    #[test]
    fn request_validates() {
        request().validate().unwrap();
    }

    #[test]
    fn rejects_non_idtoken_response_type() {
        let mut r = request();
        r.response_type = "code".into();
        assert!(r.validate().is_err());
    }

    #[test]
    fn rejects_missing_openid_scope() {
        let mut r = request();
        r.scope = "profile".into();
        assert!(r.validate().is_err());
    }

    #[test]
    fn idtoken_validates() {
        let token = SiopIdToken::new(
            "did:key:zABCDEF",
            "https://rp.example.com",
            "n-1",
            1_700_000_000,
            300,
        );
        token
            .validate("https://rp.example.com", "n-1", 1_700_000_100)
            .unwrap();
    }

    #[test]
    fn idtoken_rejects_aud_mismatch() {
        let token = SiopIdToken::new(
            "did:key:zABCDEF",
            "https://rp.example.com",
            "n-1",
            1_700_000_000,
            300,
        );
        assert!(
            token
                .validate("https://other.example.com", "n-1", 1_700_000_100)
                .is_err()
        );
    }

    #[test]
    fn idtoken_rejects_expired() {
        let token = SiopIdToken::new(
            "did:key:zABCDEF",
            "https://rp.example.com",
            "n-1",
            1_700_000_000,
            10,
        );
        assert!(
            token
                .validate("https://rp.example.com", "n-1", 1_700_001_000)
                .is_err()
        );
    }
}
