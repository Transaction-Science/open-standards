//! Token endpoint + DPoP (RFC 9449) sender-constrained tokens.
//!
//! The token endpoint is the OAuth 2.0 RFC 6749 endpoint with two
//! OID4VCI-specific extensions:
//!
//! * Support for the pre-authorized-code grant
//!   (`urn:ietf:params:oauth:grant-type:pre-authorized_code`) with an
//!   optional `tx_code` parameter.
//! * Issuance of `c_nonce` / `c_nonce_expires_in` in the success
//!   response so the credential endpoint can validate proof-of-possession.
//!
//! DPoP (RFC 9449) sender-constrains the access token to the client's
//! key by attaching a per-request JWT proof in the `DPoP` header. The
//! token endpoint binds the access token's `cnf.jkt` to the proof's
//! `jwk` thumbprint (RFC 7638).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::OidcError;

/// Common OAuth 2.0 token-error codes (RFC 6749 §5.2 plus OID4VCI
/// additions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenErrorCode {
    /// `invalid_request`.
    InvalidRequest,
    /// `invalid_grant`.
    InvalidGrant,
    /// `invalid_client`.
    InvalidClient,
    /// `unsupported_grant_type`.
    UnsupportedGrantType,
    /// `invalid_dpop_proof`.
    InvalidDpopProof,
    /// `use_dpop_nonce`.
    UseDpopNonce,
}

impl TokenErrorCode {
    /// Wire-form identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::InvalidGrant => "invalid_grant",
            Self::InvalidClient => "invalid_client",
            Self::UnsupportedGrantType => "unsupported_grant_type",
            Self::InvalidDpopProof => "invalid_dpop_proof",
            Self::UseDpopNonce => "use_dpop_nonce",
        }
    }
}

/// Token request body. Field set varies by `grant_type`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenRequest {
    /// `grant_type`.
    pub grant_type: String,
    /// `pre-authorized_code` (pre-auth grant only).
    #[serde(default, rename = "pre-authorized_code", skip_serializing_if = "Option::is_none")]
    pub pre_authorized_code: Option<String>,
    /// `tx_code` (pre-auth grant only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tx_code: Option<String>,
    /// `code` (authorisation-code grant).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// `redirect_uri` (authorisation-code grant).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redirect_uri: Option<String>,
    /// `code_verifier` (PKCE).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_verifier: Option<String>,
    /// `client_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
}

impl TokenRequest {
    /// Build a pre-authorized-code request.
    pub fn pre_authorized(
        code: impl Into<String>,
        tx_code: Option<String>,
    ) -> Self {
        Self {
            grant_type:
                "urn:ietf:params:oauth:grant-type:pre-authorized_code".into(),
            pre_authorized_code: Some(code.into()),
            tx_code,
            code: None,
            redirect_uri: None,
            code_verifier: None,
            client_id: None,
        }
    }

    /// Build an authorisation-code request.
    pub fn authorization_code(
        code: impl Into<String>,
        redirect_uri: impl Into<String>,
        code_verifier: impl Into<String>,
    ) -> Self {
        Self {
            grant_type: "authorization_code".into(),
            pre_authorized_code: None,
            tx_code: None,
            code: Some(code.into()),
            redirect_uri: Some(redirect_uri.into()),
            code_verifier: Some(code_verifier.into()),
            client_id: None,
        }
    }

    /// Wire-form `application/x-www-form-urlencoded`.
    pub fn to_form(&self) -> String {
        let mut pairs: Vec<(&str, String)> = Vec::new();
        pairs.push(("grant_type", self.grant_type.clone()));
        if let Some(c) = &self.pre_authorized_code {
            pairs.push(("pre-authorized_code", c.clone()));
        }
        if let Some(c) = &self.tx_code {
            pairs.push(("tx_code", c.clone()));
        }
        if let Some(c) = &self.code {
            pairs.push(("code", c.clone()));
        }
        if let Some(r) = &self.redirect_uri {
            pairs.push(("redirect_uri", r.clone()));
        }
        if let Some(v) = &self.code_verifier {
            pairs.push(("code_verifier", v.clone()));
        }
        if let Some(c) = &self.client_id {
            pairs.push(("client_id", c.clone()));
        }
        pairs
            .into_iter()
            .map(|(k, v)| format!("{k}={}", form_encode(&v)))
            .collect::<Vec<_>>()
            .join("&")
    }
}

fn form_encode(s: &str) -> String {
    // application/x-www-form-urlencoded: space → '+', everything else
    // RFC 3986 unreserved.
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        if b == b' ' {
            out.push('+');
        } else if b.is_ascii_alphanumeric()
            || matches!(b, b'-' | b'_' | b'.' | b'~')
        {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(hex(b >> 4));
            out.push(hex(b & 0x0f));
        }
    }
    out
}

fn hex(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'A' + n - 10) as char,
        _ => '0',
    }
}

/// Token endpoint success response (OAuth 2.0 §5.1 + OID4VCI 13).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenResponse {
    /// The access token (opaque).
    pub access_token: String,
    /// Token type. `Bearer` for non-DPoP, `DPoP` for sender-constrained.
    pub token_type: String,
    /// Lifetime in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_in: Option<u64>,
    /// Refresh token, if issued.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Scope, if scoped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Issuer-issued `c_nonce` for credential endpoint proof.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub c_nonce: Option<String>,
    /// `c_nonce` expiry in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub c_nonce_expires_in: Option<u64>,
    /// Optional authorization_details (OAuth 2.0 RFC 9396) echo.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization_details: Option<serde_json::Value>,
}

/// Token endpoint error response body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenErrorResponse {
    /// `error` code.
    pub error: String,
    /// Optional human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_description: Option<String>,
    /// Optional reference URI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_uri: Option<String>,
}

/// DPoP proof claims (RFC 9449 §4.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DpopClaims {
    /// `jti` — unique identifier.
    pub jti: String,
    /// `htm` — HTTP method (uppercase).
    pub htm: String,
    /// `htu` — HTTP target URI.
    pub htu: String,
    /// `iat` — issued at (Unix seconds).
    pub iat: i64,
    /// Optional `ath` — access-token hash (SHA-256, base64url-no-pad).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ath: Option<String>,
    /// Optional `nonce` — server-provided DPoP nonce.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nonce: Option<String>,
}

impl DpopClaims {
    /// Construct DPoP claims for an HTTP request.
    pub fn new(
        method: impl Into<String>,
        url: impl Into<String>,
        now: DateTime<Utc>,
        jti: impl Into<String>,
    ) -> Self {
        Self {
            jti: jti.into(),
            htm: method.into().to_ascii_uppercase(),
            htu: url.into(),
            iat: now.timestamp(),
            ath: None,
            nonce: None,
        }
    }

    /// Attach an access-token hash.
    pub fn with_access_token(mut self, access_token: &str) -> Self {
        use sha2::{Digest, Sha256};
        let h = Sha256::digest(access_token.as_bytes());
        self.ath = Some(base64_url_no_pad(&h));
        self
    }

    /// Attach a server-supplied DPoP nonce.
    pub fn with_nonce(mut self, nonce: impl Into<String>) -> Self {
        self.nonce = Some(nonce.into());
        self
    }
}

/// Validate a presented `DpopClaims` against an expected method + URL
/// and a clock-skew window.
pub fn validate_dpop_claims(
    claims: &DpopClaims,
    expected_method: &str,
    expected_url: &str,
    now: DateTime<Utc>,
    skew_seconds: i64,
) -> Result<(), OidcError> {
    if !claims.htm.eq_ignore_ascii_case(expected_method) {
        return Err(OidcError::Dpop(format!(
            "htm mismatch: {} != {}",
            claims.htm, expected_method
        )));
    }
    if claims.htu != expected_url {
        return Err(OidcError::Dpop(format!(
            "htu mismatch: {} != {}",
            claims.htu, expected_url
        )));
    }
    let delta = (now.timestamp() - claims.iat).abs();
    if delta > skew_seconds {
        return Err(OidcError::Dpop(format!(
            "iat skew too large: |{} - {}| > {}",
            now.timestamp(),
            claims.iat,
            skew_seconds
        )));
    }
    Ok(())
}

fn base64_url_no_pad(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn pre_auth_form_encoded() {
        let req = TokenRequest::pre_authorized("abc-123", Some("1234".into()));
        let form = req.to_form();
        assert!(form.contains("grant_type=urn"));
        assert!(form.contains("pre-authorized_code=abc-123"));
        assert!(form.contains("tx_code=1234"));
    }

    #[test]
    fn auth_code_form_encoded() {
        let req = TokenRequest::authorization_code(
            "code-1",
            "https://wallet.example/cb",
            "verifier-xyz",
        );
        let form = req.to_form();
        assert!(form.contains("grant_type=authorization_code"));
        assert!(form.contains("code_verifier=verifier-xyz"));
    }

    #[test]
    fn dpop_claims_roundtrip() {
        let now = Utc::now();
        let claims = DpopClaims::new(
            "POST",
            "https://issuer.example.com/token",
            now,
            "jti-1",
        )
        .with_access_token("access-token-xyz");
        assert_eq!(claims.htm, "POST");
        assert!(claims.ath.is_some());
        let j = serde_json::to_string(&claims).unwrap();
        let back: DpopClaims = serde_json::from_str(&j).unwrap();
        assert_eq!(back, claims);
    }

    #[test]
    fn dpop_validate_matches() {
        let now = Utc.with_ymd_and_hms(2026, 5, 24, 0, 0, 0).unwrap();
        let claims = DpopClaims::new(
            "post",
            "https://issuer.example.com/token",
            now,
            "jti-1",
        );
        validate_dpop_claims(
            &claims,
            "POST",
            "https://issuer.example.com/token",
            now,
            5,
        )
        .unwrap();
    }

    #[test]
    fn dpop_validate_rejects_method() {
        let now = Utc.with_ymd_and_hms(2026, 5, 24, 0, 0, 0).unwrap();
        let claims = DpopClaims::new(
            "POST",
            "https://issuer.example.com/token",
            now,
            "jti-1",
        );
        assert!(
            validate_dpop_claims(
                &claims,
                "GET",
                "https://issuer.example.com/token",
                now,
                5,
            )
            .is_err()
        );
    }

    #[test]
    fn dpop_validate_rejects_skew() {
        let now = Utc.with_ymd_and_hms(2026, 5, 24, 0, 0, 0).unwrap();
        let claims = DpopClaims::new(
            "POST",
            "https://issuer.example.com/token",
            now,
            "jti-1",
        );
        let later = now + chrono::Duration::seconds(120);
        assert!(
            validate_dpop_claims(
                &claims,
                "POST",
                "https://issuer.example.com/token",
                later,
                30,
            )
            .is_err()
        );
    }

    #[test]
    fn token_response_serialises() {
        let r = TokenResponse {
            access_token: "at-1".into(),
            token_type: "DPoP".into(),
            expires_in: Some(3600),
            refresh_token: None,
            scope: None,
            c_nonce: Some("n-1".into()),
            c_nonce_expires_in: Some(120),
            authorization_details: None,
        };
        let j = serde_json::to_string(&r).unwrap();
        assert!(j.contains("\"token_type\":\"DPoP\""));
        assert!(j.contains("\"c_nonce\":\"n-1\""));
    }
}
