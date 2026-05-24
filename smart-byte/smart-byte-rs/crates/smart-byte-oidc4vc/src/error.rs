//! Typed errors for OID4VC issuance, presentation, SIOP, DPoP, DCQL,
//! status lists, and profile validation (HAIP, EUDI ARF).

use thiserror::Error;

/// Single error enum covering every fallible operation in this crate.
#[derive(Debug, Error)]
pub enum OidcError {
    /// JSON encode / decode error.
    #[error("json error: {0}")]
    Json(String),
    /// Base64 / base64url codec error.
    #[error("base64 error: {0}")]
    Base64(String),
    /// URL parse error.
    #[error("url error: {0}")]
    Url(String),
    /// Authorisation-server metadata invalid.
    #[error("authorization server metadata: {0}")]
    AsMetadata(String),
    /// Credential-issuer metadata invalid.
    #[error("credential issuer metadata: {0}")]
    IssuerMetadata(String),
    /// Credential offer malformed or expired.
    #[error("credential offer: {0}")]
    Offer(String),
    /// Token endpoint error (`invalid_request`, `invalid_grant`, …).
    #[error("token endpoint: {0}")]
    Token(String),
    /// Credential endpoint error (`invalid_credential_request`, …).
    #[error("credential endpoint: {0}")]
    Credential(String),
    /// Notification endpoint error (`invalid_notification_id`, …).
    #[error("notification endpoint: {0}")]
    Notification(String),
    /// DPoP proof error (RFC 9449).
    #[error("dpop: {0}")]
    Dpop(String),
    /// DCQL query / match error.
    #[error("dcql: {0}")]
    Dcql(String),
    /// Presentation Exchange definition / submission error.
    #[error("presentation: {0}")]
    Presentation(String),
    /// SIOPv2 self-issued ID-token error.
    #[error("siop: {0}")]
    Siop(String),
    /// Status list (Bitstring / Token Status List) error.
    #[error("status list: {0}")]
    StatusList(String),
    /// HAIP profile constraint violation.
    #[error("haip: {0}")]
    Haip(String),
    /// EUDI ARF profile constraint violation.
    #[error("eudi: {0}")]
    Eudi(String),
    /// Cryptographic verification failed.
    #[error("signature: {0}")]
    Signature(String),
    /// Generic invalid input.
    #[error("invalid input: {0}")]
    Invalid(String),
}

impl From<serde_json::Error> for OidcError {
    fn from(e: serde_json::Error) -> Self {
        OidcError::Json(e.to_string())
    }
}

impl From<base64::DecodeError> for OidcError {
    fn from(e: base64::DecodeError) -> Self {
        OidcError::Base64(e.to_string())
    }
}

impl From<url::ParseError> for OidcError {
    fn from(e: url::ParseError) -> Self {
        OidcError::Url(e.to_string())
    }
}
