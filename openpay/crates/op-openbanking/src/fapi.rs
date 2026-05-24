//! FAPI 1.0 Advanced security profile interfaces.
//!
//! [FAPI](https://openid.net/specs/openid-financial-api-part-2-1_0.html) is the
//! OpenID Foundation's profile of OAuth 2.0 that the major Open Banking
//! regimes mandate. It pins:
//!
//! - mTLS for client authentication (RFC 8705), with certificate-bound
//!   access tokens (`cnf.x5t#S256`).
//! - JWS request objects (RFC 9101, formerly RFC 7515 / FAPI § 5.2.2)
//!   signed with `PS256` or `ES256`. `RS256` is forbidden in
//!   Advanced; `none` is forbidden everywhere.
//! - `s_hash` and `c_hash` covering the `state` and `code` values to
//!   bind the authorization-code grant.
//! - `x-fapi-interaction-id` request/response header for forensic
//!   traceability.
//!
//! This module exposes the *interfaces*, not implementations. Operators
//! plug in a KMS / HSM / eIDAS QSealC card behind [`JwsSigner`] and an
//! mTLS-terminating proxy behind [`MtlsClientCert`].

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// FAPI profile level the operator has deployed.
///
/// The crate is wired for Advanced everywhere; Baseline is recognised
/// for legacy ASPSPs that have not yet migrated. Some ASPSPs (UK
/// CMA9 banks) require Advanced unconditionally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FapiProfile {
    /// FAPI 1.0 Baseline. Read-only AISP-style flows only.
    Baseline,
    /// FAPI 1.0 Advanced. Mandatory for PISP, VRP, and CBPII.
    Advanced,
}

/// Approved JWS algorithms per FAPI 1.0 Advanced § 8.6.
///
/// `RS256` is explicitly *forbidden* in Advanced (CVE-2018-0114 family
/// of attacks against pre-PSS RSA signatures), so we do not enumerate
/// it here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JwsAlgorithm {
    /// RSASSA-PSS using SHA-256 and MGF1 with SHA-256.
    Ps256,
    /// ECDSA using P-256 and SHA-256.
    Es256,
}

impl JwsAlgorithm {
    /// IANA-registered string used in the JWS `alg` header parameter.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ps256 => "PS256",
            Self::Es256 => "ES256",
        }
    }
}

/// SHA-256 JWK thumbprint per RFC 7638. Twenty bytes are not enough;
/// FAPI binds tokens to the full 32-byte digest, base64url-encoded
/// in `cnf.x5t#S256`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JwkThumbprint(pub [u8; 32]);

/// A request object as carried in the FAPI `request` parameter.
///
/// The crate constructs the canonical payload (claims + protected
/// header) and hands it to [`JwsSigner::sign`]. The signer returns
/// the compact JWS string the operator sends on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestObject {
    /// JOSE protected header in JSON form.
    pub protected_header: String,
    /// Canonical claims payload in JSON form.
    pub payload: String,
    /// The algorithm the signer must use.
    pub alg: JwsAlgorithm,
    /// Key ID (`kid`) the signer should use, if multiple keys are
    /// registered with the ASPSP / OIDF directory.
    pub kid: Option<String>,
}

/// Detached JWS signature value (the trailing `..signature` form
/// FAPI uses for `x-jws-signature` over the HTTP body) or compact
/// JWS string (`header.payload.signature` for request objects).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JwsSignature(pub String);

/// Trait implemented by operators to provide cryptographic signing
/// without exposing key material to the crate.
///
/// **No default impl is provided.** Operators must wire a KMS, HSM,
/// eIDAS QSealC card, or OBSeal directory entry. The crate refuses
/// to ship a soft-key signer because a leaked operator key on a
/// regulated rail is a regulatory incident, not a bug.
pub trait JwsSigner: Send + Sync {
    /// Sign a request object.
    fn sign(&self, request: &RequestObject) -> Result<JwsSignature>;

    /// Sign an arbitrary HTTP body for the FAPI `x-jws-signature`
    /// detached-payload header.
    fn sign_detached(&self, body: &[u8], alg: JwsAlgorithm) -> Result<JwsSignature>;

    /// SHA-256 thumbprint of the public key this signer is bound to.
    fn key_thumbprint(&self) -> Result<JwkThumbprint>;
}

/// Trait implemented by operators to surface the mTLS client
/// certificate that terminates inbound on the ASPSP side.
///
/// Most production deployments terminate mTLS in a proxy (Envoy,
/// nginx, CloudHSM-fronted broker). The proxy then passes the
/// client cert thumbprint down to the application via an
/// HTTP header (`X-SSL-Client-SHA256`, similar). This trait
/// abstracts that path.
pub trait MtlsClientCert: Send + Sync {
    /// SHA-256 thumbprint of the client certificate the operator
    /// presented to the ASPSP. Used to verify `cnf.x5t#S256`
    /// binding on the resulting access token (RFC 8705 § 3).
    fn cert_thumbprint(&self) -> Result<JwkThumbprint>;

    /// Subject DN of the client certificate, as required by some
    /// ASPSPs for additional binding checks (Berlin Group XS2A
    /// passes the subject DN in `TPP-Signature-Certificate`).
    fn subject_dn(&self) -> Result<String>;
}

/// Trait implemented by operators to register JWKs with an ASPSP or
/// open-banking directory (OBIE, eIDAS QTSP, Open Banking Brasil
/// directory).
///
/// JWK registration is a side-channel: the operator publishes their
/// signing public keys to a directory, the ASPSP fetches them at
/// connection-establishment time, then accepts JWS request objects
/// signed by those keys. The trait is intentionally narrow.
pub trait JwkRegistration: Send + Sync {
    /// Look up a JWK by thumbprint. Returns the JWK as a JSON string
    /// in RFC 7517 form. `None` if not registered.
    fn lookup(&self, thumbprint: &JwkThumbprint) -> Result<Option<String>>;
}

/// An OAuth 2.0 token as returned by an ASPSP authorization server.
///
/// In FAPI deployments, `access_token.cnf.x5t#S256` is bound to the
/// mTLS client certificate. Refresh tokens are bound the same way.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OAuth2Token {
    /// Opaque or JWT-formatted access token.
    pub access_token: String,
    /// Token type. Almost always `Bearer` in FAPI; the cert-binding
    /// is enforced by the resource server, not by the token type.
    pub token_type: String,
    /// Scopes granted (may be narrower than what was requested).
    pub scopes: Vec<String>,
    /// Seconds until expiry, as advertised by the AS at issuance.
    pub expires_in: u64,
    /// Optional refresh token. UK OBIE long-lived consents emit one;
    /// Berlin Group typically does not.
    pub refresh_token: Option<String>,
    /// Certificate thumbprint the token is bound to, when the AS
    /// supports RFC 8705 § 3. `None` when binding is not in use.
    pub cert_thumbprint: Option<JwkThumbprint>,
}

impl OAuth2Token {
    /// Verify the token's certificate binding against the live
    /// mTLS client certificate.
    ///
    /// Returns [`crate::Error::CertificateBindingMismatch`] when the
    /// token carries a `cnf.x5t#S256` that does not match the
    /// certificate the operator is currently presenting.
    ///
    /// Tokens that do not carry a binding (`cert_thumbprint == None`)
    /// pass through. That's a deployment choice, not a bug — some
    /// sandbox ASPSPs accept bearer tokens without binding.
    pub fn verify_binding(&self, mtls: &dyn MtlsClientCert) -> Result<()> {
        if let Some(bound) = &self.cert_thumbprint {
            let live = mtls.cert_thumbprint()?;
            if bound != &live {
                return Err(crate::Error::CertificateBindingMismatch);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedCert(JwkThumbprint);

    impl MtlsClientCert for FixedCert {
        fn cert_thumbprint(&self) -> Result<JwkThumbprint> {
            Ok(self.0.clone())
        }
        fn subject_dn(&self) -> Result<String> {
            Ok("CN=test-tpp,O=OpenPay,C=GB".into())
        }
    }

    #[test]
    fn binding_matches() {
        let tp = JwkThumbprint([7u8; 32]);
        let token = OAuth2Token {
            access_token: "x".into(),
            token_type: "Bearer".into(),
            scopes: vec!["accounts".into()],
            expires_in: 3600,
            refresh_token: None,
            cert_thumbprint: Some(tp.clone()),
        };
        let cert = FixedCert(tp);
        token.verify_binding(&cert).expect("matches");
    }

    #[test]
    fn binding_mismatch_rejected() {
        let token = OAuth2Token {
            access_token: "x".into(),
            token_type: "Bearer".into(),
            scopes: vec!["accounts".into()],
            expires_in: 3600,
            refresh_token: None,
            cert_thumbprint: Some(JwkThumbprint([1u8; 32])),
        };
        let cert = FixedCert(JwkThumbprint([2u8; 32]));
        let err = token.verify_binding(&cert).expect_err("mismatch");
        assert!(matches!(err, crate::Error::CertificateBindingMismatch));
    }

    #[test]
    fn alg_names_match_iana() {
        assert_eq!(JwsAlgorithm::Ps256.as_str(), "PS256");
        assert_eq!(JwsAlgorithm::Es256.as_str(), "ES256");
    }

    #[test]
    fn unbound_token_passes_through() {
        let token = OAuth2Token {
            access_token: "x".into(),
            token_type: "Bearer".into(),
            scopes: vec!["accounts".into()],
            expires_in: 3600,
            refresh_token: None,
            cert_thumbprint: None,
        };
        let cert = FixedCert(JwkThumbprint([0u8; 32]));
        token.verify_binding(&cert).expect("unbound ok");
    }
}
