//! Credential endpoint + nonce endpoint (OID4VCI draft 13 §7 + §8).
//!
//! The wallet POSTs a [`CredentialRequest`] to the issuer's
//! `credential_endpoint`, including a `proof` (or `proofs`) that binds
//! the credential to a wallet-held key. The proof's `c_nonce` is
//! produced by the token endpoint (in [`crate::token::TokenResponse`])
//! or by the dedicated nonce endpoint.
//!
//! Three credential responses are possible:
//!
//! * **Immediate** — `credential` (single) or `credentials` (batch).
//! * **Deferred** — `transaction_id` plus polling at
//!   `deferred_credential_endpoint`.
//! * **Error** — `invalid_credential_request`, `invalid_proof`,
//!   `unsupported_credential_format`, `unsupported_credential_type`, ….

use serde::{Deserialize, Serialize};

use crate::error::OidcError;

/// One credential proof inside a credential request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialProof {
    /// Proof type tag, e.g. `jwt`, `cwt`, `ldp_vp`.
    pub proof_type: String,
    /// JWT proof string (proof_type == `jwt`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jwt: Option<String>,
    /// CWT proof bytes, base64url (proof_type == `cwt`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwt: Option<String>,
    /// LDP-VP proof object (proof_type == `ldp_vp`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ldp_vp: Option<serde_json::Value>,
}

/// `proofs` object — batched proofs (OID4VCI 13).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialProofs {
    /// Array of JWT proofs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub jwt: Vec<String>,
    /// Array of CWT proofs (base64url).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cwt: Vec<String>,
}

impl CredentialProofs {
    /// True if no proofs are present.
    pub fn is_empty(&self) -> bool {
        self.jwt.is_empty() && self.cwt.is_empty()
    }
}

/// `credential_response_encryption` block (encrypted response).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialResponseEncryption {
    /// JWK the issuer must encrypt to.
    pub jwk: serde_json::Value,
    /// Key-management `alg`.
    pub alg: String,
    /// Content `enc`.
    pub enc: String,
}

/// Credential request body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialRequest {
    /// Credential configuration id selected by the wallet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential_configuration_id: Option<String>,
    /// `format` (legacy parameter; mutually exclusive with
    /// `credential_configuration_id` in some deployments).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// VCT for SD-JWT VC selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vct: Option<String>,
    /// Doctype for mdoc selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doctype: Option<String>,
    /// Single proof.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof: Option<CredentialProof>,
    /// Multi-proof batch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proofs: Option<CredentialProofs>,
    /// Optional response encryption.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential_response_encryption: Option<CredentialResponseEncryption>,
}

impl CredentialRequest {
    /// Validate that exactly one of `proof` or `proofs` is present, and
    /// that some configuration identifier is supplied.
    pub fn validate(&self) -> Result<(), OidcError> {
        let has_single = self.proof.is_some();
        let has_batch = self
            .proofs
            .as_ref()
            .map(|p| !p.is_empty())
            .unwrap_or(false);
        if has_single && has_batch {
            return Err(OidcError::Credential(
                "request MUST NOT include both proof and proofs".into(),
            ));
        }
        if !has_single && !has_batch {
            return Err(OidcError::Credential(
                "request MUST include proof or proofs".into(),
            ));
        }
        if self.credential_configuration_id.is_none()
            && self.format.is_none()
            && self.vct.is_none()
            && self.doctype.is_none()
        {
            return Err(OidcError::Credential(
                "credential_configuration_id, format, vct, or doctype required"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Credential response (immediate or deferred).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialResponse {
    /// Single credential string (compact-serialised JWT, SD-JWT, mdoc
    /// CBOR base64url, or JSON-LD VC).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential: Option<serde_json::Value>,
    /// Batch credentials.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub credentials: Vec<serde_json::Value>,
    /// Transaction id for deferred issuance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transaction_id: Option<String>,
    /// Optional fresh `c_nonce` (for follow-up requests).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub c_nonce: Option<String>,
    /// `c_nonce` expiry in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub c_nonce_expires_in: Option<u64>,
    /// Notification id (for [`crate::notification`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notification_id: Option<String>,
}

impl CredentialResponse {
    /// True if the response is deferred (transaction_id present).
    pub fn is_deferred(&self) -> bool {
        self.transaction_id.is_some()
    }
}

/// Nonce endpoint response (OID4VCI 13 §7.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NonceResponse {
    /// New nonce string.
    pub c_nonce: String,
    /// Lifetime in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub c_nonce_expires_in: Option<u64>,
}

/// Construct a credential request bearing a single JWT proof.
pub fn credential_request_jwt(
    configuration_id: impl Into<String>,
    proof_jwt: impl Into<String>,
) -> CredentialRequest {
    CredentialRequest {
        credential_configuration_id: Some(configuration_id.into()),
        format: None,
        vct: None,
        doctype: None,
        proof: Some(CredentialProof {
            proof_type: "jwt".into(),
            jwt: Some(proof_jwt.into()),
            cwt: None,
            ldp_vp: None,
        }),
        proofs: None,
        credential_response_encryption: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jwt_request_validates() {
        let req = credential_request_jwt("UniversityDegree_SD_JWT", "ey.zzz");
        req.validate().unwrap();
    }

    #[test]
    fn rejects_no_proof() {
        let req = CredentialRequest {
            credential_configuration_id: Some("c1".into()),
            format: None,
            vct: None,
            doctype: None,
            proof: None,
            proofs: None,
            credential_response_encryption: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn rejects_both_proofs() {
        let req = CredentialRequest {
            credential_configuration_id: Some("c1".into()),
            format: None,
            vct: None,
            doctype: None,
            proof: Some(CredentialProof {
                proof_type: "jwt".into(),
                jwt: Some("ey.x".into()),
                cwt: None,
                ldp_vp: None,
            }),
            proofs: Some(CredentialProofs {
                jwt: vec!["ey.y".into()],
                cwt: vec![],
            }),
            credential_response_encryption: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn deferred_response() {
        let r = CredentialResponse {
            credential: None,
            credentials: vec![],
            transaction_id: Some("tx-1".into()),
            c_nonce: None,
            c_nonce_expires_in: None,
            notification_id: None,
        };
        assert!(r.is_deferred());
    }

    #[test]
    fn immediate_response_serialises() {
        let r = CredentialResponse {
            credential: Some(serde_json::Value::from("ey.payload.sig")),
            credentials: vec![],
            transaction_id: None,
            c_nonce: Some("n-2".into()),
            c_nonce_expires_in: Some(120),
            notification_id: Some("notif-1".into()),
        };
        let j = serde_json::to_string(&r).unwrap();
        assert!(j.contains("\"credential\":\"ey.payload.sig\""));
    }
}
