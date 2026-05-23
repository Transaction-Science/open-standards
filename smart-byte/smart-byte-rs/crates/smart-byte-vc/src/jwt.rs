//! VC-JWT — RFC 7519 JWT carrying a Verifiable Credential payload.
//!
//! Signing algorithm: EdDSA (Ed25519). Encoding: compact JWS
//! (`base64url(header) . base64url(payload) . base64url(signature)`).
//!
//! Per VCDM 2.0 §6.3 ("Securing Mechanism"), the payload is the
//! credential JSON itself (no `iss`/`sub`/`iat` shoehorning) — the
//! VC fields are authoritative.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ed25519_dalek::{
    Signature, SigningKey, Verifier, VerifyingKey, ed25519::signature::Signer,
};
use serde::{Deserialize, Serialize};

use crate::credential::VerifiableCredential;
use crate::error::VcError;
use crate::proof::JwtProof;

/// JOSE header for VC-JWT.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JwtHeader {
    /// Signing algorithm. We support `"EdDSA"`.
    pub alg: String,
    /// Token type. `"vc+jwt"` per VCDM 2.0; legacy issuers use `"JWT"`.
    pub typ: String,
    /// Verification method (key id). Typically a DID URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kid: Option<String>,
}

impl JwtHeader {
    /// Header preset for EdDSA / VC-JWT.
    pub fn eddsa(kid: Option<String>) -> Self {
        Self {
            alg: "EdDSA".to_string(),
            typ: "vc+jwt".to_string(),
            kid,
        }
    }
}

/// Issue a VC-JWT. The compact serialisation is returned both as a raw
/// string and wrapped in a [`JwtProof`] for ergonomic insertion into a
/// credential's `proof` array (when in-band binding is desired).
pub fn issue_vc_jwt(
    vc: &VerifiableCredential,
    signing_key: &SigningKey,
    kid: Option<String>,
) -> Result<(String, JwtProof), VcError> {
    let header = JwtHeader::eddsa(kid);
    let header_bytes = serde_json::to_vec(&header)?;
    let payload_bytes = serde_jcs::to_vec(vc).map_err(|e| VcError::Jcs(e.to_string()))?;
    let h = URL_SAFE_NO_PAD.encode(&header_bytes);
    let p = URL_SAFE_NO_PAD.encode(&payload_bytes);
    let signing_input = format!("{h}.{p}");
    let sig: Signature = signing_key.sign(signing_input.as_bytes());
    let s = URL_SAFE_NO_PAD.encode(sig.to_bytes());
    let jwt = format!("{signing_input}.{s}");
    let proof = JwtProof {
        type_: "JwtProof2020".to_string(),
        jwt: jwt.clone(),
    };
    Ok((jwt, proof))
}

/// Verify a VC-JWT compact string with `verifying_key`. Returns the
/// decoded credential on success.
pub fn verify_vc_jwt(
    jwt: &str,
    verifying_key: &VerifyingKey,
) -> Result<VerifiableCredential, VcError> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        return Err(VcError::Jwt(format!(
            "expected 3 compact JWS parts, got {}",
            parts.len()
        )));
    }
    let header_bytes = URL_SAFE_NO_PAD.decode(parts[0])?;
    let header: JwtHeader = serde_json::from_slice(&header_bytes)?;
    if header.alg != "EdDSA" {
        return Err(VcError::Jwt(format!(
            "unsupported alg: {}",
            header.alg
        )));
    }
    let payload_bytes = URL_SAFE_NO_PAD.decode(parts[1])?;
    let sig_bytes = URL_SAFE_NO_PAD.decode(parts[2])?;
    if sig_bytes.len() != 64 {
        return Err(VcError::Jwt(format!(
            "expected 64-byte Ed25519 signature, got {}",
            sig_bytes.len()
        )));
    }
    let mut s = [0u8; 64];
    s.copy_from_slice(&sig_bytes);
    let signature = Signature::from_bytes(&s);
    let signing_input = format!("{}.{}", parts[0], parts[1]);
    verifying_key
        .verify(signing_input.as_bytes(), &signature)
        .map_err(|e| VcError::Signature(e.to_string()))?;
    let vc: VerifiableCredential = serde_json::from_slice(&payload_bytes)?;
    Ok(vc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credential::{CredentialSubject, VcBuilder};
    use crate::issuer::Issuer;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn fixture_vc() -> VerifiableCredential {
        let subj = CredentialSubject {
            id: Some("did:example:alice".parse().unwrap()),
            claims: serde_json::Map::new(),
        };
        VcBuilder::new()
            .issuer(Issuer::Uri("did:example:issuer".parse().unwrap()))
            .subject(subj)
            .build()
            .unwrap()
    }

    #[test]
    fn jwt_roundtrip() {
        let sk = SigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let vc = fixture_vc();
        let (jwt, _proof) = issue_vc_jwt(
            &vc,
            &sk,
            Some("did:example:issuer#keys-1".into()),
        )
        .unwrap();
        let back = verify_vc_jwt(&jwt, &vk).unwrap();
        assert_eq!(back.issuer.id().as_str(), "did:example:issuer");
    }

    #[test]
    fn jwt_rejects_wrong_key() {
        let sk = SigningKey::generate(&mut OsRng);
        let other = SigningKey::generate(&mut OsRng);
        let vc = fixture_vc();
        let (jwt, _) = issue_vc_jwt(&vc, &sk, None).unwrap();
        assert!(verify_vc_jwt(&jwt, &other.verifying_key()).is_err());
    }
}
