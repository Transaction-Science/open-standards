//! `did:jwk` — `did:jwk:<base64url-encoded-jwk>`.
//!
//! Offline. The identifier is the base64url (no-pad) of a canonical
//! JSON-encoded JWK. Resolution synthesises a DID document with one
//! verification method whose `publicKeyJwk` is the embedded JWK.

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;

use crate::did::{Did, DidMethod};
use crate::document::{
    DidDocument, Jwk, VerificationMethod, VerificationRelationship,
};
use crate::error::DidError;
use crate::resolver::{DocumentMetadata, ResolutionMetadata, ResolutionResult, Resolver};

/// `did:jwk` resolver.
pub struct JwkResolver;

impl JwkResolver {
    /// Construct a new resolver.
    pub fn new() -> Self {
        JwkResolver
    }
}

impl Default for JwkResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Resolver for JwkResolver {
    async fn resolve(&self, did: &Did) -> Result<ResolutionResult, DidError> {
        let doc = build_did_jwk_document(did)?;
        Ok(ResolutionResult {
            did_document: Some(doc),
            did_resolution_metadata: ResolutionMetadata {
                content_type: Some("application/did+json".into()),
                error: None,
            },
            did_document_metadata: DocumentMetadata::default(),
        })
    }
}

/// Encode a JWK into a `did:jwk` method-specific id.
pub fn encode_did_jwk(jwk: &Jwk) -> Result<String, DidError> {
    let bytes = serde_json::to_vec(jwk)?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

/// Decode a `did:jwk` method-specific id into a [`Jwk`].
pub fn decode_did_jwk(msid: &str) -> Result<Jwk, DidError> {
    let bytes = URL_SAFE_NO_PAD.decode(msid)?;
    let jwk: Jwk = serde_json::from_slice(&bytes)?;
    Ok(jwk)
}

/// Synthesise the DID document for a `did:jwk` DID.
pub fn build_did_jwk_document(did: &Did) -> Result<DidDocument, DidError> {
    if did.method != DidMethod::Jwk {
        return Err(DidError::InvalidIdentifier(format!(
            "not a did:jwk: {did}"
        )));
    }
    let jwk = decode_did_jwk(&did.method_specific_id)?;
    let vm_id = format!("{did}#0");
    let vm = VerificationMethod {
        id: vm_id.clone(),
        type_: "JsonWebKey".into(),
        controller: did.clone(),
        public_key_multibase: None,
        public_key_jwk: Some(jwk),
    };
    let mut doc = DidDocument::new(did.clone());
    doc.verification_method.push(vm);
    let rel = VerificationRelationship::Reference(vm_id);
    doc.authentication.push(rel.clone());
    doc.assertion_method.push(rel.clone());
    doc.capability_invocation.push(rel.clone());
    doc.capability_delegation.push(rel.clone());
    doc.key_agreement.push(rel);
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_ed25519_jwk() {
        let jwk = Jwk {
            kty: "OKP".into(),
            crv: Some("Ed25519".into()),
            x: Some("11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaPcHURo".into()),
            y: None,
            alg: Some("EdDSA".into()),
            kid: None,
            use_: None,
        };
        let msid = encode_did_jwk(&jwk).unwrap();
        let did: Did = format!("did:jwk:{msid}").parse().unwrap();
        let decoded = decode_did_jwk(&did.method_specific_id).unwrap();
        assert_eq!(decoded, jwk);
        let doc = build_did_jwk_document(&did).unwrap();
        assert_eq!(doc.verification_method.len(), 1);
        assert_eq!(
            doc.verification_method[0]
                .public_key_jwk
                .as_ref()
                .unwrap()
                .kty,
            "OKP"
        );
    }
}
