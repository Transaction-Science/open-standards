//! `did:peer` — inline-encoded DID documents (Aries / DIDComm).
//!
//! We implement the two practical numalgos used in deployments:
//!
//! * **numalgo 2** — `did:peer:2.<purpose><multikey>...[<service>]`.
//!   The method-specific id encodes one verification method per purpose
//!   ("V" = verification, "A" = assertion, "E" = key-agreement / encryption,
//!   "I" = capability-invocation, "D" = capability-delegation), each
//!   followed by a multibase Multikey; trailing `.S` segments carry
//!   base64url-encoded JSON service entries.
//! * **numalgo 4** — `did:peer:4<hash>:<encoded-document>`. The
//!   `<encoded-document>` is the multibase-encoded canonical document;
//!   `<hash>` is the SHA-256 of the encoded document (also multibased).
//!   This crate parses numalgo 4 strictly enough to expose the
//!   embedded document.

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use sha2::Digest;

use crate::did::{Did, DidMethod};
use crate::document::{
    DidDocument, Service, VerificationMethod, VerificationRelationship,
};
use crate::error::DidError;
use crate::resolver::{DocumentMetadata, ResolutionMetadata, ResolutionResult, Resolver};

/// `did:peer` resolver.
pub struct PeerResolver;

impl PeerResolver {
    /// Construct a new resolver.
    pub fn new() -> Self {
        PeerResolver
    }
}

impl Default for PeerResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Resolver for PeerResolver {
    async fn resolve(&self, did: &Did) -> Result<ResolutionResult, DidError> {
        let doc = resolve_peer(did)?;
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

/// Synthesise the DID document for a `did:peer` DID.
pub fn resolve_peer(did: &Did) -> Result<DidDocument, DidError> {
    if did.method != DidMethod::Peer {
        return Err(DidError::InvalidIdentifier(format!(
            "not a did:peer: {did}"
        )));
    }
    let msid = &did.method_specific_id;
    let numalgo = msid.chars().next().ok_or_else(|| {
        DidError::InvalidIdentifier("empty did:peer body".into())
    })?;
    match numalgo {
        '2' => resolve_numalgo2(did, &msid[1..]),
        '4' => resolve_numalgo4(did, &msid[1..]),
        other => Err(DidError::MethodNotSupported(format!(
            "did:peer numalgo {other} not implemented"
        ))),
    }
}

/// Build a numalgo 2 method-specific id from a list of (purpose, multikey).
pub fn encode_numalgo2(
    purposes: &[(char, String)],
    services_b64: &[String],
) -> String {
    let mut out = String::from("2");
    for (p, k) in purposes {
        out.push('.');
        out.push(*p);
        out.push_str(k);
    }
    for s in services_b64 {
        out.push_str(".S");
        out.push_str(s);
    }
    out
}

fn resolve_numalgo2(did: &Did, body: &str) -> Result<DidDocument, DidError> {
    // Body starts with `.` then has a sequence of `<purpose><multikey>`
    // segments separated by `.`, possibly followed by `.S<b64>` services.
    let segments: Vec<&str> = body.split('.').filter(|s| !s.is_empty()).collect();
    if segments.is_empty() {
        return Err(DidError::InvalidIdentifier(
            "did:peer numalgo 2 has no segments".into(),
        ));
    }
    let mut doc = DidDocument::new(did.clone());
    for seg in segments {
        let mut chars = seg.chars();
        let purpose = chars.next().ok_or_else(|| {
            DidError::InvalidIdentifier(
                "empty did:peer segment".into(),
            )
        })?;
        let rest = &seg[purpose.len_utf8()..];
        if purpose == 'S' {
            // Base64url-decoded JSON service.
            let bytes = URL_SAFE_NO_PAD.decode(rest).map_err(|e| {
                DidError::InvalidIdentifier(format!(
                    "did:peer service b64 decode: {e}"
                ))
            })?;
            let svc: Service = serde_json::from_slice(&bytes)?;
            doc.service.push(svc);
            continue;
        }
        // Otherwise: <purpose><multikey>. The multikey is multibase
        // (z-prefixed base58btc) carrying multicodec || key.
        if !rest.starts_with('z') {
            return Err(DidError::InvalidIdentifier(format!(
                "did:peer multikey must be base58btc (z-prefix), got {rest}"
            )));
        }
        let vm_id = format!("{did}#{rest}");
        let vm = VerificationMethod {
            id: vm_id.clone(),
            type_: "Multikey".into(),
            controller: did.clone(),
            public_key_multibase: Some(rest.to_string()),
            public_key_jwk: None,
        };
        doc.verification_method.push(vm);
        let rel = VerificationRelationship::Reference(vm_id);
        match purpose {
            'V' => doc.authentication.push(rel),
            'A' => doc.assertion_method.push(rel),
            'E' => doc.key_agreement.push(rel),
            'I' => doc.capability_invocation.push(rel),
            'D' => doc.capability_delegation.push(rel),
            other => {
                return Err(DidError::InvalidIdentifier(format!(
                    "unknown did:peer purpose '{other}'"
                )));
            }
        }
    }
    Ok(doc)
}

fn resolve_numalgo4(did: &Did, body: &str) -> Result<DidDocument, DidError> {
    // Body is `<hash>:<encoded-document>`. Both are multibase strings.
    let (hash_part, encoded_part) = body.split_once(':').ok_or_else(|| {
        DidError::InvalidIdentifier(
            "did:peer numalgo 4 missing ':' separator".into(),
        )
    })?;
    let (_, hash_bytes) = multibase::decode(hash_part)?;
    let (_, doc_bytes) = multibase::decode(encoded_part)?;
    // Verify the embedded document hashes correctly.
    let mut hasher = sha2::Sha256::new();
    hasher.update(encoded_part.as_bytes());
    let computed = hasher.finalize();
    if hash_bytes.len() < 2 || computed[..] != hash_bytes[2..] {
        // The hash is allowed to be a multihash with a 2-byte prefix
        // (0x12 0x20 for sha256-256). Accept either form by checking
        // the suffix. If neither matches, error.
        if computed[..] != hash_bytes[..] {
            return Err(DidError::InvalidDocument(
                "did:peer numalgo 4 hash mismatch".into(),
            ));
        }
    }
    let doc: DidDocument = serde_json::from_slice(&doc_bytes)?;
    if doc.id != *did {
        return Err(DidError::InvalidDocument(format!(
            "embedded did:peer document id {} does not match {}",
            doc.id, did
        )));
    }
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numalgo2_round_trip_one_key() {
        // Use a known Ed25519 multikey from the W3C registry.
        let mk = "z6MkiTBz1ymuepAQ4HEHYSF1H8quG5GLVVQR3djdX3mDooWp";
        let msid = encode_numalgo2(&[('V', mk.into())], &[]);
        let did_str = format!("did:peer:{msid}");
        let did: Did = did_str.parse().unwrap();
        let doc = resolve_peer(&did).unwrap();
        assert_eq!(doc.verification_method.len(), 1);
        assert_eq!(doc.authentication.len(), 1);
    }

    #[test]
    fn numalgo2_with_service() {
        let mk = "z6MkiTBz1ymuepAQ4HEHYSF1H8quG5GLVVQR3djdX3mDooWp";
        let svc = serde_json::json!({
            "id": "#didcomm",
            "type": "DIDCommMessaging",
            "serviceEndpoint": "https://example.com/endpoint",
        });
        let s_b64 = URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&svc).unwrap());
        let msid = encode_numalgo2(&[('V', mk.into())], &[s_b64]);
        let did: Did = format!("did:peer:{msid}").parse().unwrap();
        let doc = resolve_peer(&did).unwrap();
        assert_eq!(doc.service.len(), 1);
        assert_eq!(doc.service[0].type_, "DIDCommMessaging");
    }
}
