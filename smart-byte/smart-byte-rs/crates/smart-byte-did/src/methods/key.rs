//! `did:key` — single-key, self-certifying DIDs (W3C DID Method registry).
//!
//! The method-specific id is `z<multibase-base58btc>(<varint-multicodec> ||
//! <raw-public-key>)`. Resolution is purely offline: given the
//! multibase-encoded key, we synthesise a DID document with one
//! [`VerificationMethod`] of type `Multikey` (DID Core / Multikey 2024).
//!
//! Supported codecs: Ed25519, P-256, secp256k1; BLS12-381 G2 behind the
//! `bls12_381` feature (parsing only — synthesis still reuses the raw
//! multibase form).

use async_trait::async_trait;

use crate::did::{Did, DidMethod};
use crate::document::{
    DidDocument, VerificationMethod, VerificationRelationship,
};
use crate::error::DidError;
use crate::methods::multicodec::{
    BLS12_381_G2_PUB, ED25519_PUB, P256_PUB, SECP256K1_PUB, varint_decode,
    varint_encode,
};
use crate::resolver::{DocumentMetadata, ResolutionMetadata, ResolutionResult, Resolver};

/// Recognised key types for `did:key`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyType {
    /// Ed25519 (multicodec 0xed).
    Ed25519,
    /// NIST P-256 / secp256r1 (multicodec 0x1200).
    P256,
    /// secp256k1 (multicodec 0xe7).
    Secp256k1,
    /// BLS12-381 G2 (multicodec 0xeb).
    Bls12381G2,
}

impl KeyType {
    /// Expected raw key length, in bytes, for this codec.
    pub fn key_len(&self) -> usize {
        match self {
            KeyType::Ed25519 => 32,
            // Compressed SEC1.
            KeyType::P256 | KeyType::Secp256k1 => 33,
            KeyType::Bls12381G2 => 96,
        }
    }

    fn codec(&self) -> u64 {
        match self {
            KeyType::Ed25519 => ED25519_PUB,
            KeyType::P256 => P256_PUB,
            KeyType::Secp256k1 => SECP256K1_PUB,
            KeyType::Bls12381G2 => BLS12_381_G2_PUB,
        }
    }

    fn from_codec(c: u64) -> Result<Self, DidError> {
        match c {
            ED25519_PUB => Ok(KeyType::Ed25519),
            P256_PUB => Ok(KeyType::P256),
            SECP256K1_PUB => Ok(KeyType::Secp256k1),
            BLS12_381_G2_PUB => Ok(KeyType::Bls12381G2),
            other => Err(DidError::UnsupportedKeyCodec(other)),
        }
    }
}

/// Encode a public key + codec as a `did:key` method-specific id.
pub fn encode_did_key(kind: KeyType, raw_pub: &[u8]) -> Result<String, DidError> {
    if raw_pub.len() != kind.key_len() {
        return Err(DidError::InvalidKey(format!(
            "expected {} bytes for {:?}, got {}",
            kind.key_len(),
            kind,
            raw_pub.len()
        )));
    }
    let mut buf = Vec::with_capacity(raw_pub.len() + 2);
    varint_encode(kind.codec(), &mut buf);
    buf.extend_from_slice(raw_pub);
    Ok(multibase::encode(multibase::Base::Base58Btc, &buf))
}

/// Decode a `did:key` method-specific id into `(KeyType, raw_pub)`.
pub fn decode_did_key(msid: &str) -> Result<(KeyType, Vec<u8>), DidError> {
    let (base, bytes) = multibase::decode(msid)?;
    if base != multibase::Base::Base58Btc {
        return Err(DidError::InvalidIdentifier(format!(
            "did:key must use base58btc (z-prefix); got {base:?}"
        )));
    }
    let (codec, used) = varint_decode(&bytes).ok_or_else(|| {
        DidError::InvalidIdentifier("malformed multicodec varint".into())
    })?;
    let kind = KeyType::from_codec(codec)?;
    let raw = bytes[used..].to_vec();
    if raw.len() != kind.key_len() {
        return Err(DidError::InvalidKey(format!(
            "expected {} bytes for {:?}, got {}",
            kind.key_len(),
            kind,
            raw.len()
        )));
    }
    // For p256/k256, validate point-compression byte.
    match kind {
        KeyType::P256 => {
            p256::PublicKey::from_sec1_bytes(&raw).map_err(|e| {
                DidError::InvalidKey(format!("p256 sec1 decode: {e}"))
            })?;
        }
        KeyType::Secp256k1 => {
            k256::PublicKey::from_sec1_bytes(&raw).map_err(|e| {
                DidError::InvalidKey(format!(
                    "secp256k1 sec1 decode: {e}"
                ))
            })?;
        }
        KeyType::Ed25519 => {
            // ed25519-dalek validates on use; we accept any 32 bytes here.
            let arr: [u8; 32] = raw
                .clone()
                .try_into()
                .map_err(|_| DidError::InvalidKey("ed25519 length".into()))?;
            ed25519_dalek::VerifyingKey::from_bytes(&arr).map_err(|e| {
                DidError::InvalidKey(format!("ed25519: {e}"))
            })?;
        }
        KeyType::Bls12381G2 => {
            // Parsing/validation not in scope without feature deps.
        }
    }
    Ok((kind, raw))
}

/// Synthesise the DID document for a `did:key` DID.
pub fn build_did_key_document(did: &Did) -> Result<DidDocument, DidError> {
    if did.method != DidMethod::Key {
        return Err(DidError::InvalidIdentifier(format!(
            "not a did:key: {did}"
        )));
    }
    let (_kind, _raw) = decode_did_key(&did.method_specific_id)?;
    let vm_id = format!("{did}#{}", did.method_specific_id);
    let vm = VerificationMethod {
        id: vm_id.clone(),
        type_: "Multikey".to_string(),
        controller: did.clone(),
        public_key_multibase: Some(did.method_specific_id.clone()),
        public_key_jwk: None,
    };
    let mut doc = DidDocument::new(did.clone());
    doc.verification_method.push(vm);
    let rel = VerificationRelationship::Reference(vm_id);
    doc.authentication.push(rel.clone());
    doc.assertion_method.push(rel.clone());
    doc.capability_invocation.push(rel.clone());
    doc.capability_delegation.push(rel);
    Ok(doc)
}

/// `did:key` resolver. Resolution is offline.
pub struct KeyResolver;

impl KeyResolver {
    /// Construct a new resolver.
    pub fn new() -> Self {
        KeyResolver
    }
}

impl Default for KeyResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Resolver for KeyResolver {
    async fn resolve(&self, did: &Did) -> Result<ResolutionResult, DidError> {
        let doc = build_did_key_document(did)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ed25519_well_known_fixture_roundtrip() {
        // W3C did:key test vector (Multikey form for an Ed25519 key).
        let did_str =
            "did:key:z6MkiTBz1ymuepAQ4HEHYSF1H8quG5GLVVQR3djdX3mDooWp";
        let did: Did = did_str.parse().unwrap();
        let (kind, raw) = decode_did_key(&did.method_specific_id).unwrap();
        assert_eq!(kind, KeyType::Ed25519);
        assert_eq!(raw.len(), 32);
        let reencoded = encode_did_key(KeyType::Ed25519, &raw).unwrap();
        assert_eq!(reencoded, did.method_specific_id);
    }

    #[test]
    fn p256_roundtrip_from_deterministic_scalar() {
        use p256::elliptic_curve::sec1::ToSec1Point;
        // Deterministic scalar — any non-zero value below the group order.
        let scalar_bytes = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
            0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13,
            0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f,
        ];
        let sk = p256::SecretKey::from_slice(&scalar_bytes).unwrap();
        let pk = sk.public_key();
        let pt = pk.to_sec1_point(true);
        let raw = pt.as_bytes().to_vec();
        let msid = encode_did_key(KeyType::P256, &raw).unwrap();
        let (kind, decoded) = decode_did_key(&msid).unwrap();
        assert_eq!(kind, KeyType::P256);
        assert_eq!(decoded, raw);
    }

    #[test]
    fn secp256k1_roundtrip_from_deterministic_scalar() {
        use k256::elliptic_curve::sec1::ToSec1Point;
        let scalar_bytes = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
            0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13,
            0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f,
        ];
        let sk = k256::SecretKey::from_slice(&scalar_bytes).unwrap();
        let pk = sk.public_key();
        let pt = pk.to_sec1_point(true);
        let raw = pt.as_bytes().to_vec();
        let msid = encode_did_key(KeyType::Secp256k1, &raw).unwrap();
        let (kind, decoded) = decode_did_key(&msid).unwrap();
        assert_eq!(kind, KeyType::Secp256k1);
        assert_eq!(decoded, raw);
    }

    #[test]
    fn rejects_wrong_codec() {
        // Multibase-decode something with a non-key codec.
        let mut bad = Vec::new();
        varint_encode(0x12, &mut bad); // sha2-256 — not a key codec
        bad.extend_from_slice(&[0u8; 32]);
        let msid = multibase::encode(multibase::Base::Base58Btc, &bad);
        let err = decode_did_key(&msid).unwrap_err();
        assert!(matches!(err, DidError::UnsupportedKeyCodec(_)));
    }

    #[test]
    fn builds_did_document() {
        let did: Did =
            "did:key:z6MkiTBz1ymuepAQ4HEHYSF1H8quG5GLVVQR3djdX3mDooWp"
                .parse()
                .unwrap();
        let doc = build_did_key_document(&did).unwrap();
        assert_eq!(doc.id, did);
        assert_eq!(doc.verification_method.len(), 1);
        assert_eq!(doc.authentication.len(), 1);
    }

}
