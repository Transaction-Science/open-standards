//! ISO 18013-5 / 18013-7 session transcript.
//!
//! Both the offline device-engagement flow (18013-5 §9.1.5) and the
//! online presentment flow (18013-7) bind device authentication to a
//! session transcript:
//!
//! ```text
//! SessionTranscript = [
//!   DeviceEngagementBytes,   // #6.24(bstr .cbor DeviceEngagement) or null
//!   EReaderKeyBytes,         // #6.24(bstr .cbor COSE_Key) or null
//!   Handover,                // [HandoverType, optional bytes]
//! ]
//! ```
//!
//! The device-authentication structure that the holder signs is
//!
//! ```text
//! DeviceAuthentication = [
//!   "DeviceAuthentication",
//!   SessionTranscript,
//!   DocType,
//!   DeviceNameSpacesBytes,    // tag-24 wrapped device-signed namespaces
//! ]
//! ```
//!
//! and `DeviceAuthenticationBytes = #6.24(bstr .cbor DeviceAuthentication)`
//! is what the COSE_Sign1 detached payload commits to.

use ciborium::value::Value as CborValue;

use crate::error::MdlError;
use crate::mdoc::{TAG_ENCODED_CBOR, encode_cbor};

/// Session transcript per ISO 18013-5 §9.1.5.1.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionTranscript {
    /// Device-engagement bytes (tag-24 wrapped). `None` permitted for
    /// transports that elide engagement (e.g. some online flows).
    pub device_engagement_bytes: Option<Vec<u8>>,
    /// Reader ephemeral key bytes (tag-24 wrapped). `None` permitted.
    pub e_reader_key_bytes: Option<Vec<u8>>,
    /// Handover record: a CBOR value identifying the transport hand-off.
    pub handover: CborValue,
}

impl SessionTranscript {
    /// Construct a transcript for an OID4VP-style online flow where
    /// engagement is omitted and the handover is the `OID4VPHandover`
    /// (ISO 18013-7) hash of the authorisation request.
    pub fn for_oid4vp(
        client_id: &str,
        response_uri: &str,
        nonce: &str,
        mdoc_generated_nonce: &str,
    ) -> Self {
        // OID4VPHandover = [ clientIdHash, responseUriHash, nonce ]
        // hashes are SHA-256(clientId || mdocGeneratedNonce) per the ARF.
        use sha2::Digest;
        let mut h1 = sha2::Sha256::new();
        h1.update(client_id.as_bytes());
        h1.update(mdoc_generated_nonce.as_bytes());
        let client_id_hash = h1.finalize().to_vec();

        let mut h2 = sha2::Sha256::new();
        h2.update(response_uri.as_bytes());
        h2.update(mdoc_generated_nonce.as_bytes());
        let response_uri_hash = h2.finalize().to_vec();

        let handover = CborValue::Array(vec![
            CborValue::Bytes(client_id_hash),
            CborValue::Bytes(response_uri_hash),
            CborValue::Text(nonce.into()),
        ]);
        Self {
            device_engagement_bytes: None,
            e_reader_key_bytes: None,
            handover,
        }
    }

    /// Construct a transcript for an 18013-5 QR-code device-engagement.
    pub fn for_device_engagement(
        device_engagement_bytes: Vec<u8>,
        e_reader_key_bytes: Vec<u8>,
    ) -> Self {
        Self {
            device_engagement_bytes: Some(device_engagement_bytes),
            e_reader_key_bytes: Some(e_reader_key_bytes),
            // "QRHandover" is the empty array per ISO 18013-5 §9.1.5.1.
            handover: CborValue::Array(Vec::new()),
        }
    }

    /// Encode the transcript as the canonical 3-element array.
    pub fn to_value(&self) -> CborValue {
        let de = match &self.device_engagement_bytes {
            Some(b) => {
                CborValue::Tag(TAG_ENCODED_CBOR, Box::new(CborValue::Bytes(b.clone())))
            }
            None => CborValue::Null,
        };
        let er = match &self.e_reader_key_bytes {
            Some(b) => {
                CborValue::Tag(TAG_ENCODED_CBOR, Box::new(CborValue::Bytes(b.clone())))
            }
            None => CborValue::Null,
        };
        CborValue::Array(vec![de, er, self.handover.clone()])
    }

    /// Build the `DeviceAuthentication` byte string that the device
    /// signature commits to.
    ///
    /// `device_namespaces_bytes` is the inner CBOR bytes of the
    /// device-signed namespaces map (it is wrapped here as tag-24 inside
    /// the DeviceAuthentication tuple).
    pub fn device_authentication_bytes(
        &self,
        doc_type: &str,
        device_namespaces_bytes: &[u8],
    ) -> Result<Vec<u8>, MdlError> {
        let device_ns_tagged = CborValue::Tag(
            TAG_ENCODED_CBOR,
            Box::new(CborValue::Bytes(device_namespaces_bytes.to_vec())),
        );
        let inner = CborValue::Array(vec![
            CborValue::Text("DeviceAuthentication".into()),
            self.to_value(),
            CborValue::Text(doc_type.into()),
            device_ns_tagged,
        ]);
        let inner_bytes = encode_cbor(&inner)?;
        let tagged = CborValue::Tag(TAG_ENCODED_CBOR, Box::new(CborValue::Bytes(inner_bytes)));
        encode_cbor(&tagged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oid4vp_transcript_is_deterministic() {
        let a = SessionTranscript::for_oid4vp(
            "client",
            "https://verifier.example.com/cb",
            "nonce-abc",
            "mdoc-nonce-xyz",
        );
        let b = SessionTranscript::for_oid4vp(
            "client",
            "https://verifier.example.com/cb",
            "nonce-abc",
            "mdoc-nonce-xyz",
        );
        assert_eq!(a, b);
    }

    #[test]
    fn device_engagement_handover_is_empty_array() {
        let t = SessionTranscript::for_device_engagement(vec![1, 2], vec![3, 4]);
        match &t.handover {
            CborValue::Array(a) => assert!(a.is_empty()),
            _ => panic!("handover not array"),
        }
    }

    #[test]
    fn device_authentication_bytes_stable() {
        let t = SessionTranscript::for_oid4vp("c", "u", "n", "m");
        let a = t
            .device_authentication_bytes("org.iso.18013.5.1.mDL", b"ns-bytes")
            .unwrap();
        let b = t
            .device_authentication_bytes("org.iso.18013.5.1.mDL", b"ns-bytes")
            .unwrap();
        assert_eq!(a, b);
    }
}
