//! Salted-hash selective disclosure (ISO 18013-5 §9.1.2.4).
//!
//! The mDL native selective-disclosure primitive is:
//!
//! 1. Each `IssuerSignedItem` carries a random salt (`random` field).
//! 2. The MSO commits to `H(tag-24(IssuerSignedItem))` for every item.
//! 3. To present a subset, the holder includes only the issuer-signed
//!    items it wishes to reveal. The MSO digests for the withheld items
//!    remain in the MSO but with no corresponding revealed item — the
//!    verifier learns nothing about their value beyond the digest.
//!
//! This module also handles the device-signed half: the holder signs a
//! `DeviceAuthentication` structure over the session transcript with
//! the device key whose public part is bound by the MSO.

use std::collections::BTreeMap;

use ciborium::value::{Integer, Value as CborValue};
use p256::ecdsa::{SigningKey as P256SigningKey, signature::Signer};

use crate::error::MdlError;
use crate::issuer::COSE_ALG_ES256;
use crate::mdoc::{
    CoseSign1, DeviceAuth, DeviceSigned, IssuerSigned, MobileDoc, encode_cbor,
};
use crate::session_transcript::SessionTranscript;

/// Trait abstracting the device signer. Implementors hold the private
/// key whose public form is committed inside the MSO `deviceKeyInfo`.
pub trait DeviceSigner {
    /// COSE algorithm identifier (e.g. `-7` for ES256).
    fn alg(&self) -> i64;
    /// Sign the given bytes and return the raw signature (algorithm
    /// dependent — fixed-length r||s for ECDSA).
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, MdlError>;
}

/// P-256 (ES256) device signer.
pub struct P256DeviceSigner {
    /// The holder-bound P-256 signing key.
    pub key: P256SigningKey,
}

impl DeviceSigner for P256DeviceSigner {
    fn alg(&self) -> i64 {
        COSE_ALG_ES256
    }
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, MdlError> {
        let sig: p256::ecdsa::Signature = self.key.sign(message);
        Ok(sig.to_bytes().to_vec())
    }
}

/// Holder operation: produce a presented `MobileDoc` that reveals only
/// the requested namespace / element pairs and carries a device
/// signature bound to the session transcript.
///
/// * `doc` — the full issued credential (issuerAuth intact).
/// * `requested` — `namespace -> [element identifiers]`. Items not in
///   this map remain withheld; their digests are still committed in the
///   MSO so the verifier reconstructs the digest tree.
/// * `device_signer` — signer over the device-binding key.
/// * `transcript` — session transcript shared with the verifier.
/// * `device_namespaces` — optional device-signed claims (typically
///   empty in mDL; non-empty in some 18013-7 flows).
pub fn present(
    doc: &MobileDoc,
    requested: &BTreeMap<String, Vec<String>>,
    device_signer: &dyn DeviceSigner,
    transcript: &SessionTranscript,
    device_namespaces: BTreeMap<String, BTreeMap<String, CborValue>>,
) -> Result<MobileDoc, MdlError> {
    // Filter issuer-signed items to the requested subset, preserving order.
    let mut filtered: BTreeMap<String, Vec<crate::mdoc::IssuerSignedItem>> =
        BTreeMap::new();
    for (ns, items) in &doc.issuer_signed.name_spaces {
        let allow = requested.get(ns);
        let kept: Vec<_> = items
            .iter()
            .filter(|it| {
                allow
                    .map(|a| a.iter().any(|e| e == &it.element_identifier))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        if !kept.is_empty() {
            filtered.insert(ns.clone(), kept);
        }
    }

    // Encode the device-signed namespaces map; the tag-24 wrap is added
    // by the SessionTranscript helper when building DeviceAuthentication.
    let mut ns_entries: Vec<(CborValue, CborValue)> = Vec::new();
    for (ns, items) in &device_namespaces {
        let mut inner: Vec<(CborValue, CborValue)> = Vec::new();
        for (k, v) in items {
            inner.push((CborValue::Text(k.clone()), v.clone()));
        }
        ns_entries.push((CborValue::Text(ns.clone()), CborValue::Map(inner)));
    }
    let device_ns_bytes = encode_cbor(&CborValue::Map(ns_entries))?;
    let detached_payload =
        transcript.device_authentication_bytes(&doc.doc_type, &device_ns_bytes)?;

    // COSE_Sign1 with detached payload: payload field empty, but the
    // Sig_structure carries the DeviceAuthenticationBytes.
    let protected_map = CborValue::Map(vec![(
        CborValue::Integer(Integer::from(1i64)),
        CborValue::Integer(Integer::from(device_signer.alg())),
    )]);
    let protected = encode_cbor(&protected_map)?;
    let unprotected = encode_cbor(&CborValue::Map(Vec::new()))?;

    // Sig_structure has the detached bytes in the payload slot.
    let sig_structure = CborValue::Array(vec![
        CborValue::Text("Signature1".into()),
        CborValue::Bytes(protected.clone()),
        CborValue::Bytes(Vec::new()),
        CborValue::Bytes(detached_payload),
    ]);
    let sig_input = encode_cbor(&sig_structure)?;
    let signature = device_signer.sign(&sig_input)?;

    let device_signature = CoseSign1 {
        protected,
        unprotected,
        payload: Vec::new(),
        signature,
    };

    Ok(MobileDoc {
        doc_type: doc.doc_type.clone(),
        issuer_signed: IssuerSigned {
            name_spaces: filtered,
            issuer_auth: doc.issuer_signed.issuer_auth.clone(),
        },
        device_signed: Some(DeviceSigned {
            name_spaces: device_namespaces,
            device_auth: DeviceAuth::Signature(device_signature),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::issuer::{Issuer, IssuerKey, ValidityInfo};
    use chrono::{TimeZone, Utc};

    fn make_doc_and_device_key() -> (MobileDoc, P256SigningKey) {
        let issuer_key = IssuerKey::generate_es256();
        let device_key = IssuerKey::generate_es256();
        let validity = ValidityInfo {
            signed: Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap(),
            valid_from: Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap(),
            valid_until: Utc.with_ymd_and_hms(2030, 5, 23, 12, 0, 0).unwrap(),
            expected_update: None,
        };
        let issuer = Issuer::new(issuer_key, Vec::new(), validity);
        let mut ns: BTreeMap<String, BTreeMap<String, CborValue>> = BTreeMap::new();
        let mut inner: BTreeMap<String, CborValue> = BTreeMap::new();
        inner.insert("family_name".into(), CborValue::Text("Doe".into()));
        inner.insert("given_name".into(), CborValue::Text("Jane".into()));
        inner.insert(
            "birth_date".into(),
            CborValue::Text("1990-04-12".into()),
        );
        inner.insert("age_over_21".into(), CborValue::Bool(true));
        inner.insert(
            "document_number".into(),
            CborValue::Text("Z123456".into()),
        );
        ns.insert(crate::namespace::NS_MDL.into(), inner);
        let doc = issuer
            .issue(
                "org.iso.18013.5.1.mDL",
                ns,
                device_key.cose_public_key(),
                b"deterministic",
            )
            .unwrap();
        (doc, device_key.p256)
    }

    #[test]
    fn present_subset() {
        let (doc, dk) = make_doc_and_device_key();
        let signer = P256DeviceSigner { key: dk };
        let transcript =
            SessionTranscript::for_oid4vp("client", "https://x.example", "n", "m");
        let mut req: BTreeMap<String, Vec<String>> = BTreeMap::new();
        req.insert(
            crate::namespace::NS_MDL.into(),
            vec!["family_name".into(), "age_over_21".into()],
        );
        let presented =
            present(&doc, &req, &signer, &transcript, BTreeMap::new()).unwrap();
        let items =
            &presented.issuer_signed.name_spaces[crate::namespace::NS_MDL];
        assert_eq!(items.len(), 2);
        assert!(
            items
                .iter()
                .any(|i| i.element_identifier == "family_name")
        );
        assert!(items.iter().any(|i| i.element_identifier == "age_over_21"));
        assert!(presented.device_signed.is_some());
    }
}
