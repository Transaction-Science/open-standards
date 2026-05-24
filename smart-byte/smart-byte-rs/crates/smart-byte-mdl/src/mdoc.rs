//! ISO/IEC 18013-5 `mdoc` data structures and CBOR codec.
//!
//! Wire layout (ISO/IEC 18013-5 §8.3.2.1.2.2):
//!
//! ```text
//! mdoc = {
//!   "docType":      tstr,
//!   "issuerSigned": IssuerSigned,
//!   ? "deviceSigned": DeviceSigned,
//! }
//! IssuerSigned = {
//!   "nameSpaces": { tstr => [+ #6.24(bstr .cbor IssuerSignedItem)] },
//!   "issuerAuth": COSE_Sign1, ; detached payload = bstr .cbor MSO
//! }
//! IssuerSignedItem = {
//!   "digestID":          uint,
//!   "random":            bstr (>= 16 bytes),
//!   "elementIdentifier": tstr,
//!   "elementValue":      any,
//! }
//! DeviceSigned = {
//!   "nameSpaces": #6.24(bstr .cbor { tstr => { tstr => any } }),
//!   "deviceAuth": { "deviceSignature": COSE_Sign1 / "deviceMac": COSE_Mac0 },
//! }
//! ```
//!
//! Tag 24 (`encoded-cbor`) wraps an inner CBOR encoding so the bytes
//! delivered to the digest function are stable: implementations must
//! hash the *exact* tag-24 bytes, not re-encode the inner item.

use std::collections::BTreeMap;

use ciborium::value::{Integer, Value as CborValue};
use serde::{Deserialize, Serialize};

use crate::error::MdlError;

/// CBOR tag 24 — "encoded-cbor data item" (RFC 8949 §3.4.5.1).
pub const TAG_ENCODED_CBOR: u64 = 24;

/// A single issuer-signed claim, plus its random salt and digest id.
///
/// The digest committed to in the MSO is `H(encoded-cbor(IssuerSignedItem))`
/// where the outer bytes are the *exact* tag-24 encoding of the item's
/// CBOR map.
#[derive(Clone, Debug, PartialEq)]
pub struct IssuerSignedItem {
    /// Index assigned by the issuer for this item within its namespace.
    /// The MSO indexes digests under this id.
    pub digest_id: u64,
    /// Random salt (>= 16 bytes per ISO 18013-5 §9.1.2.4).
    pub random: Vec<u8>,
    /// Element name (e.g. `family_name`).
    pub element_identifier: String,
    /// Element value (any CBOR type).
    pub element_value: CborValue,
}

impl IssuerSignedItem {
    /// Encode the item as a CBOR map.
    pub fn to_map(&self) -> CborValue {
        CborValue::Map(vec![
            (
                CborValue::Text("digestID".into()),
                CborValue::Integer(Integer::from(self.digest_id)),
            ),
            (
                CborValue::Text("random".into()),
                CborValue::Bytes(self.random.clone()),
            ),
            (
                CborValue::Text("elementIdentifier".into()),
                CborValue::Text(self.element_identifier.clone()),
            ),
            (
                CborValue::Text("elementValue".into()),
                self.element_value.clone(),
            ),
        ])
    }

    /// Parse from a CBOR map.
    pub fn from_map(value: &CborValue) -> Result<Self, MdlError> {
        let entries = match value {
            CborValue::Map(m) => m,
            _ => return Err(MdlError::Type("IssuerSignedItem not a map".into())),
        };
        let mut digest_id: Option<u64> = None;
        let mut random: Option<Vec<u8>> = None;
        let mut element_identifier: Option<String> = None;
        let mut element_value: Option<CborValue> = None;
        for (k, v) in entries {
            let CborValue::Text(name) = k else { continue };
            match name.as_str() {
                "digestID" => {
                    if let CborValue::Integer(i) = v {
                        let n: i128 = (*i).into();
                        if n < 0 {
                            return Err(MdlError::Type(
                                "digestID negative".into(),
                            ));
                        }
                        digest_id = Some(n as u64);
                    }
                }
                "random" => {
                    if let CborValue::Bytes(b) = v {
                        random = Some(b.clone());
                    }
                }
                "elementIdentifier" => {
                    if let CborValue::Text(s) = v {
                        element_identifier = Some(s.clone());
                    }
                }
                "elementValue" => element_value = Some(v.clone()),
                _ => {}
            }
        }
        Ok(Self {
            digest_id: digest_id
                .ok_or_else(|| MdlError::Missing("digestID".into()))?,
            random: random
                .ok_or_else(|| MdlError::Missing("random".into()))?,
            element_identifier: element_identifier
                .ok_or_else(|| MdlError::Missing("elementIdentifier".into()))?,
            element_value: element_value
                .ok_or_else(|| MdlError::Missing("elementValue".into()))?,
        })
    }

    /// Encode as the exact tag-24 bytes that the MSO digests over.
    ///
    /// Returns the *outer* tagged CBOR encoding `#6.24(bstr .cbor item)`.
    /// Verifiers MUST hash these exact bytes, not re-encode the inner.
    pub fn to_tag24_bytes(&self) -> Result<Vec<u8>, MdlError> {
        let inner = encode_cbor(&self.to_map())?;
        let tagged = CborValue::Tag(TAG_ENCODED_CBOR, Box::new(CborValue::Bytes(inner)));
        encode_cbor(&tagged)
    }

    /// Parse the inner item out of a tag-24 encoding.
    pub fn from_tag24_bytes(bytes: &[u8]) -> Result<Self, MdlError> {
        let v: CborValue = decode_cbor(bytes)?;
        let (tag, inner) = match v {
            CborValue::Tag(t, inner) => (t, *inner),
            _ => {
                return Err(MdlError::Type(
                    "expected tag-24 tagged value".into(),
                ));
            }
        };
        if tag != TAG_ENCODED_CBOR {
            return Err(MdlError::Type(format!(
                "expected tag 24, got {tag}"
            )));
        }
        let inner_bytes = match inner {
            CborValue::Bytes(b) => b,
            _ => {
                return Err(MdlError::Type(
                    "tag-24 content not bytes".into(),
                ));
            }
        };
        let inner_value: CborValue = decode_cbor(&inner_bytes)?;
        Self::from_map(&inner_value)
    }
}

/// Issuer-signed half of an mdoc.
#[derive(Clone, Debug, PartialEq)]
pub struct IssuerSigned {
    /// Map of namespace -> list of issuer-signed items.
    pub name_spaces: BTreeMap<String, Vec<IssuerSignedItem>>,
    /// Detached COSE_Sign1 — payload is the bstr-encoded MSO.
    pub issuer_auth: CoseSign1,
}

impl IssuerSigned {
    /// Encode as the canonical CBOR map.
    pub fn to_value(&self) -> Result<CborValue, MdlError> {
        let mut ns_entries: Vec<(CborValue, CborValue)> = Vec::new();
        for (ns, items) in &self.name_spaces {
            let mut arr: Vec<CborValue> = Vec::with_capacity(items.len());
            for item in items {
                let inner = encode_cbor(&item.to_map())?;
                arr.push(CborValue::Tag(
                    TAG_ENCODED_CBOR,
                    Box::new(CborValue::Bytes(inner)),
                ));
            }
            ns_entries.push((CborValue::Text(ns.clone()), CborValue::Array(arr)));
        }
        let map = vec![
            (CborValue::Text("nameSpaces".into()), CborValue::Map(ns_entries)),
            (CborValue::Text("issuerAuth".into()), self.issuer_auth.to_value()),
        ];
        Ok(CborValue::Map(map))
    }

    /// Decode from a CBOR map.
    pub fn from_value(v: &CborValue) -> Result<Self, MdlError> {
        let entries = match v {
            CborValue::Map(m) => m,
            _ => return Err(MdlError::Type("IssuerSigned not a map".into())),
        };
        let mut name_spaces: BTreeMap<String, Vec<IssuerSignedItem>> = BTreeMap::new();
        let mut issuer_auth: Option<CoseSign1> = None;
        for (k, val) in entries {
            let CborValue::Text(name) = k else { continue };
            match name.as_str() {
                "nameSpaces" => {
                    let CborValue::Map(ns_map) = val else {
                        return Err(MdlError::Type(
                            "nameSpaces not a map".into(),
                        ));
                    };
                    for (ns_key, items_val) in ns_map {
                        let CborValue::Text(ns_name) = ns_key else {
                            continue;
                        };
                        let CborValue::Array(items) = items_val else {
                            return Err(MdlError::Type(
                                "namespace items not an array".into(),
                            ));
                        };
                        let mut parsed = Vec::with_capacity(items.len());
                        for it in items {
                            let bytes = encode_cbor(it)?;
                            parsed.push(IssuerSignedItem::from_tag24_bytes(&bytes)?);
                        }
                        name_spaces.insert(ns_name.clone(), parsed);
                    }
                }
                "issuerAuth" => {
                    issuer_auth = Some(CoseSign1::from_value(val)?);
                }
                _ => {}
            }
        }
        Ok(Self {
            name_spaces,
            issuer_auth: issuer_auth
                .ok_or_else(|| MdlError::Missing("issuerAuth".into()))?,
        })
    }
}

/// Device-signed half of an mdoc.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceSigned {
    /// Device-signed namespaces (typically empty — most claims are issuer-signed).
    pub name_spaces: BTreeMap<String, BTreeMap<String, CborValue>>,
    /// Device authentication: either a signature (COSE_Sign1) or a MAC
    /// (COSE_Mac0). This crate ships the signature path.
    pub device_auth: DeviceAuth,
}

impl DeviceSigned {
    /// Encode as a CBOR map. The `nameSpaces` field is the standard
    /// tag-24-wrapped CBOR encoding for canonical hashing.
    pub fn to_value(&self) -> Result<CborValue, MdlError> {
        let mut ns_entries: Vec<(CborValue, CborValue)> = Vec::new();
        for (ns, items) in &self.name_spaces {
            let mut inner: Vec<(CborValue, CborValue)> = Vec::new();
            for (k, v) in items {
                inner.push((CborValue::Text(k.clone()), v.clone()));
            }
            ns_entries.push((CborValue::Text(ns.clone()), CborValue::Map(inner)));
        }
        let ns_bytes = encode_cbor(&CborValue::Map(ns_entries))?;
        let tagged = CborValue::Tag(
            TAG_ENCODED_CBOR,
            Box::new(CborValue::Bytes(ns_bytes)),
        );
        let map = vec![
            (CborValue::Text("nameSpaces".into()), tagged),
            (CborValue::Text("deviceAuth".into()), self.device_auth.to_value()),
        ];
        Ok(CborValue::Map(map))
    }

    /// Decode from a CBOR map.
    pub fn from_value(v: &CborValue) -> Result<Self, MdlError> {
        let entries = match v {
            CborValue::Map(m) => m,
            _ => return Err(MdlError::Type("DeviceSigned not a map".into())),
        };
        let mut name_spaces: BTreeMap<String, BTreeMap<String, CborValue>> =
            BTreeMap::new();
        let mut device_auth: Option<DeviceAuth> = None;
        for (k, val) in entries {
            let CborValue::Text(name) = k else { continue };
            match name.as_str() {
                "nameSpaces" => {
                    let (tag, inner) = match val {
                        CborValue::Tag(t, b) => (*t, b.as_ref().clone()),
                        _ => {
                            return Err(MdlError::Type(
                                "nameSpaces not tagged".into(),
                            ));
                        }
                    };
                    if tag != TAG_ENCODED_CBOR {
                        return Err(MdlError::Type(format!(
                            "deviceSigned.nameSpaces wrong tag {tag}"
                        )));
                    }
                    let bytes = match inner {
                        CborValue::Bytes(b) => b,
                        _ => {
                            return Err(MdlError::Type(
                                "tag-24 content not bytes".into(),
                            ));
                        }
                    };
                    let inner_value: CborValue = decode_cbor(&bytes)?;
                    let CborValue::Map(ns_map) = inner_value else {
                        return Err(MdlError::Type(
                            "deviceSigned namespaces inner not map".into(),
                        ));
                    };
                    for (ns_key, ns_val) in ns_map {
                        let CborValue::Text(ns_name) = ns_key else {
                            continue;
                        };
                        let CborValue::Map(items) = ns_val else {
                            continue;
                        };
                        let mut m: BTreeMap<String, CborValue> = BTreeMap::new();
                        for (k2, v2) in items {
                            if let CborValue::Text(s) = k2 {
                                m.insert(s, v2);
                            }
                        }
                        name_spaces.insert(ns_name, m);
                    }
                }
                "deviceAuth" => device_auth = Some(DeviceAuth::from_value(val)?),
                _ => {}
            }
        }
        Ok(Self {
            name_spaces,
            device_auth: device_auth
                .ok_or_else(|| MdlError::Missing("deviceAuth".into()))?,
        })
    }
}

/// Device-authentication payload. ISO 18013-5 §9.1.3 allows either a
/// COSE_Sign1 signature or a COSE_Mac0 MAC. We expose both shapes;
/// only the signature path is signed/verified by [`crate::verifier`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceAuth {
    /// COSE_Sign1 signature over the DeviceAuthentication structure.
    Signature(CoseSign1),
    /// COSE_Mac0 MAC over the DeviceAuthentication structure.
    Mac(CoseMac0),
}

impl DeviceAuth {
    /// Encode as a CBOR map.
    pub fn to_value(&self) -> CborValue {
        match self {
            DeviceAuth::Signature(s) => CborValue::Map(vec![(
                CborValue::Text("deviceSignature".into()),
                s.to_value(),
            )]),
            DeviceAuth::Mac(m) => CborValue::Map(vec![(
                CborValue::Text("deviceMac".into()),
                m.to_value(),
            )]),
        }
    }

    /// Parse from a CBOR map.
    pub fn from_value(v: &CborValue) -> Result<Self, MdlError> {
        let entries = match v {
            CborValue::Map(m) => m,
            _ => return Err(MdlError::Type("DeviceAuth not a map".into())),
        };
        for (k, val) in entries {
            if let CborValue::Text(name) = k {
                match name.as_str() {
                    "deviceSignature" => {
                        return Ok(DeviceAuth::Signature(CoseSign1::from_value(val)?));
                    }
                    "deviceMac" => {
                        return Ok(DeviceAuth::Mac(CoseMac0::from_value(val)?));
                    }
                    _ => {}
                }
            }
        }
        Err(MdlError::Missing("deviceAuth body".into()))
    }
}

/// Minimal COSE_Sign1 representation used by mdoc. The full COSE wire
/// form is a 4-tuple `[protected, unprotected, payload, signature]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoseSign1 {
    /// Protected header parameters, serialized as CBOR bytes (per RFC 9052).
    pub protected: Vec<u8>,
    /// Unprotected header parameters, encoded as a CBOR map's bytes.
    pub unprotected: Vec<u8>,
    /// Payload bytes (the bstr-encoded MSO for issuerAuth, or empty for
    /// detached signatures).
    pub payload: Vec<u8>,
    /// Raw signature bytes (algorithm-specific).
    pub signature: Vec<u8>,
}

impl CoseSign1 {
    /// Build the array CBOR value `[protected, unprotected, payload, signature]`.
    pub fn to_value(&self) -> CborValue {
        let unprotected_val = decode_cbor(&self.unprotected)
            .unwrap_or(CborValue::Map(Vec::new()));
        CborValue::Array(vec![
            CborValue::Bytes(self.protected.clone()),
            unprotected_val,
            CborValue::Bytes(self.payload.clone()),
            CborValue::Bytes(self.signature.clone()),
        ])
    }

    /// Parse from a CBOR array.
    pub fn from_value(v: &CborValue) -> Result<Self, MdlError> {
        let arr = match v {
            CborValue::Array(a) => a,
            _ => return Err(MdlError::Type("COSE_Sign1 not array".into())),
        };
        if arr.len() != 4 {
            return Err(MdlError::Type(format!(
                "COSE_Sign1 len {} (want 4)",
                arr.len()
            )));
        }
        let protected = match &arr[0] {
            CborValue::Bytes(b) => b.clone(),
            _ => return Err(MdlError::Type("protected not bstr".into())),
        };
        let unprotected = encode_cbor(&arr[1])?;
        let payload = match &arr[2] {
            CborValue::Bytes(b) => b.clone(),
            CborValue::Null => Vec::new(),
            _ => return Err(MdlError::Type("payload not bstr".into())),
        };
        let signature = match &arr[3] {
            CborValue::Bytes(b) => b.clone(),
            _ => return Err(MdlError::Type("signature not bstr".into())),
        };
        Ok(Self {
            protected,
            unprotected,
            payload,
            signature,
        })
    }

    /// Build the Sig_structure as defined in RFC 9052 §4.4. This is the
    /// exact byte string a signer must hash and sign.
    pub fn sig_structure(&self, external_aad: &[u8]) -> Result<Vec<u8>, MdlError> {
        let arr = CborValue::Array(vec![
            CborValue::Text("Signature1".into()),
            CborValue::Bytes(self.protected.clone()),
            CborValue::Bytes(external_aad.to_vec()),
            CborValue::Bytes(self.payload.clone()),
        ]);
        encode_cbor(&arr)
    }
}

/// Minimal COSE_Mac0 representation. Carried by mdocs that use MAC
/// device authentication; verification is out of scope for this crate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoseMac0 {
    /// Protected header bytes.
    pub protected: Vec<u8>,
    /// Unprotected header bytes.
    pub unprotected: Vec<u8>,
    /// MACed payload bytes (or empty if detached).
    pub payload: Vec<u8>,
    /// Tag bytes.
    pub tag: Vec<u8>,
}

impl CoseMac0 {
    /// Build the array CBOR value.
    pub fn to_value(&self) -> CborValue {
        let unprotected_val = decode_cbor(&self.unprotected)
            .unwrap_or(CborValue::Map(Vec::new()));
        CborValue::Array(vec![
            CborValue::Bytes(self.protected.clone()),
            unprotected_val,
            CborValue::Bytes(self.payload.clone()),
            CborValue::Bytes(self.tag.clone()),
        ])
    }

    /// Parse from a CBOR array.
    pub fn from_value(v: &CborValue) -> Result<Self, MdlError> {
        let arr = match v {
            CborValue::Array(a) => a,
            _ => return Err(MdlError::Type("COSE_Mac0 not array".into())),
        };
        if arr.len() != 4 {
            return Err(MdlError::Type(format!(
                "COSE_Mac0 len {} (want 4)",
                arr.len()
            )));
        }
        let protected = match &arr[0] {
            CborValue::Bytes(b) => b.clone(),
            _ => return Err(MdlError::Type("protected not bstr".into())),
        };
        let unprotected = encode_cbor(&arr[1])?;
        let payload = match &arr[2] {
            CborValue::Bytes(b) => b.clone(),
            CborValue::Null => Vec::new(),
            _ => return Err(MdlError::Type("payload not bstr".into())),
        };
        let tag = match &arr[3] {
            CborValue::Bytes(b) => b.clone(),
            _ => return Err(MdlError::Type("tag not bstr".into())),
        };
        Ok(Self {
            protected,
            unprotected,
            payload,
            tag,
        })
    }
}

/// A complete `mdoc` (ISO 18013-5 §8.3.2.1.2.2).
#[derive(Clone, Debug, PartialEq)]
pub struct MobileDoc {
    /// `docType`, e.g. `org.iso.18013.5.1.mDL`.
    pub doc_type: String,
    /// Issuer-signed component.
    pub issuer_signed: IssuerSigned,
    /// Optional device-signed component (omitted in an offline issuance
    /// bundle; present in a presentment).
    pub device_signed: Option<DeviceSigned>,
}

impl MobileDoc {
    /// Encode to canonical CBOR.
    pub fn to_cbor(&self) -> Result<Vec<u8>, MdlError> {
        let issuer_signed = self.issuer_signed.to_value()?;
        let mut entries = vec![
            (
                CborValue::Text("docType".into()),
                CborValue::Text(self.doc_type.clone()),
            ),
            (CborValue::Text("issuerSigned".into()), issuer_signed),
        ];
        if let Some(d) = &self.device_signed {
            entries.push((
                CborValue::Text("deviceSigned".into()),
                d.to_value()?,
            ));
        }
        encode_cbor(&CborValue::Map(entries))
    }

    /// Decode from canonical CBOR.
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, MdlError> {
        let v: CborValue = decode_cbor(bytes)?;
        let entries = match v {
            CborValue::Map(m) => m,
            _ => return Err(MdlError::Type("mdoc not a map".into())),
        };
        let mut doc_type: Option<String> = None;
        let mut issuer_signed: Option<IssuerSigned> = None;
        let mut device_signed: Option<DeviceSigned> = None;
        for (k, val) in entries {
            let CborValue::Text(name) = k else { continue };
            match name.as_str() {
                "docType" => {
                    if let CborValue::Text(s) = val {
                        doc_type = Some(s);
                    }
                }
                "issuerSigned" => {
                    issuer_signed = Some(IssuerSigned::from_value(&val)?);
                }
                "deviceSigned" => {
                    device_signed = Some(DeviceSigned::from_value(&val)?);
                }
                _ => {}
            }
        }
        Ok(Self {
            doc_type: doc_type
                .ok_or_else(|| MdlError::Missing("docType".into()))?,
            issuer_signed: issuer_signed
                .ok_or_else(|| MdlError::Missing("issuerSigned".into()))?,
            device_signed,
        })
    }
}

/// Encode a CBOR value with ciborium.
pub fn encode_cbor(value: &CborValue) -> Result<Vec<u8>, MdlError> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(value, &mut buf)
        .map_err(|e| MdlError::Cbor(e.to_string()))?;
    Ok(buf)
}

/// Decode a CBOR value with ciborium.
pub fn decode_cbor(bytes: &[u8]) -> Result<CborValue, MdlError> {
    ciborium::de::from_reader(bytes).map_err(|e| MdlError::Cbor(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_item(id: u64, name: &str, value: &str) -> IssuerSignedItem {
        IssuerSignedItem {
            digest_id: id,
            random: vec![0xAB; 16],
            element_identifier: name.into(),
            element_value: CborValue::Text(value.into()),
        }
    }

    #[test]
    fn tag24_round_trip() {
        let item = fixture_item(1, "family_name", "Doe");
        let bytes = item.to_tag24_bytes().unwrap();
        let back = IssuerSignedItem::from_tag24_bytes(&bytes).unwrap();
        assert_eq!(item, back);
    }

    #[test]
    fn issuer_signed_round_trip() {
        let mut ns: BTreeMap<String, Vec<IssuerSignedItem>> = BTreeMap::new();
        ns.insert(
            crate::namespace::NS_MDL.into(),
            vec![
                fixture_item(0, "family_name", "Doe"),
                fixture_item(1, "given_name", "Jane"),
            ],
        );
        let issuer_signed = IssuerSigned {
            name_spaces: ns,
            issuer_auth: CoseSign1 {
                protected: vec![0xA1, 0x01, 0x26],
                unprotected: encode_cbor(&CborValue::Map(Vec::new())).unwrap(),
                payload: vec![1, 2, 3, 4],
                signature: vec![9; 64],
            },
        };
        let v = issuer_signed.to_value().unwrap();
        let back = IssuerSigned::from_value(&v).unwrap();
        assert_eq!(back, issuer_signed);
    }

    #[test]
    fn mdoc_round_trip() {
        let mut ns: BTreeMap<String, Vec<IssuerSignedItem>> = BTreeMap::new();
        ns.insert(
            crate::namespace::NS_MDL.into(),
            vec![fixture_item(0, "family_name", "Doe")],
        );
        let doc = MobileDoc {
            doc_type: "org.iso.18013.5.1.mDL".into(),
            issuer_signed: IssuerSigned {
                name_spaces: ns,
                issuer_auth: CoseSign1 {
                    protected: vec![],
                    unprotected: encode_cbor(&CborValue::Map(Vec::new())).unwrap(),
                    payload: vec![],
                    signature: vec![],
                },
            },
            device_signed: None,
        };
        let bytes = doc.to_cbor().unwrap();
        let back = MobileDoc::from_cbor(&bytes).unwrap();
        assert_eq!(back, doc);
    }
}
