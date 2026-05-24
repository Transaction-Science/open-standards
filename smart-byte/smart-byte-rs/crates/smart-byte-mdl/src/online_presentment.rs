//! ISO/IEC 18013-7 online presentment.
//!
//! In the online flow the reader (a "Relying Party") POSTs an
//! authorisation request — typically OpenID4VP — to the wallet, which
//! responds with one or more mdoc presentations packaged in a CBOR
//! `DeviceResponse`. The wallet binds the response to the request
//! through the OID4VP session transcript ([`SessionTranscript::for_oid4vp`]).
//!
//! This module ships the wire types and the reader-facing helpers
//! needed to compose and parse these messages. Transport (HTTPS, the
//! OID4VP authorization request format) is intentionally out of scope —
//! a downstream crate plugs in `reqwest` (or `wasm-bindgen`) and the
//! exact `presentation_definition` shape required by the verifier.

use std::collections::BTreeMap;

use ciborium::value::{Integer, Value as CborValue};

use crate::error::MdlError;
use crate::mdoc::{MobileDoc, decode_cbor, encode_cbor};

/// Reader request for a set of namespaces and elements.
///
/// In the OID4VP flow this is conveyed inside a `presentation_definition`
/// JSON document; in the offline flow it is conveyed as a CBOR
/// `ItemsRequest`. We use the same in-memory shape for both.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ItemsRequest {
    /// `docType` requested.
    pub doc_type: String,
    /// Per-namespace request — `namespace -> { element -> intent_to_retain }`.
    pub name_spaces: BTreeMap<String, BTreeMap<String, bool>>,
}

impl ItemsRequest {
    /// Encode as the ISO 18013-5 CBOR ItemsRequest map.
    pub fn to_cbor(&self) -> Result<Vec<u8>, MdlError> {
        let mut ns_entries: Vec<(CborValue, CborValue)> = Vec::new();
        for (ns, items) in &self.name_spaces {
            let mut inner: Vec<(CborValue, CborValue)> = Vec::new();
            for (el, retain) in items {
                inner.push((
                    CborValue::Text(el.clone()),
                    CborValue::Bool(*retain),
                ));
            }
            ns_entries.push((CborValue::Text(ns.clone()), CborValue::Map(inner)));
        }
        let map = CborValue::Map(vec![
            (
                CborValue::Text("docType".into()),
                CborValue::Text(self.doc_type.clone()),
            ),
            (
                CborValue::Text("nameSpaces".into()),
                CborValue::Map(ns_entries),
            ),
        ]);
        encode_cbor(&map)
    }

    /// Decode from CBOR bytes.
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, MdlError> {
        let v: CborValue = decode_cbor(bytes)?;
        let entries = match v {
            CborValue::Map(m) => m,
            _ => return Err(MdlError::Type("ItemsRequest not map".into())),
        };
        let mut doc_type: Option<String> = None;
        let mut name_spaces: BTreeMap<String, BTreeMap<String, bool>> =
            BTreeMap::new();
        for (k, val) in entries {
            let CborValue::Text(name) = k else { continue };
            match name.as_str() {
                "docType" => {
                    if let CborValue::Text(s) = val {
                        doc_type = Some(s);
                    }
                }
                "nameSpaces" => {
                    if let CborValue::Map(m) = val {
                        for (k2, v2) in m {
                            if let (CborValue::Text(ns), CborValue::Map(inner)) =
                                (k2, v2)
                            {
                                let mut out: BTreeMap<String, bool> = BTreeMap::new();
                                for (ek, ev) in inner {
                                    if let (
                                        CborValue::Text(e),
                                        CborValue::Bool(r),
                                    ) = (ek, ev)
                                    {
                                        out.insert(e, r);
                                    }
                                }
                                name_spaces.insert(ns, out);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(Self {
            doc_type: doc_type
                .ok_or_else(|| MdlError::Missing("ItemsRequest docType".into()))?,
            name_spaces,
        })
    }

    /// Convert the request into the namespace/element selection map the
    /// holder uses with [`crate::selective_disclosure::present`].
    pub fn requested(&self) -> BTreeMap<String, Vec<String>> {
        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (ns, items) in &self.name_spaces {
            out.insert(ns.clone(), items.keys().cloned().collect());
        }
        out
    }
}

/// `DeviceResponse` wire type (ISO 18013-5 §8.3.2.1.2.1).
///
/// A `DeviceResponse` may carry multiple documents (e.g. an mDL and a
/// university card). Each document is a fully-formed [`MobileDoc`].
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceResponse {
    /// Spec version. Always `"1.0"`.
    pub version: String,
    /// One or more presented documents.
    pub documents: Vec<MobileDoc>,
    /// `status` integer — `0` means success, non-zero is an error code.
    pub status: u64,
}

impl DeviceResponse {
    /// Build a successful response carrying one or more documents.
    pub fn ok(documents: Vec<MobileDoc>) -> Self {
        Self {
            version: "1.0".into(),
            documents,
            status: 0,
        }
    }

    /// Encode to canonical CBOR.
    pub fn to_cbor(&self) -> Result<Vec<u8>, MdlError> {
        let mut docs: Vec<CborValue> = Vec::with_capacity(self.documents.len());
        for d in &self.documents {
            docs.push(decode_cbor(&d.to_cbor()?)?);
        }
        let map = CborValue::Map(vec![
            (
                CborValue::Text("version".into()),
                CborValue::Text(self.version.clone()),
            ),
            (
                CborValue::Text("documents".into()),
                CborValue::Array(docs),
            ),
            (
                CborValue::Text("status".into()),
                CborValue::Integer(Integer::from(self.status)),
            ),
        ]);
        encode_cbor(&map)
    }

    /// Decode from CBOR bytes.
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, MdlError> {
        let v: CborValue = decode_cbor(bytes)?;
        let entries = match v {
            CborValue::Map(m) => m,
            _ => return Err(MdlError::Type("DeviceResponse not map".into())),
        };
        let mut version: Option<String> = None;
        let mut documents: Vec<MobileDoc> = Vec::new();
        let mut status: u64 = 0;
        for (k, val) in entries {
            let CborValue::Text(name) = k else { continue };
            match name.as_str() {
                "version" => {
                    if let CborValue::Text(s) = val {
                        version = Some(s);
                    }
                }
                "documents" => {
                    if let CborValue::Array(arr) = val {
                        for doc_val in arr {
                            let bytes = encode_cbor(&doc_val)?;
                            documents.push(MobileDoc::from_cbor(&bytes)?);
                        }
                    }
                }
                "status" => {
                    if let CborValue::Integer(i) = val {
                        let n: i128 = i.into();
                        if n >= 0 {
                            status = n as u64;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(Self {
            version: version.unwrap_or_else(|| "1.0".into()),
            documents,
            status,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_request_round_trip() {
        let mut ns: BTreeMap<String, BTreeMap<String, bool>> = BTreeMap::new();
        let mut inner: BTreeMap<String, bool> = BTreeMap::new();
        inner.insert("family_name".into(), false);
        inner.insert("age_over_21".into(), true);
        ns.insert(crate::namespace::NS_MDL.into(), inner);
        let req = ItemsRequest {
            doc_type: "org.iso.18013.5.1.mDL".into(),
            name_spaces: ns,
        };
        let bytes = req.to_cbor().unwrap();
        let back = ItemsRequest::from_cbor(&bytes).unwrap();
        assert_eq!(back, req);
    }

    #[test]
    fn requested_selection_translates() {
        let mut ns: BTreeMap<String, BTreeMap<String, bool>> = BTreeMap::new();
        let mut inner: BTreeMap<String, bool> = BTreeMap::new();
        inner.insert("family_name".into(), false);
        ns.insert(crate::namespace::NS_MDL.into(), inner);
        let req = ItemsRequest {
            doc_type: "org.iso.18013.5.1.mDL".into(),
            name_spaces: ns,
        };
        let sel = req.requested();
        assert_eq!(sel[crate::namespace::NS_MDL], vec!["family_name"]);
    }
}
