//! mDL / mDOC issuance.
//!
//! The issuer constructs:
//!
//! 1. Random per-item salts and an `IssuerSignedItem` for every claim.
//! 2. A Mobile Security Object (MSO) committing to a SHA-256 digest of
//!    each item's tag-24 encoding, plus the device public key, validity
//!    window, and docType.
//! 3. A COSE_Sign1 over the bstr-encoded MSO, using ECDSA-with-SHA-256
//!    on the NIST P-256 curve (COSE alg `-7`, ES256). ES256 is the
//!    interoperable baseline for ISO 18013-5; ES384 and ES512 are
//!    permitted but rarely deployed in practice.
//!
//! The output of [`Issuer::issue`] is a [`MobileDoc`] with no
//! `device_signed` half. The holder later attaches a `device_signed`
//! via the [`crate::selective_disclosure`] module when presenting.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use ciborium::value::{Integer, Value as CborValue};
use p256::ecdsa::{
    SigningKey as P256SigningKey, VerifyingKey as P256VerifyingKey,
    signature::Signer,
};
use rand::TryRngCore;
use rand::rngs::OsRng;
use sha2::{Digest, Sha256};

use crate::error::MdlError;
use crate::mdoc::{
    CoseSign1, IssuerSigned, IssuerSignedItem, MobileDoc, encode_cbor,
};

/// COSE algorithm identifier for ECDSA-with-SHA-256 (RFC 9053).
pub const COSE_ALG_ES256: i64 = -7;

/// SHA-256 digest algorithm tag used in the MSO (ISO 18013-5 §9.1.2.4).
pub const DIGEST_ALG_SHA256: &str = "SHA-256";

/// MSO version constant (ISO 18013-5 §9.1.2.4).
pub const MSO_VERSION: &str = "1.0";

/// Validity-info block embedded in the MSO.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidityInfo {
    /// `signed` timestamp — when the issuer signed this MSO.
    pub signed: DateTime<Utc>,
    /// `validFrom` — earliest time the MSO is considered valid.
    pub valid_from: DateTime<Utc>,
    /// `validUntil` — latest time the MSO is considered valid.
    pub valid_until: DateTime<Utc>,
    /// Optional `expectedUpdate` hint.
    pub expected_update: Option<DateTime<Utc>>,
}

impl ValidityInfo {
    /// Encode as a CBOR map. Timestamps are tag-0 RFC 3339 text per
    /// ISO 18013-5 §9.1.2.4.
    pub fn to_value(&self) -> CborValue {
        let mut entries = vec![
            (
                CborValue::Text("signed".into()),
                CborValue::Tag(0, Box::new(CborValue::Text(rfc3339(&self.signed)))),
            ),
            (
                CborValue::Text("validFrom".into()),
                CborValue::Tag(
                    0,
                    Box::new(CborValue::Text(rfc3339(&self.valid_from))),
                ),
            ),
            (
                CborValue::Text("validUntil".into()),
                CborValue::Tag(
                    0,
                    Box::new(CborValue::Text(rfc3339(&self.valid_until))),
                ),
            ),
        ];
        if let Some(u) = &self.expected_update {
            entries.push((
                CborValue::Text("expectedUpdate".into()),
                CborValue::Tag(0, Box::new(CborValue::Text(rfc3339(u)))),
            ));
        }
        CborValue::Map(entries)
    }

    /// Parse from a CBOR map.
    pub fn from_value(v: &CborValue) -> Result<Self, MdlError> {
        let entries = match v {
            CborValue::Map(m) => m,
            _ => return Err(MdlError::Type("validityInfo not map".into())),
        };
        let mut signed: Option<DateTime<Utc>> = None;
        let mut valid_from: Option<DateTime<Utc>> = None;
        let mut valid_until: Option<DateTime<Utc>> = None;
        let mut expected_update: Option<DateTime<Utc>> = None;
        for (k, val) in entries {
            let CborValue::Text(name) = k else { continue };
            let parsed = extract_datetime(val)?;
            match name.as_str() {
                "signed" => signed = parsed,
                "validFrom" => valid_from = parsed,
                "validUntil" => valid_until = parsed,
                "expectedUpdate" => expected_update = parsed,
                _ => {}
            }
        }
        Ok(Self {
            signed: signed.ok_or_else(|| MdlError::Missing("signed".into()))?,
            valid_from: valid_from
                .ok_or_else(|| MdlError::Missing("validFrom".into()))?,
            valid_until: valid_until
                .ok_or_else(|| MdlError::Missing("validUntil".into()))?,
            expected_update,
        })
    }
}

fn rfc3339(dt: &DateTime<Utc>) -> String {
    dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn extract_datetime(v: &CborValue) -> Result<Option<DateTime<Utc>>, MdlError> {
    let inner = match v {
        CborValue::Tag(0, inner) => inner.as_ref(),
        other => other,
    };
    let text = match inner {
        CborValue::Text(s) => s.as_str(),
        _ => return Ok(None),
    };
    DateTime::parse_from_rfc3339(text)
        .map(|d| Some(d.with_timezone(&Utc)))
        .map_err(|e| MdlError::Type(format!("rfc3339 parse: {e}")))
}

/// Cryptographic configuration for issuance. Only ES256 is implemented;
/// the type carries the algorithm identifier so future ES384 / ES512
/// support can land without API churn.
#[derive(Clone, Debug)]
pub struct IssuerKey {
    /// COSE algorithm identifier.
    pub alg: i64,
    /// Raw P-256 signing key.
    pub p256: P256SigningKey,
}

impl IssuerKey {
    /// Generate a fresh ES256 issuer key.
    ///
    /// Reads 32 bytes from the OS RNG and tries to construct a P-256
    /// scalar from them. On the astronomically rare rejection (the
    /// 32-byte value is not in `[1, n-1]`) we retry. This routes
    /// around the `CryptoRng` version skew between `rand_core 0.10`
    /// (what p256 0.14-rc wants) and the `rand` 0.9 ecosystem.
    pub fn generate_es256() -> Self {
        let mut buf = [0u8; 32];
        loop {
            OsRng
                .try_fill_bytes(&mut buf)
                .expect("OS RNG must provide entropy");
            if let Ok(sk) = P256SigningKey::from_slice(&buf) {
                return Self {
                    alg: COSE_ALG_ES256,
                    p256: sk,
                };
            }
        }
    }

    /// Public key as a COSE_Key map (RFC 9052 §7.1, EC2 / P-256).
    pub fn cose_public_key(&self) -> CborValue {
        let vk: P256VerifyingKey = *self.p256.verifying_key();
        // SEC1 uncompressed = 0x04 || X || Y (65 bytes for P-256).
        let sec1 = vk.to_sec1_bytes();
        let (x, y) = if sec1.len() == 65 && sec1[0] == 0x04 {
            (sec1[1..33].to_vec(), sec1[33..65].to_vec())
        } else {
            (Vec::new(), Vec::new())
        };
        // COSE_Key:
        //  1 (kty)     = 2 (EC2)
        //  3 (alg)     = -7 (ES256)
        // -1 (crv)     = 1 (P-256)
        // -2 (x)       = bstr
        // -3 (y)       = bstr
        CborValue::Map(vec![
            (CborValue::Integer(Integer::from(1i64)), CborValue::Integer(Integer::from(2i64))),
            (CborValue::Integer(Integer::from(3i64)), CborValue::Integer(Integer::from(self.alg))),
            (CborValue::Integer(Integer::from(-1i64)), CborValue::Integer(Integer::from(1i64))),
            (CborValue::Integer(Integer::from(-2i64)), CborValue::Bytes(x)),
            (CborValue::Integer(Integer::from(-3i64)), CborValue::Bytes(y)),
        ])
    }
}

/// Issuer that mints `MobileDoc` credentials.
pub struct Issuer {
    /// Issuer signing key.
    pub key: IssuerKey,
    /// X.509 certificate chain (DER-encoded). The chain is carried in
    /// the COSE_Sign1 unprotected header as `x5chain` (label 33).
    pub certificate_chain: Vec<Vec<u8>>,
    /// Default validity window applied to every issued credential.
    pub validity_info: ValidityInfo,
}

impl Issuer {
    /// Build an issuer with the given key, X.509 chain, and validity window.
    pub fn new(
        key: IssuerKey,
        certificate_chain: Vec<Vec<u8>>,
        validity_info: ValidityInfo,
    ) -> Self {
        Self {
            key,
            certificate_chain,
            validity_info,
        }
    }

    /// Issue a fresh mdoc.
    ///
    /// * `doc_type` — typically `org.iso.18013.5.1.mDL`.
    /// * `claims` — `namespace -> { elementIdentifier -> elementValue }`.
    /// * `device_public_key_cose` — the holder's binding key, formatted
    ///   as a COSE_Key map. Verifiers will check that any device-signed
    ///   half was produced by this key.
    /// * `salt_seed` — optional deterministic seed. If non-empty, salts
    ///   are derived from `SHA-256(seed || ns || element || index)` for
    ///   reproducible test vectors. If empty, salts are sampled from the
    ///   OS RNG.
    pub fn issue(
        &self,
        doc_type: &str,
        claims: BTreeMap<String, BTreeMap<String, CborValue>>,
        device_public_key_cose: CborValue,
        salt_seed: &[u8],
    ) -> Result<MobileDoc, MdlError> {
        // Build IssuerSignedItems with deterministic digest IDs per namespace.
        let mut name_spaces: BTreeMap<String, Vec<IssuerSignedItem>> =
            BTreeMap::new();
        let mut value_digests: BTreeMap<String, BTreeMap<u64, Vec<u8>>> =
            BTreeMap::new();

        for (ns, items) in &claims {
            let mut ns_items: Vec<IssuerSignedItem> = Vec::with_capacity(items.len());
            let mut ns_digests: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
            for (idx, (name, value)) in items.iter().enumerate() {
                let id = idx as u64;
                let salt = if salt_seed.is_empty() {
                    let mut buf = [0u8; 32];
                    OsRng
                        .try_fill_bytes(&mut buf)
                        .map_err(|e| MdlError::Signature(format!("os rng: {e}")))?;
                    buf.to_vec()
                } else {
                    let mut h = Sha256::new();
                    h.update(salt_seed);
                    h.update(ns.as_bytes());
                    h.update(name.as_bytes());
                    h.update(id.to_be_bytes());
                    h.finalize().to_vec()
                };
                let item = IssuerSignedItem {
                    digest_id: id,
                    random: salt,
                    element_identifier: name.clone(),
                    element_value: value.clone(),
                };
                let tag24 = item.to_tag24_bytes()?;
                let digest = Sha256::digest(&tag24).to_vec();
                ns_digests.insert(id, digest);
                ns_items.push(item);
            }
            name_spaces.insert(ns.clone(), ns_items);
            value_digests.insert(ns.clone(), ns_digests);
        }

        let mso = build_mso(
            doc_type,
            &value_digests,
            &device_public_key_cose,
            &self.validity_info,
        )?;
        let mso_bytes = encode_cbor(&mso)?;

        // Build COSE_Sign1 with protected = {alg: ES256}, unprotected = {x5chain: ...}
        let protected_map = CborValue::Map(vec![(
            CborValue::Integer(Integer::from(1i64)), // alg label
            CborValue::Integer(Integer::from(self.key.alg)),
        )]);
        let protected = encode_cbor(&protected_map)?;
        let unprotected_map = if self.certificate_chain.is_empty() {
            CborValue::Map(Vec::new())
        } else if self.certificate_chain.len() == 1 {
            CborValue::Map(vec![(
                CborValue::Integer(Integer::from(33i64)), // x5chain label
                CborValue::Bytes(self.certificate_chain[0].clone()),
            )])
        } else {
            let arr: Vec<CborValue> = self
                .certificate_chain
                .iter()
                .map(|c| CborValue::Bytes(c.clone()))
                .collect();
            CborValue::Map(vec![(
                CborValue::Integer(Integer::from(33i64)),
                CborValue::Array(arr),
            )])
        };
        let unprotected = encode_cbor(&unprotected_map)?;

        let mut sign1 = CoseSign1 {
            protected,
            unprotected,
            payload: mso_bytes,
            signature: Vec::new(),
        };
        let sig_input = sign1.sig_structure(&[])?;
        let raw_sig: p256::ecdsa::Signature = self.key.p256.sign(&sig_input);
        sign1.signature = raw_sig.to_bytes().to_vec();

        Ok(MobileDoc {
            doc_type: doc_type.to_string(),
            issuer_signed: IssuerSigned {
                name_spaces,
                issuer_auth: sign1,
            },
            device_signed: None,
        })
    }
}

/// Construct the MobileSecurityObject CBOR value.
fn build_mso(
    doc_type: &str,
    value_digests: &BTreeMap<String, BTreeMap<u64, Vec<u8>>>,
    device_key: &CborValue,
    validity: &ValidityInfo,
) -> Result<CborValue, MdlError> {
    let mut digests_entries: Vec<(CborValue, CborValue)> = Vec::new();
    for (ns, items) in value_digests {
        let mut inner: Vec<(CborValue, CborValue)> = Vec::new();
        for (id, digest) in items {
            inner.push((
                CborValue::Integer(Integer::from(*id)),
                CborValue::Bytes(digest.clone()),
            ));
        }
        digests_entries.push((CborValue::Text(ns.clone()), CborValue::Map(inner)));
    }
    let device_key_info = CborValue::Map(vec![(
        CborValue::Text("deviceKey".into()),
        device_key.clone(),
    )]);
    Ok(CborValue::Map(vec![
        (
            CborValue::Text("version".into()),
            CborValue::Text(MSO_VERSION.into()),
        ),
        (
            CborValue::Text("digestAlgorithm".into()),
            CborValue::Text(DIGEST_ALG_SHA256.into()),
        ),
        (
            CborValue::Text("valueDigests".into()),
            CborValue::Map(digests_entries),
        ),
        (CborValue::Text("deviceKeyInfo".into()), device_key_info),
        (
            CborValue::Text("docType".into()),
            CborValue::Text(doc_type.into()),
        ),
        (
            CborValue::Text("validityInfo".into()),
            validity.to_value(),
        ),
    ]))
}

/// Parse the MSO out of an issuerAuth payload (the bstr-encoded inner).
#[derive(Clone, Debug)]
pub struct MobileSecurityObject {
    /// `version` string.
    pub version: String,
    /// `digestAlgorithm` (only `SHA-256` is supported).
    pub digest_algorithm: String,
    /// `valueDigests` map: namespace -> digestID -> digest bytes.
    pub value_digests: BTreeMap<String, BTreeMap<u64, Vec<u8>>>,
    /// Device public key as a COSE_Key map.
    pub device_key: CborValue,
    /// `docType`.
    pub doc_type: String,
    /// `validityInfo` block.
    pub validity_info: ValidityInfo,
}

impl MobileSecurityObject {
    /// Decode from the bstr payload of `issuerAuth`.
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, MdlError> {
        let v: CborValue = crate::mdoc::decode_cbor(bytes)?;
        Self::from_value(&v)
    }

    /// Decode from a CBOR value already parsed.
    pub fn from_value(v: &CborValue) -> Result<Self, MdlError> {
        let entries = match v {
            CborValue::Map(m) => m,
            _ => return Err(MdlError::Type("MSO not map".into())),
        };
        let mut version: Option<String> = None;
        let mut digest_algorithm: Option<String> = None;
        let mut value_digests: BTreeMap<String, BTreeMap<u64, Vec<u8>>> =
            BTreeMap::new();
        let mut device_key: Option<CborValue> = None;
        let mut doc_type: Option<String> = None;
        let mut validity_info: Option<ValidityInfo> = None;
        for (k, val) in entries {
            let CborValue::Text(name) = k else { continue };
            match name.as_str() {
                "version" => {
                    if let CborValue::Text(s) = val {
                        version = Some(s.clone());
                    }
                }
                "digestAlgorithm" => {
                    if let CborValue::Text(s) = val {
                        digest_algorithm = Some(s.clone());
                    }
                }
                "valueDigests" => {
                    let CborValue::Map(ns_map) = val else {
                        return Err(MdlError::Type(
                            "valueDigests not map".into(),
                        ));
                    };
                    for (ns_k, ns_v) in ns_map {
                        let CborValue::Text(ns_name) = ns_k else {
                            continue;
                        };
                        let CborValue::Map(inner) = ns_v else { continue };
                        let mut m: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
                        for (id_k, id_v) in inner {
                            let CborValue::Integer(id_i) = id_k else {
                                continue;
                            };
                            let id_n: i128 = (*id_i).into();
                            if id_n < 0 {
                                continue;
                            }
                            let CborValue::Bytes(b) = id_v else { continue };
                            m.insert(id_n as u64, b.clone());
                        }
                        value_digests.insert(ns_name.clone(), m);
                    }
                }
                "deviceKeyInfo" => {
                    let CborValue::Map(dki) = val else { continue };
                    for (dk, dv) in dki {
                        if let CborValue::Text(s) = dk
                            && s == "deviceKey"
                        {
                            device_key = Some(dv.clone());
                        }
                    }
                }
                "docType" => {
                    if let CborValue::Text(s) = val {
                        doc_type = Some(s.clone());
                    }
                }
                "validityInfo" => {
                    validity_info = Some(ValidityInfo::from_value(val)?);
                }
                _ => {}
            }
        }
        Ok(Self {
            version: version.ok_or_else(|| MdlError::Missing("version".into()))?,
            digest_algorithm: digest_algorithm
                .ok_or_else(|| MdlError::Missing("digestAlgorithm".into()))?,
            value_digests,
            device_key: device_key
                .ok_or_else(|| MdlError::Missing("deviceKey".into()))?,
            doc_type: doc_type
                .ok_or_else(|| MdlError::Missing("docType".into()))?,
            validity_info: validity_info
                .ok_or_else(|| MdlError::Missing("validityInfo".into()))?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn validity() -> ValidityInfo {
        ValidityInfo {
            signed: Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap(),
            valid_from: Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap(),
            valid_until: Utc.with_ymd_and_hms(2030, 5, 23, 12, 0, 0).unwrap(),
            expected_update: None,
        }
    }

    #[test]
    fn validity_info_round_trip() {
        let v = validity();
        let val = v.to_value();
        let back = ValidityInfo::from_value(&val).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn issuance_produces_valid_mdoc() {
        let key = IssuerKey::generate_es256();
        let device_key = IssuerKey::generate_es256();
        let issuer = Issuer::new(key, Vec::new(), validity());

        let mut ns: BTreeMap<String, BTreeMap<String, CborValue>> = BTreeMap::new();
        let mut inner: BTreeMap<String, CborValue> = BTreeMap::new();
        inner.insert(
            "family_name".into(),
            CborValue::Text("Doe".into()),
        );
        inner.insert(
            "given_name".into(),
            CborValue::Text("Jane".into()),
        );
        ns.insert(crate::namespace::NS_MDL.into(), inner);

        let doc = issuer
            .issue(
                "org.iso.18013.5.1.mDL",
                ns,
                device_key.cose_public_key(),
                b"test-seed",
            )
            .unwrap();
        let bytes = doc.to_cbor().unwrap();
        let back = MobileDoc::from_cbor(&bytes).unwrap();
        assert_eq!(back.doc_type, doc.doc_type);
        assert_eq!(
            back.issuer_signed.name_spaces.len(),
            doc.issuer_signed.name_spaces.len()
        );
        // MSO parses cleanly out of the COSE_Sign1 payload.
        let mso = MobileSecurityObject::from_cbor(&doc.issuer_signed.issuer_auth.payload)
            .unwrap();
        assert_eq!(mso.doc_type, doc.doc_type);
        assert_eq!(mso.digest_algorithm, "SHA-256");
        let item_count = doc.issuer_signed.name_spaces[crate::namespace::NS_MDL].len();
        let digest_count = mso.value_digests[crate::namespace::NS_MDL].len();
        assert_eq!(item_count, digest_count);
    }
}
