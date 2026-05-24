//! mDL / mDOC verifier.
//!
//! Verification steps (ISO 18013-5 §9.1.3):
//!
//! 1. Parse the `issuerAuth` COSE_Sign1 and re-derive its
//!    `Sig_structure`. Verify the signature against either an embedded
//!    `x5chain` certificate's subject public key or a trust anchor's
//!    raw public key.
//! 2. Decode the MSO from the COSE_Sign1 payload. Confirm
//!    `validityInfo` brackets the verifier's current time.
//! 3. For each revealed `IssuerSignedItem`, hash its tag-24 encoding
//!    and compare against the MSO digest under the matching `digestID`.
//!    Any mismatch fails the verification.
//! 4. If a `device_signed` half is present, reconstruct the
//!    `DeviceAuthentication` bytes from the supplied session transcript
//!    and verify the device COSE_Sign1 against the device public key
//!    listed in the MSO `deviceKeyInfo`.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use ciborium::value::Value as CborValue;
use p256::ecdsa::{
    Signature as P256Signature, VerifyingKey as P256VerifyingKey,
    signature::Verifier as _,
};
use sha2::{Digest, Sha256};

use crate::error::MdlError;
use crate::issuer::{COSE_ALG_ES256, MobileSecurityObject};
use crate::mdoc::{
    DeviceAuth, MobileDoc, decode_cbor, encode_cbor,
};
use crate::session_transcript::SessionTranscript;

/// Wall-clock provider — abstracted so tests can drive the verifier
/// at deterministic instants.
pub trait TimeProvider: Send + Sync {
    /// Current time, in UTC.
    fn now(&self) -> DateTime<Utc>;
}

/// `TimeProvider` backed by `chrono::Utc::now()`.
#[derive(Default, Debug)]
pub struct SystemTime;

impl TimeProvider for SystemTime {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// A fixed time for deterministic tests.
#[derive(Debug)]
pub struct FixedTime(pub DateTime<Utc>);

impl TimeProvider for FixedTime {
    fn now(&self) -> DateTime<Utc> {
        self.0
    }
}

/// Trust anchor — an issuer public key the verifier trusts directly.
///
/// In a full deployment a verifier would walk an X.509 chain; this
/// crate models trust as a flat list of accepted public keys keyed by
/// COSE algorithm. Production deployments can wrap a chain validator
/// around this primitive.
#[derive(Clone, Debug)]
pub struct TrustAnchor {
    /// COSE algorithm identifier (e.g. `-7` for ES256).
    pub alg: i64,
    /// Raw subject public key in SEC1 uncompressed form (P-256: 65 bytes
    /// starting with `0x04`).
    pub public_key_sec1: Vec<u8>,
    /// Optional name (free text) for logging.
    pub name: Option<String>,
}

impl TrustAnchor {
    /// Build a trust anchor for an ES256 raw P-256 verifying key.
    pub fn es256(vk: &P256VerifyingKey, name: Option<String>) -> Self {
        let sec1 = vk.to_sec1_bytes();
        Self {
            alg: COSE_ALG_ES256,
            public_key_sec1: sec1.to_vec(),
            name,
        }
    }

    /// Recover a P-256 VerifyingKey from this anchor.
    pub fn as_p256(&self) -> Result<P256VerifyingKey, MdlError> {
        if self.alg != COSE_ALG_ES256 {
            return Err(MdlError::UnsupportedAlg(format!(
                "trust anchor alg {} not ES256",
                self.alg
            )));
        }
        P256VerifyingKey::from_sec1_bytes(&self.public_key_sec1)
            .map_err(|e| MdlError::Signature(format!("anchor sec1: {e}")))
    }
}

/// Output of a successful verification: per-namespace revealed claims
/// keyed by `elementIdentifier`.
#[derive(Clone, Debug)]
pub struct VerifiedClaims {
    /// `docType` from the mdoc.
    pub doc_type: String,
    /// `namespace -> { element -> value }` — only the revealed items.
    pub claims: BTreeMap<String, BTreeMap<String, CborValue>>,
    /// Time at which the verifier accepted the credential.
    pub verified_at: DateTime<Utc>,
    /// Validity-info echoed from the MSO.
    pub validity_signed: DateTime<Utc>,
    /// Validity-info echoed from the MSO.
    pub validity_until: DateTime<Utc>,
}

/// Verifier handle.
pub struct Verifier {
    /// Trust anchors accepted for issuer COSE_Sign1.
    pub trusted_issuers: Vec<TrustAnchor>,
    /// Current-time source.
    pub time_provider: Arc<dyn TimeProvider>,
    /// Optional session transcript. Required if the mdoc carries a
    /// `device_signed` half; ignored otherwise.
    pub session_transcript: Option<SessionTranscript>,
}

impl Verifier {
    /// Build a verifier with explicit trust anchors and time source.
    pub fn new(
        trusted_issuers: Vec<TrustAnchor>,
        time_provider: Arc<dyn TimeProvider>,
    ) -> Self {
        Self {
            trusted_issuers,
            time_provider,
            session_transcript: None,
        }
    }

    /// Attach a session transcript. Required when the credential being
    /// verified includes a `device_signed` block.
    pub fn with_transcript(mut self, t: SessionTranscript) -> Self {
        self.session_transcript = Some(t);
        self
    }

    /// Run the full verification pipeline.
    pub fn verify(&self, doc: &MobileDoc) -> Result<VerifiedClaims, MdlError> {
        // 1. Verify issuer COSE_Sign1.
        let sign1 = &doc.issuer_signed.issuer_auth;
        let sig_input = sign1.sig_structure(&[])?;
        let alg = read_alg_from_protected(&sign1.protected)?;
        let mut accepted = false;
        for anchor in &self.trusted_issuers {
            if anchor.alg != alg {
                continue;
            }
            if verify_sig_with_anchor(anchor, &sig_input, &sign1.signature).is_ok() {
                accepted = true;
                break;
            }
        }
        if !accepted {
            return Err(MdlError::NoTrustAnchor);
        }

        // 2. Decode MSO and check validity.
        let mso = MobileSecurityObject::from_cbor(&sign1.payload)?;
        if mso.doc_type != doc.doc_type {
            return Err(MdlError::Type(format!(
                "MSO docType {} does not match mdoc docType {}",
                mso.doc_type, doc.doc_type
            )));
        }
        let now = self.time_provider.now();
        if now < mso.validity_info.valid_from || now > mso.validity_info.valid_until {
            return Err(MdlError::ValidityOutOfRange {
                now: now.to_rfc3339(),
                signed: mso.validity_info.signed.to_rfc3339(),
                valid_from: mso.validity_info.valid_from.to_rfc3339(),
                valid_until: mso.validity_info.valid_until.to_rfc3339(),
            });
        }

        // 3. Check digests for every revealed item.
        let mut revealed: BTreeMap<String, BTreeMap<String, CborValue>> =
            BTreeMap::new();
        for (ns, items) in &doc.issuer_signed.name_spaces {
            let ns_digests = mso.value_digests.get(ns).ok_or_else(|| {
                MdlError::Missing(format!("MSO valueDigests[{ns}]"))
            })?;
            let mut out: BTreeMap<String, CborValue> = BTreeMap::new();
            for item in items {
                let computed = Sha256::digest(item.to_tag24_bytes()?).to_vec();
                let expected =
                    ns_digests.get(&item.digest_id).ok_or_else(|| {
                        MdlError::Missing(format!(
                            "MSO valueDigests[{ns}][{}]",
                            item.digest_id
                        ))
                    })?;
                if computed != *expected {
                    return Err(MdlError::DigestMismatch {
                        namespace: ns.clone(),
                        element: item.element_identifier.clone(),
                    });
                }
                out.insert(item.element_identifier.clone(), item.element_value.clone());
            }
            revealed.insert(ns.clone(), out);
        }

        // 4. Verify device authentication, if present.
        if let Some(device_signed) = &doc.device_signed {
            let transcript = self.session_transcript.as_ref().ok_or_else(|| {
                MdlError::Missing("session transcript required to verify device_signed".into())
            })?;
            verify_device_auth(device_signed, &mso, transcript, &doc.doc_type)?;
            for (ns, m) in &device_signed.name_spaces {
                let entry = revealed.entry(ns.clone()).or_default();
                for (k, v) in m {
                    entry.insert(k.clone(), v.clone());
                }
            }
        }

        Ok(VerifiedClaims {
            doc_type: doc.doc_type.clone(),
            claims: revealed,
            verified_at: now,
            validity_signed: mso.validity_info.signed,
            validity_until: mso.validity_info.valid_until,
        })
    }
}

fn verify_device_auth(
    device_signed: &crate::mdoc::DeviceSigned,
    mso: &MobileSecurityObject,
    transcript: &SessionTranscript,
    doc_type: &str,
) -> Result<(), MdlError> {
    let sig = match &device_signed.device_auth {
        DeviceAuth::Signature(s) => s,
        DeviceAuth::Mac(_) => {
            return Err(MdlError::UnsupportedAlg(
                "device MAC authentication not implemented".into(),
            ));
        }
    };
    // Reconstruct device-signed namespaces bytes (must match what the
    // holder signed).
    let mut ns_entries: Vec<(CborValue, CborValue)> = Vec::new();
    for (ns, items) in &device_signed.name_spaces {
        let mut inner: Vec<(CborValue, CborValue)> = Vec::new();
        for (k, v) in items {
            inner.push((CborValue::Text(k.clone()), v.clone()));
        }
        ns_entries.push((CborValue::Text(ns.clone()), CborValue::Map(inner)));
    }
    let device_ns_bytes = encode_cbor(&CborValue::Map(ns_entries))?;
    let detached = transcript.device_authentication_bytes(doc_type, &device_ns_bytes)?;
    let sig_structure = CborValue::Array(vec![
        CborValue::Text("Signature1".into()),
        CborValue::Bytes(sig.protected.clone()),
        CborValue::Bytes(Vec::new()),
        CborValue::Bytes(detached),
    ]);
    let sig_input = encode_cbor(&sig_structure)?;
    let alg = read_alg_from_protected(&sig.protected)?;
    if alg != COSE_ALG_ES256 {
        return Err(MdlError::UnsupportedAlg(format!(
            "device alg {alg} (only ES256 implemented)"
        )));
    }
    let device_vk = device_key_p256_from_cose(&mso.device_key)?;
    let signature = P256Signature::from_slice(&sig.signature)
        .map_err(|e| MdlError::Signature(format!("device sig parse: {e}")))?;
    device_vk
        .verify(&sig_input, &signature)
        .map_err(|e| MdlError::Signature(format!("device verify: {e}")))?;
    Ok(())
}

fn device_key_p256_from_cose(v: &CborValue) -> Result<P256VerifyingKey, MdlError> {
    let entries = match v {
        CborValue::Map(m) => m,
        _ => return Err(MdlError::Type("deviceKey not a map".into())),
    };
    let mut x: Option<Vec<u8>> = None;
    let mut y: Option<Vec<u8>> = None;
    for (k, val) in entries {
        if let CborValue::Integer(i) = k {
            let n: i128 = (*i).into();
            match n {
                -2 => {
                    if let CborValue::Bytes(b) = val {
                        x = Some(b.clone());
                    }
                }
                -3 => {
                    if let CborValue::Bytes(b) = val {
                        y = Some(b.clone());
                    }
                }
                _ => {}
            }
        }
    }
    let x = x.ok_or_else(|| MdlError::Missing("deviceKey x".into()))?;
    let y = y.ok_or_else(|| MdlError::Missing("deviceKey y".into()))?;
    let mut sec1 = Vec::with_capacity(1 + x.len() + y.len());
    sec1.push(0x04);
    sec1.extend_from_slice(&x);
    sec1.extend_from_slice(&y);
    P256VerifyingKey::from_sec1_bytes(&sec1)
        .map_err(|e| MdlError::Signature(format!("device sec1: {e}")))
}

fn read_alg_from_protected(protected: &[u8]) -> Result<i64, MdlError> {
    if protected.is_empty() {
        return Err(MdlError::Missing("alg missing: protected empty".into()));
    }
    let v: CborValue = decode_cbor(protected)?;
    let entries = match v {
        CborValue::Map(m) => m,
        _ => {
            return Err(MdlError::Type(
                "protected not a map".into(),
            ));
        }
    };
    for (k, val) in entries {
        if let CborValue::Integer(i) = k {
            let n: i128 = i.into();
            if n == 1
                && let CborValue::Integer(ai) = val
            {
                let an: i128 = ai.into();
                return i64::try_from(an)
                    .map_err(|e| MdlError::Type(format!("alg int: {e}")));
            }
        }
    }
    Err(MdlError::Missing("alg".into()))
}

fn verify_sig_with_anchor(
    anchor: &TrustAnchor,
    sig_input: &[u8],
    signature_bytes: &[u8],
) -> Result<(), MdlError> {
    if anchor.alg != COSE_ALG_ES256 {
        return Err(MdlError::UnsupportedAlg(format!(
            "anchor alg {}",
            anchor.alg
        )));
    }
    let vk = anchor.as_p256()?;
    let signature = P256Signature::from_slice(signature_bytes)
        .map_err(|e| MdlError::Signature(format!("sig parse: {e}")))?;
    vk.verify(sig_input, &signature)
        .map_err(|e| MdlError::Signature(format!("verify: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::issuer::{Issuer, IssuerKey, ValidityInfo};
    use chrono::TimeZone;

    fn validity_window() -> ValidityInfo {
        ValidityInfo {
            signed: Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap(),
            valid_from: Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap(),
            valid_until: Utc.with_ymd_and_hms(2030, 5, 23, 12, 0, 0).unwrap(),
            expected_update: None,
        }
    }

    fn issue_simple() -> (MobileDoc, IssuerKey) {
        let issuer_key = IssuerKey::generate_es256();
        let device_key = IssuerKey::generate_es256();
        let issuer = Issuer::new(
            IssuerKey {
                alg: issuer_key.alg,
                p256: issuer_key.p256.clone(),
            },
            Vec::new(),
            validity_window(),
        );
        let mut ns: BTreeMap<String, BTreeMap<String, CborValue>> = BTreeMap::new();
        let mut inner: BTreeMap<String, CborValue> = BTreeMap::new();
        inner.insert("family_name".into(), CborValue::Text("Doe".into()));
        inner.insert("given_name".into(), CborValue::Text("Jane".into()));
        ns.insert(crate::namespace::NS_MDL.into(), inner);
        let doc = issuer
            .issue(
                "org.iso.18013.5.1.mDL",
                ns,
                device_key.cose_public_key(),
                b"seed",
            )
            .unwrap();
        (doc, issuer_key)
    }

    #[test]
    fn verifier_accepts_valid_doc() {
        let (doc, issuer_key) = issue_simple();
        let anchor = TrustAnchor::es256(issuer_key.p256.verifying_key(), None);
        let time = Arc::new(FixedTime(
            Utc.with_ymd_and_hms(2027, 1, 1, 0, 0, 0).unwrap(),
        ));
        let v = Verifier::new(vec![anchor], time);
        let claims = v.verify(&doc).unwrap();
        assert_eq!(claims.doc_type, "org.iso.18013.5.1.mDL");
        assert_eq!(
            claims.claims[crate::namespace::NS_MDL]["family_name"],
            CborValue::Text("Doe".into())
        );
    }

    #[test]
    fn verifier_rejects_wrong_anchor() {
        let (doc, _issuer_key) = issue_simple();
        let bogus = IssuerKey::generate_es256();
        let anchor = TrustAnchor::es256(bogus.p256.verifying_key(), None);
        let time = Arc::new(FixedTime(
            Utc.with_ymd_and_hms(2027, 1, 1, 0, 0, 0).unwrap(),
        ));
        let v = Verifier::new(vec![anchor], time);
        assert!(matches!(v.verify(&doc), Err(MdlError::NoTrustAnchor)));
    }
}
