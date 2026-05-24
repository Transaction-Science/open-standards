//! Smart Byte cargo bridge for mDL / mDOC presentations.
//!
//! An mdoc presentation is packaged as `Cargo::Custom { type_uri,
//! body }` where:
//!
//! * `type_uri` is the constant [`MDL_CARGO_TYPE_URI`].
//! * `body` is the canonical CBOR encoding of a [`MobileDoc`].
//!
//! Because the envelope's SAID is BLAKE3 over the canonical CBOR of the
//! envelope (with the body an opaque byte string), the SAID is stable
//! across any round trip of the mdoc that preserves its byte image —
//! including the selective-disclosure projection (the projected mdoc
//! has a *different* SAID, because items were removed; an unprojected
//! round-trip preserves it).

use smart_byte_core::{
    Cargo, Envelope, JouleCost, OwnershipChain, Provenance, Said,
};

use crate::error::MdlError;
use crate::mdoc::MobileDoc;

/// Cargo `type_uri` constant for an mDL/mDOC payload.
pub const MDL_CARGO_TYPE_URI: &str = "urn:smart-byte:cargo:mdoc:v1";

/// Pack an mdoc into a Smart Byte envelope.
pub fn mdoc_envelope(
    doc: &MobileDoc,
    provenance: Provenance,
    ownership: OwnershipChain,
    joule_cost: JouleCost,
) -> Result<Envelope, MdlError> {
    let body = doc.to_cbor()?;
    let cargo = Cargo::Custom {
        type_uri: MDL_CARGO_TYPE_URI.into(),
        body,
    };
    Envelope::new(provenance, ownership, cargo, joule_cost)
        .map_err(|e| MdlError::Bridge(e.to_string()))
}

/// Extract an mdoc from an envelope.
pub fn mdoc_from_envelope(env: &Envelope) -> Result<MobileDoc, MdlError> {
    match &env.cargo {
        Cargo::Custom { type_uri, body } if type_uri == MDL_CARGO_TYPE_URI => {
            MobileDoc::from_cbor(body)
        }
        Cargo::Custom { type_uri, .. } => Err(MdlError::Bridge(format!(
            "unexpected cargo type_uri: {type_uri}"
        ))),
        other => Err(MdlError::Bridge(format!(
            "cargo kind {} is not an mdoc payload",
            other.kind()
        ))),
    }
}

/// Compute the SAID an envelope would have if an mdoc were stamped with
/// `provenance`, `ownership`, and `joule_cost`.
pub fn mdoc_said(
    doc: &MobileDoc,
    provenance: Provenance,
    ownership: OwnershipChain,
    joule_cost: JouleCost,
) -> Result<Said, MdlError> {
    let env = mdoc_envelope(doc, provenance, ownership, joule_cost)?;
    Ok(env.id)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::issuer::{Issuer, IssuerKey, ValidityInfo};
    use chrono::TimeZone;
    use ciborium::value::Value as CborValue;
    use smart_byte_core::JouleCost;

    fn issue_mdl() -> MobileDoc {
        let issuer_key = IssuerKey::generate_es256();
        let device_key = IssuerKey::generate_es256();
        let validity = ValidityInfo {
            signed: chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            valid_from: chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            valid_until: chrono::Utc.with_ymd_and_hms(2030, 1, 1, 0, 0, 0).unwrap(),
            expected_update: None,
        };
        let issuer = Issuer::new(issuer_key, Vec::new(), validity);
        let mut ns: BTreeMap<String, BTreeMap<String, CborValue>> = BTreeMap::new();
        let mut inner: BTreeMap<String, CborValue> = BTreeMap::new();
        inner.insert("family_name".into(), CborValue::Text("Doe".into()));
        ns.insert(crate::namespace::NS_MDL.into(), inner);
        issuer
            .issue(
                "org.iso.18013.5.1.mDL",
                ns,
                device_key.cose_public_key(),
                b"seed",
            )
            .unwrap()
    }

    fn provenance() -> Provenance {
        Provenance::new(
            Said::hash(b"mdl-issuer"),
            chrono::Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap(),
            b"x5chain".to_vec(),
        )
    }

    #[test]
    fn envelope_round_trip() {
        let doc = issue_mdl();
        let env = mdoc_envelope(
            &doc,
            provenance(),
            OwnershipChain::empty(),
            JouleCost::measured(11),
        )
        .unwrap();
        env.verify_said().unwrap();
        let back = mdoc_from_envelope(&env).unwrap();
        assert_eq!(back.doc_type, doc.doc_type);
        assert_eq!(
            back.issuer_signed.name_spaces.len(),
            doc.issuer_signed.name_spaces.len()
        );
    }

    #[test]
    fn said_stable_across_calls() {
        let doc = issue_mdl();
        let a = mdoc_said(
            &doc,
            provenance(),
            OwnershipChain::empty(),
            JouleCost::measured(11),
        )
        .unwrap();
        let b = mdoc_said(
            &doc,
            provenance(),
            OwnershipChain::empty(),
            JouleCost::measured(11),
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_wrong_cargo_kind() {
        let env = Envelope::new(
            provenance(),
            OwnershipChain::empty(),
            Cargo::Bytes(vec![1, 2, 3]),
            JouleCost::default(),
        )
        .unwrap();
        assert!(mdoc_from_envelope(&env).is_err());
    }
}
