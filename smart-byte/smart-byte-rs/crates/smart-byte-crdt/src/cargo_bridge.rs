//! Smart Byte cargo bridge.
//!
//! Wraps a [`CrdtDocument`] (full snapshot) or a batch of [`Op`]s
//! (delta) as a `Cargo::Custom` payload. The SAID over the resulting
//! envelope is stable: identical document state → identical CBOR →
//! identical SAID.

use serde::{Deserialize, Serialize};
use smart_byte_core::Cargo;

use crate::document::CrdtDocument;
use crate::error::Result;
use crate::ops::Op;

/// The canonical URI for full CRDT snapshots wrapped in `Cargo::Custom`.
pub const CRDT_SNAPSHOT_TYPE_URI: &str = "urn:smart-byte:cargo:crdt:v1";

/// The canonical URI for CRDT op-batches wrapped in `Cargo::Custom`.
pub const CRDT_DELTA_TYPE_URI: &str = "urn:smart-byte:cargo:crdt-delta:v1";

/// Envelope-ready encoding of a CRDT payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "body")]
pub enum CrdtCargoPayload {
    Snapshot(CrdtDocument),
    Delta(Vec<Op>),
}

impl CrdtCargoPayload {
    /// Encode as canonical CBOR.
    pub fn to_cbor(&self) -> Result<Vec<u8>> {
        Ok(serde_cbor::to_vec(self)?)
    }
    /// Decode from canonical CBOR.
    pub fn from_cbor(bytes: &[u8]) -> Result<Self> {
        Ok(serde_cbor::from_slice(bytes)?)
    }
}

/// Wrap a full snapshot as Smart Byte `Cargo::Custom`.
pub fn snapshot_to_cargo(doc: &CrdtDocument) -> Result<Cargo> {
    let payload = CrdtCargoPayload::Snapshot(doc.clone());
    Ok(Cargo::Custom {
        type_uri: CRDT_SNAPSHOT_TYPE_URI.into(),
        body: payload.to_cbor()?,
    })
}

/// Wrap a delta (batch of ops) as Smart Byte `Cargo::Custom`.
pub fn delta_to_cargo(ops: &[Op]) -> Result<Cargo> {
    let payload = CrdtCargoPayload::Delta(ops.to_vec());
    Ok(Cargo::Custom {
        type_uri: CRDT_DELTA_TYPE_URI.into(),
        body: payload.to_cbor()?,
    })
}

/// Decode a Smart Byte cargo back to a CRDT payload, validating the
/// URI matches one of the canonical strings.
pub fn cargo_to_payload(cargo: &Cargo) -> Result<CrdtCargoPayload> {
    match cargo {
        Cargo::Custom { type_uri, body } => {
            if type_uri != CRDT_SNAPSHOT_TYPE_URI && type_uri != CRDT_DELTA_TYPE_URI {
                return Err(crate::error::CrdtError::OpIntegrity(format!(
                    "unrecognised cargo type_uri: {type_uri}"
                )));
            }
            CrdtCargoPayload::from_cbor(body)
        }
        other => Err(crate::error::CrdtError::OpIntegrity(format!(
            "expected Cargo::Custom, got {}",
            other.kind()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{CrdtDocument, DocumentId};
    use crate::hlc::{HlcClock, ReplicaId};
    use chrono::TimeZone;
    use smart_byte_core::{Envelope, JouleCost, OwnershipChain, Provenance, Said};

    fn fixture_doc() -> CrdtDocument {
        let id = DocumentId::from_bytes(b"d");
        let r = ReplicaId::new(1);
        let mut doc = CrdtDocument::new(id, r);
        let mut clock = HlcClock::with_manual_wall(r, 1);
        doc.set_at("/users/alice/balance", 100, &mut clock).unwrap();
        doc
    }

    #[test]
    fn snapshot_round_trip_through_cargo() {
        let doc = fixture_doc();
        let cargo = snapshot_to_cargo(&doc).unwrap();
        let back = cargo_to_payload(&cargo).unwrap();
        match back {
            CrdtCargoPayload::Snapshot(d) => assert_eq!(d.id, doc.id),
            _ => panic!("expected snapshot"),
        }
    }

    #[test]
    fn envelope_said_is_stable() {
        let doc = fixture_doc();
        let cargo = snapshot_to_cargo(&doc).unwrap();
        let prov = Provenance::new(
            Said::hash(b"issuer"),
            chrono::Utc.with_ymd_and_hms(2026, 5, 23, 12, 0, 0).unwrap(),
            vec![],
        );
        let env1 = Envelope::new(
            prov.clone(),
            OwnershipChain::empty(),
            cargo.clone(),
            JouleCost::default(),
        )
        .unwrap();
        let env2 = Envelope::new(
            prov,
            OwnershipChain::empty(),
            cargo,
            JouleCost::default(),
        )
        .unwrap();
        assert_eq!(env1.id, env2.id);
        env1.verify_said().unwrap();
    }
}
