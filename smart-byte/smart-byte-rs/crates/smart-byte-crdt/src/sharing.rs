//! Canonical sharing patterns.
//!
//! [`ShareSimplex`] is one-writer-many-readers: the writer batches deltas
//! and emits Smart Byte envelopes; readers apply them.
//!
//! [`ShareMultiplex`] is many-writers: every writer can both emit deltas
//! and apply other writers' deltas. CRDT convergence guarantees identical
//! eventual state.

use smart_byte_core::Cargo;

use crate::cargo_bridge::{cargo_to_payload, delta_to_cargo, CrdtCargoPayload};
use crate::document::CrdtDocument;
use crate::error::Result;
use crate::hlc::ReplicaId;
use crate::ops::Op;
use crate::sync::{apply, clock_of, diff};
use crate::vector_clock::VectorClock;

/// One-writer-many-readers helper.
#[derive(Clone, Debug)]
pub struct ShareSimplex {
    pub writer: ReplicaId,
    pub doc: CrdtDocument,
}

impl ShareSimplex {
    /// Construct a Simplex share over `doc`.
    pub fn new(writer: ReplicaId, doc: CrdtDocument) -> Self {
        Self { writer, doc }
    }

    /// Produce a delta cargo for a reader that has `reader_clock`.
    pub fn delta_for(&self, reader_clock: &VectorClock) -> Result<Cargo> {
        let ops = diff(&self.doc, reader_clock);
        delta_to_cargo(&ops)
    }

    /// Apply an inbound cargo (snapshot or delta) into the reader's local
    /// document. Returns the number of new ops applied.
    pub fn apply_inbound(reader: &mut CrdtDocument, cargo: &Cargo) -> Result<usize> {
        match cargo_to_payload(cargo)? {
            CrdtCargoPayload::Snapshot(doc) => {
                reader.merge(&doc)?;
                Ok(doc.history.len())
            }
            CrdtCargoPayload::Delta(ops) => apply(reader, &ops),
        }
    }
}

/// Many-writers helper.
#[derive(Clone, Debug)]
pub struct ShareMultiplex {
    pub local: ReplicaId,
    pub doc: CrdtDocument,
}

impl ShareMultiplex {
    /// Construct a Multiplex share over `doc`.
    pub fn new(local: ReplicaId, doc: CrdtDocument) -> Self {
        Self { local, doc }
    }

    /// Cargo containing all ops the peer at `peer_clock` is missing.
    pub fn delta_for(&self, peer_clock: &VectorClock) -> Result<Cargo> {
        let ops = diff(&self.doc, peer_clock);
        delta_to_cargo(&ops)
    }

    /// Apply an inbound cargo into the local document.
    pub fn apply_inbound(&mut self, cargo: &Cargo) -> Result<usize> {
        ShareSimplex::apply_inbound(&mut self.doc, cargo)
    }

    /// Current observed clock — pass this to peers so they can compute
    /// the minimum delta they need to send.
    pub fn clock(&self) -> VectorClock {
        clock_of(&self.doc)
    }

    /// Append a local op (already applied) to the doc's history. Use this
    /// only when an op is constructed externally; the document helpers
    /// already push to history.
    pub fn record_local(&mut self, op: Op) -> Result<()> {
        self.doc.apply_op(&op)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::DocumentId;
    use crate::hlc::HlcClock;

    fn rid(n: u128) -> ReplicaId {
        ReplicaId::new(n)
    }

    #[test]
    fn simplex_writer_reader_converge() {
        let id = DocumentId::from_bytes(b"d");
        let w = rid(1);
        let mut writer_doc = CrdtDocument::new(id, w);
        let mut clock = HlcClock::with_manual_wall(w, 1);
        writer_doc.set_at("/x", 1, &mut clock).unwrap();
        writer_doc.set_at("/y", 2, &mut clock).unwrap();

        let writer = ShareSimplex::new(w, writer_doc);
        let mut reader_doc = CrdtDocument::new(id, rid(2));
        let cargo = writer.delta_for(&VectorClock::new()).unwrap();
        let n = ShareSimplex::apply_inbound(&mut reader_doc, &cargo).unwrap();
        assert_eq!(n, 2);
        assert!(reader_doc.get_node("/x").is_some());
        assert!(reader_doc.get_node("/y").is_some());
    }

    #[test]
    fn multiplex_converges_after_exchange() {
        let id = DocumentId::from_bytes(b"d");
        let mut a_clock = HlcClock::with_manual_wall(rid(1), 1);
        let mut b_clock = HlcClock::with_manual_wall(rid(2), 1);

        let mut a = ShareMultiplex::new(rid(1), CrdtDocument::new(id, rid(1)));
        let mut b = ShareMultiplex::new(rid(2), CrdtDocument::new(id, rid(2)));

        a.doc.set_at("/from_a", 11, &mut a_clock).unwrap();
        b.doc.set_at("/from_b", 22, &mut b_clock).unwrap();

        let from_a = a.delta_for(&b.clock()).unwrap();
        let from_b = b.delta_for(&a.clock()).unwrap();
        a.apply_inbound(&from_b).unwrap();
        b.apply_inbound(&from_a).unwrap();

        assert!(a.doc.get_node("/from_b").is_some());
        assert!(b.doc.get_node("/from_a").is_some());
    }
}
