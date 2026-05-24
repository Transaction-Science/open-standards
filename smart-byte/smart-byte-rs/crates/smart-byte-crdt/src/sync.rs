//! Delta computation and idempotent op application.
//!
//! Two helpers:
//!
//! * [`diff`] inspects a local document and a remote [`VectorClock`] and
//!   returns the operations the remote is missing.
//! * [`apply`] feeds a batch of operations into a document, applying
//!   them in HLC order, deduplicating by op id.

use crate::document::CrdtDocument;
use crate::error::Result;
use crate::ops::Op;
use crate::vector_clock::VectorClock;

/// Return ops in `local`'s history that the remote (at `remote_clock`)
/// hasn't yet seen.
pub fn diff(local: &CrdtDocument, remote_clock: &VectorClock) -> Vec<Op> {
    // We assign a per-replica monotonic sequence by counting the local
    // appearance order of each replica's ops in the log.
    let mut counters: std::collections::HashMap<crate::hlc::ReplicaId, u64> =
        std::collections::HashMap::new();
    let mut out = Vec::new();
    for op in &local.history {
        let entry = counters.entry(op.replica).or_insert(0);
        *entry += 1;
        let local_seq = *entry;
        let remote_seq = remote_clock.get(op.replica);
        if local_seq > remote_seq {
            out.push(op.clone());
        }
    }
    out.sort_by_key(|a| a.hlc);
    out
}

/// Apply a batch of ops to `doc`. Ordering by HLC; duplicates ignored.
/// Idempotent.
pub fn apply(doc: &mut CrdtDocument, ops: &[Op]) -> Result<usize> {
    let mut sorted: Vec<Op> = ops.to_vec();
    sorted.sort_by_key(|a| a.hlc);
    let mut applied = 0usize;
    for op in &sorted {
        if doc.apply_op(op)? {
            applied += 1;
        }
    }
    Ok(applied)
}

/// Build a [`VectorClock`] reflecting how many of each replica's ops
/// `doc` currently has in its history. Useful as the argument to
/// [`diff`] when synchronising.
pub fn clock_of(doc: &CrdtDocument) -> VectorClock {
    let mut vc = VectorClock::new();
    let mut counters: std::collections::HashMap<crate::hlc::ReplicaId, u64> =
        std::collections::HashMap::new();
    for op in &doc.history {
        let entry = counters.entry(op.replica).or_insert(0);
        *entry += 1;
        vc.observe(op.replica, *entry);
    }
    vc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::DocumentId;
    use crate::hlc::{HlcClock, ReplicaId};

    fn rid(n: u128) -> ReplicaId {
        ReplicaId::new(n)
    }

    #[test]
    fn diff_returns_missing_ops_only() {
        let id = DocumentId::from_bytes(b"d");
        let r = rid(1);
        let mut doc = CrdtDocument::new(id, r);
        let mut clock = HlcClock::with_manual_wall(r, 1);
        let _o1 = doc.set_at("/a", 1, &mut clock).unwrap();
        let _o2 = doc.set_at("/b", 2, &mut clock).unwrap();
        let _o3 = doc.set_at("/c", 3, &mut clock).unwrap();

        let mut peer = VectorClock::new();
        peer.observe(r, 1);
        let missing = diff(&doc, &peer);
        assert_eq!(missing.len(), 2);
    }

    #[test]
    fn apply_is_idempotent() {
        let id = DocumentId::from_bytes(b"d");
        let r = rid(1);
        let mut a = CrdtDocument::new(id, r);
        let mut clock = HlcClock::with_manual_wall(r, 1);
        let op = a.set_at("/a", 1, &mut clock).unwrap();
        let mut b = CrdtDocument::new(id, r);
        let n1 = apply(&mut b, std::slice::from_ref(&op)).unwrap();
        let n2 = apply(&mut b, std::slice::from_ref(&op)).unwrap();
        assert_eq!(n1, 1);
        assert_eq!(n2, 0);
    }
}
