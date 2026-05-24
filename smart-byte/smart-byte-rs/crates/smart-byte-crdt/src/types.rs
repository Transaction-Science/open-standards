//! Native CRDT building blocks.
//!
//! Each type implements the [`Crdt`] trait: `merge` (commutative,
//! associative, idempotent), `delta` (the operations needed to bring a
//! peer at a given vector clock up to date), and `id` (stable CRDT
//! identifier).
//!
//! Conventions:
//!
//! * "LWW" = Last-Write-Wins. Ties are broken by `(HLC, ReplicaId)`,
//!   which is a total order.
//! * Counters are positive-only (`GCounter`) or signed (`PnCounter`).
//! * `OrSet` is the Observed-Remove Set: an item remains present if any
//!   add-tag has not been observed-as-removed. This is the canonical
//!   "intuitive" set CRDT.
//! * `RgaList` is a Replicated Growable Array — the canonical ordered
//!   list CRDT for collaborative text-style sequences.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::hlc::{HybridLogicalClock, ReplicaId};
use crate::vector_clock::VectorClock;

/// Stable identifier for an individual CRDT instance.
///
/// Wire format: big-endian 16-byte array (CBOR-friendly).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "[u8; 16]", into = "[u8; 16]")]
pub struct CrdtId(pub u128);

impl From<[u8; 16]> for CrdtId {
    fn from(b: [u8; 16]) -> Self {
        Self(u128::from_be_bytes(b))
    }
}
impl From<CrdtId> for [u8; 16] {
    fn from(c: CrdtId) -> Self {
        c.0.to_be_bytes()
    }
}

impl CrdtId {
    /// Construct a CRDT id by hashing arbitrary bytes.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let h = blake3::hash(bytes);
        let b = h.as_bytes();
        let mut raw = [0u8; 16];
        raw.copy_from_slice(&b[..16]);
        Self(u128::from_be_bytes(raw))
    }

    /// Construct a CRDT id from a raw 128-bit value.
    pub const fn new(raw: u128) -> Self {
        Self(raw)
    }
}

/// Per-add unique tag used by [`OrSet`] to track distinct add events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct UniqueTag {
    pub hlc: HybridLogicalClock,
    pub replica: ReplicaId,
    pub nonce: u64,
}

/// The CRDT trait. Implementors guarantee that `merge` is commutative,
/// associative, and idempotent — the mathematical conditions for
/// convergence.
pub trait Crdt {
    /// Merge `other` into `self`. Idempotent on repeat application.
    fn merge(&mut self, other: &Self);
    /// Produce the slice of state a peer at `since` is missing. The
    /// returned value is itself a CRDT of the same type that, when
    /// merged into the peer's replica, brings it up to date.
    fn delta(&self, since: &VectorClock) -> Self;
    /// Stable identifier for this CRDT instance.
    fn id(&self) -> CrdtId;
}

// -- LwwRegister ------------------------------------------------------

/// Last-Write-Wins register. The winning write is the one with the
/// largest `(hlc, replica)` pair.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LwwRegister<T: Clone + Debug + PartialEq> {
    pub id: CrdtId,
    pub value: T,
    pub timestamp: HybridLogicalClock,
    pub replica: ReplicaId,
}

impl<T: Clone + Debug + PartialEq> LwwRegister<T> {
    /// Construct an initial register.
    pub fn new(id: CrdtId, value: T, timestamp: HybridLogicalClock, replica: ReplicaId) -> Self {
        Self {
            id,
            value,
            timestamp,
            replica,
        }
    }

    /// Update the register if `timestamp` exceeds the current one.
    pub fn write(&mut self, value: T, timestamp: HybridLogicalClock, replica: ReplicaId) {
        if Self::wins(timestamp, replica, self.timestamp, self.replica) {
            self.value = value;
            self.timestamp = timestamp;
            self.replica = replica;
        }
    }

    /// Borrow the current value.
    pub fn get(&self) -> &T {
        &self.value
    }

    fn wins(
        ts_a: HybridLogicalClock,
        r_a: ReplicaId,
        ts_b: HybridLogicalClock,
        r_b: ReplicaId,
    ) -> bool {
        ts_a > ts_b || (ts_a == ts_b && r_a > r_b)
    }
}

impl<T: Clone + Debug + PartialEq> Crdt for LwwRegister<T> {
    fn merge(&mut self, other: &Self) {
        if Self::wins(other.timestamp, other.replica, self.timestamp, self.replica) {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
            self.replica = other.replica;
        }
    }
    fn delta(&self, since: &VectorClock) -> Self {
        // A register's "delta" is itself if the writer's seq exceeds the
        // peer's view (which we can't know exactly without an external
        // sequence — register state is fully shipped). Conservative: ship
        // self unless the peer has already seen our exact (replica, wall).
        let _ = since;
        self.clone()
    }
    fn id(&self) -> CrdtId {
        self.id
    }
}

// -- GCounter ---------------------------------------------------------

/// Grow-only counter. Each replica's contribution is monotonic; total
/// is the sum across replicas.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GCounter {
    pub id: CrdtId,
    pub entries: BTreeMap<ReplicaId, u64>,
}

impl Default for GCounter {
    fn default() -> Self {
        GCounter::new(CrdtId::new(0))
    }
}

impl GCounter {
    /// Construct an empty counter.
    pub fn new(id: CrdtId) -> Self {
        Self {
            id,
            entries: BTreeMap::new(),
        }
    }
    /// Increment the local replica's contribution.
    pub fn increment(&mut self, replica: ReplicaId, by: u64) {
        let entry = self.entries.entry(replica).or_insert(0);
        *entry = entry.saturating_add(by);
    }
    /// Sum across replicas.
    pub fn value(&self) -> u64 {
        self.entries.values().copied().sum()
    }
}

impl Crdt for GCounter {
    fn merge(&mut self, other: &Self) {
        for (k, v) in &other.entries {
            let entry = self.entries.entry(*k).or_insert(0);
            if *v > *entry {
                *entry = *v;
            }
        }
    }
    fn delta(&self, since: &VectorClock) -> Self {
        let mut out = GCounter::new(self.id);
        for (k, v) in &self.entries {
            if *v > since.get(*k) {
                out.entries.insert(*k, *v);
            }
        }
        out
    }
    fn id(&self) -> CrdtId {
        self.id
    }
}

// -- PnCounter --------------------------------------------------------

/// Positive-Negative counter: a pair of G-counters, one for increments
/// and one for decrements. The value is `positive.value() -
/// negative.value()` as an `i128` to avoid overflow on extreme inputs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PnCounter {
    pub id: CrdtId,
    pub positive: GCounter,
    pub negative: GCounter,
}

impl PnCounter {
    /// Construct an empty PN-counter.
    pub fn new(id: CrdtId) -> Self {
        Self {
            id,
            positive: GCounter::new(id),
            negative: GCounter::new(id),
        }
    }
    /// Increment the local replica's positive contribution.
    pub fn increment(&mut self, replica: ReplicaId, by: u64) {
        self.positive.increment(replica, by);
    }
    /// Increment the local replica's negative contribution.
    pub fn decrement(&mut self, replica: ReplicaId, by: u64) {
        self.negative.increment(replica, by);
    }
    /// Net value.
    pub fn value(&self) -> i128 {
        self.positive.value() as i128 - self.negative.value() as i128
    }
}

impl Crdt for PnCounter {
    fn merge(&mut self, other: &Self) {
        self.positive.merge(&other.positive);
        self.negative.merge(&other.negative);
    }
    fn delta(&self, since: &VectorClock) -> Self {
        Self {
            id: self.id,
            positive: self.positive.delta(since),
            negative: self.negative.delta(since),
        }
    }
    fn id(&self) -> CrdtId {
        self.id
    }
}

// -- OrSet ------------------------------------------------------------

/// Observed-Remove Set. Adds carry unique tags; removes record which
/// tags they observed. An element is present iff it has at least one
/// add-tag that is not present in `removed_tags`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrSet<T: Clone + Debug + Eq + Hash> {
    pub id: CrdtId,
    pub items: HashMap<T, BTreeSet<UniqueTag>>,
    pub removed_tags: BTreeSet<UniqueTag>,
}

impl<T: Clone + Debug + Eq + Hash> OrSet<T> {
    /// Construct an empty OR-set.
    pub fn new(id: CrdtId) -> Self {
        Self {
            id,
            items: HashMap::new(),
            removed_tags: BTreeSet::new(),
        }
    }
    /// Add `item` with the supplied unique tag.
    pub fn add(&mut self, item: T, tag: UniqueTag) {
        self.items.entry(item).or_default().insert(tag);
    }
    /// Remove all currently-observed tags for `item`. The item disappears
    /// only if every concurrent add elsewhere has also been observed
    /// here as removed; otherwise the OR-set's classic semantics keep it.
    pub fn remove(&mut self, item: &T) {
        if let Some(tags) = self.items.get(item) {
            for t in tags {
                self.removed_tags.insert(*t);
            }
        }
    }
    /// Is `item` currently a member?
    pub fn contains(&self, item: &T) -> bool {
        match self.items.get(item) {
            None => false,
            Some(tags) => tags.iter().any(|t| !self.removed_tags.contains(t)),
        }
    }
    /// Iterate currently-present items.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items
            .iter()
            .filter(|(_, tags)| tags.iter().any(|t| !self.removed_tags.contains(t)))
            .map(|(k, _)| k)
    }
    /// Snapshot of currently-present items.
    pub fn snapshot(&self) -> HashSet<T> {
        self.iter().cloned().collect()
    }
}

impl<T: Clone + Debug + Eq + Hash> Crdt for OrSet<T> {
    fn merge(&mut self, other: &Self) {
        for (k, tags) in &other.items {
            let entry = self.items.entry(k.clone()).or_default();
            for t in tags {
                entry.insert(*t);
            }
        }
        for t in &other.removed_tags {
            self.removed_tags.insert(*t);
        }
    }
    fn delta(&self, since: &VectorClock) -> Self {
        let mut out = Self::new(self.id);
        for (k, tags) in &self.items {
            let filtered: BTreeSet<UniqueTag> = tags
                .iter()
                .filter(|t| since.get(t.replica) < tag_seq(t))
                .copied()
                .collect();
            if !filtered.is_empty() {
                out.items.insert(k.clone(), filtered);
            }
        }
        out.removed_tags = self
            .removed_tags
            .iter()
            .filter(|t| since.get(t.replica) < tag_seq(t))
            .copied()
            .collect();
        out
    }
    fn id(&self) -> CrdtId {
        self.id
    }
}

fn tag_seq(t: &UniqueTag) -> u64 {
    // The combined "sequence" of a tag is its HLC wall*shift + logical.
    // This makes delta filtering monotone in HLC.
    (t.hlc.wall.saturating_mul(1 << 20)) | (t.hlc.logical as u64)
}

// -- LwwMap -----------------------------------------------------------

/// Last-Write-Wins map. Each entry carries the HLC + replica id of its
/// winning write.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LwwMap<K, V>
where
    K: Clone + Debug + Eq + Hash,
    V: Clone + Debug + PartialEq,
{
    pub id: CrdtId,
    pub entries: HashMap<K, (V, HybridLogicalClock, ReplicaId)>,
    pub tombstones: HashMap<K, (HybridLogicalClock, ReplicaId)>,
}

impl<K, V> LwwMap<K, V>
where
    K: Clone + Debug + Eq + Hash,
    V: Clone + Debug + PartialEq,
{
    /// Construct an empty LWW-map.
    pub fn new(id: CrdtId) -> Self {
        Self {
            id,
            entries: HashMap::new(),
            tombstones: HashMap::new(),
        }
    }

    /// Insert / overwrite `key` if the write is newer than any prior
    /// write or tombstone.
    pub fn set(&mut self, key: K, value: V, ts: HybridLogicalClock, replica: ReplicaId) {
        if let Some((t, r)) = self.tombstones.get(&key)
            && !beats(ts, replica, *t, *r)
        {
            return;
        }
        match self.entries.get(&key) {
            Some((_, t, r)) if !beats(ts, replica, *t, *r) => {}
            _ => {
                self.entries.insert(key, (value, ts, replica));
            }
        }
    }

    /// Tombstone `key` if the removal is newer than any prior write.
    pub fn remove(&mut self, key: &K, ts: HybridLogicalClock, replica: ReplicaId) {
        if let Some((_, t, r)) = self.entries.get(key) {
            if !beats(ts, replica, *t, *r) {
                return;
            }
            self.entries.remove(key);
        }
        let entry = self
            .tombstones
            .entry(key.clone())
            .or_insert((ts, replica));
        if beats(ts, replica, entry.0, entry.1) {
            *entry = (ts, replica);
        }
    }

    /// Lookup a value.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|(v, _, _)| v)
    }

    /// Iterate (key, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(k, (v, _, _))| (k, v))
    }
}

fn beats(
    ts_a: HybridLogicalClock,
    r_a: ReplicaId,
    ts_b: HybridLogicalClock,
    r_b: ReplicaId,
) -> bool {
    ts_a > ts_b || (ts_a == ts_b && r_a > r_b)
}

impl<K, V> Crdt for LwwMap<K, V>
where
    K: Clone + Debug + Eq + Hash,
    V: Clone + Debug + PartialEq,
{
    fn merge(&mut self, other: &Self) {
        for (k, (v, ts, r)) in &other.entries {
            self.set(k.clone(), v.clone(), *ts, *r);
        }
        for (k, (ts, r)) in &other.tombstones {
            self.remove(k, *ts, *r);
        }
    }
    fn delta(&self, _since: &VectorClock) -> Self {
        // LWW-map delta is conservatively the whole map; precise vector
        // filtering is handled at the op-log layer.
        self.clone()
    }
    fn id(&self) -> CrdtId {
        self.id
    }
}

// -- RgaList ----------------------------------------------------------

/// One element in an RGA list. The position id is `(hlc, replica,
/// counter)`, the parent is the preceding element's position id (or
/// `None` for the head).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RgaNode<T: Clone + Debug + PartialEq> {
    pub pos: RgaPos,
    pub parent: Option<RgaPos>,
    pub value: T,
    pub tombstone: bool,
}

/// Lexicographic position identifier for an RGA element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RgaPos {
    pub hlc: HybridLogicalClock,
    pub replica: ReplicaId,
    pub counter: u32,
}

/// Replicated Growable Array — an ordered-list CRDT.
///
/// Insertions reference a parent position. When two replicas insert
/// after the same parent, the new elements are ordered deterministically
/// by `(hlc, replica)` descending — newer inserts appear *first* after
/// the parent, which matches the canonical RGA tie-break rule.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RgaList<T: Clone + Debug + PartialEq> {
    pub id: CrdtId,
    pub nodes: Vec<RgaNode<T>>,
}

impl<T: Clone + Debug + PartialEq> RgaList<T> {
    /// Construct an empty RGA list.
    pub fn new(id: CrdtId) -> Self {
        Self {
            id,
            nodes: Vec::new(),
        }
    }

    /// Insert `value` after `parent` (or at the head if `None`). Returns
    /// the new position id.
    pub fn insert_after(
        &mut self,
        parent: Option<RgaPos>,
        value: T,
        ts: HybridLogicalClock,
        replica: ReplicaId,
        counter: u32,
    ) -> RgaPos {
        let pos = RgaPos {
            hlc: ts,
            replica,
            counter,
        };
        self.nodes.push(RgaNode {
            pos,
            parent,
            value,
            tombstone: false,
        });
        self.resort();
        pos
    }

    /// Push a fully-specified node (used when the caller already knows
    /// the exact position id). Triggers a resort.
    pub fn push_node(&mut self, parent: Option<RgaPos>, pos: RgaPos, value: T) {
        self.nodes.push(RgaNode {
            pos,
            parent,
            value,
            tombstone: false,
        });
        self.resort();
    }

    /// Mark the element at `pos` as deleted.
    pub fn delete(&mut self, pos: RgaPos) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.pos == pos) {
            n.tombstone = true;
        }
    }

    /// Iterate over live values in order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.nodes.iter().filter(|n| !n.tombstone).map(|n| &n.value)
    }

    /// Snapshot live values.
    pub fn to_vec(&self) -> Vec<T> {
        self.iter().cloned().collect()
    }

    /// Sort `nodes` into RGA traversal order: a depth-first walk where
    /// children of each parent are ordered by `(hlc, replica)`
    /// descending. Implemented iteratively for simplicity; list sizes
    /// in collaborative-document scenarios are bounded.
    fn resort(&mut self) {
        // Build a parent -> children map (children ordered desc).
        let mut by_parent: BTreeMap<Option<RgaPos>, Vec<RgaNode<T>>> = BTreeMap::new();
        for n in self.nodes.drain(..) {
            by_parent.entry(n.parent).or_default().push(n);
        }
        for v in by_parent.values_mut() {
            // Sort children: newer first by (hlc, replica) desc.
            v.sort_by(|a, b| {
                b.pos
                    .hlc
                    .cmp(&a.pos.hlc)
                    .then_with(|| b.pos.replica.cmp(&a.pos.replica))
                    .then_with(|| b.pos.counter.cmp(&a.pos.counter))
            });
        }
        // Iterative DFS from `None` parent.
        let mut out: Vec<RgaNode<T>> = Vec::new();
        let mut stack: Vec<RgaNode<T>> = by_parent.remove(&None).unwrap_or_default();
        // Stack is LIFO; we pushed children desc so popping yields desc — we
        // want to *emit* desc order, then recurse, so reverse before extending.
        stack.reverse();
        while let Some(node) = stack.pop() {
            let pos = node.pos;
            out.push(node);
            if let Some(mut children) = by_parent.remove(&Some(pos)) {
                children.reverse();
                stack.extend(children);
            }
        }
        // Append any orphans (parent not yet seen) deterministically by pos.
        let mut orphans: Vec<RgaNode<T>> =
            by_parent.into_values().flatten().collect();
        orphans.sort_by_key(|a| a.pos);
        out.extend(orphans);
        self.nodes = out;
    }
}

impl<T: Clone + Debug + PartialEq> Crdt for RgaList<T> {
    fn merge(&mut self, other: &Self) {
        for n in &other.nodes {
            match self.nodes.iter_mut().find(|m| m.pos == n.pos) {
                Some(existing) => {
                    if n.tombstone {
                        existing.tombstone = true;
                    }
                }
                None => self.nodes.push(n.clone()),
            }
        }
        self.resort();
    }
    fn delta(&self, since: &VectorClock) -> Self {
        let mut out = Self::new(self.id);
        for n in &self.nodes {
            if since.get(n.pos.replica) < n.pos.counter as u64 {
                out.nodes.push(n.clone());
            }
        }
        out.resort();
        out
    }
    fn id(&self) -> CrdtId {
        self.id
    }
}

// -- TwoPhaseSet ------------------------------------------------------

/// Two-Phase Set. Once an element is removed it cannot be re-added.
/// Simpler than [`OrSet`] but loses the "re-add" capability.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TwoPhaseSet<T: Clone + Debug + Eq + Hash> {
    pub id: CrdtId,
    pub added: HashSet<T>,
    pub removed: HashSet<T>,
}

impl<T: Clone + Debug + Eq + Hash> TwoPhaseSet<T> {
    /// Construct an empty 2P-set.
    pub fn new(id: CrdtId) -> Self {
        Self {
            id,
            added: HashSet::new(),
            removed: HashSet::new(),
        }
    }
    /// Add `item`. No-op if `item` is already removed.
    pub fn add(&mut self, item: T) {
        if !self.removed.contains(&item) {
            self.added.insert(item);
        }
    }
    /// Remove `item` permanently.
    pub fn remove(&mut self, item: &T) {
        if self.added.contains(item) {
            self.removed.insert(item.clone());
        }
    }
    /// Is `item` currently a member?
    pub fn contains(&self, item: &T) -> bool {
        self.added.contains(item) && !self.removed.contains(item)
    }
}

impl<T: Clone + Debug + Eq + Hash> Crdt for TwoPhaseSet<T> {
    fn merge(&mut self, other: &Self) {
        for a in &other.added {
            self.added.insert(a.clone());
        }
        for r in &other.removed {
            self.removed.insert(r.clone());
        }
    }
    fn delta(&self, _since: &VectorClock) -> Self {
        self.clone()
    }
    fn id(&self) -> CrdtId {
        self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: u128) -> ReplicaId {
        ReplicaId::new(n)
    }

    fn ts(wall: u64, logical: u32, node: ReplicaId) -> HybridLogicalClock {
        HybridLogicalClock {
            wall,
            logical,
            node,
        }
    }

    #[test]
    fn lww_register_later_hlc_wins() {
        let a = LwwRegister::new(CrdtId::new(1), 10_i32, ts(1, 0, r(1)), r(1));
        let b = LwwRegister::new(CrdtId::new(1), 20_i32, ts(2, 0, r(2)), r(2));
        let mut merged = a.clone();
        merged.merge(&b);
        assert_eq!(*merged.get(), 20);

        let mut other_way = b.clone();
        other_way.merge(&a);
        assert_eq!(*other_way.get(), 20);
    }

    #[test]
    fn g_counter_converges() {
        let mut a = GCounter::new(CrdtId::new(1));
        let mut b = GCounter::new(CrdtId::new(1));
        let mut c = GCounter::new(CrdtId::new(1));
        a.increment(r(1), 3);
        b.increment(r(2), 5);
        c.increment(r(3), 7);

        let mut all = a.clone();
        all.merge(&b);
        all.merge(&c);
        assert_eq!(all.value(), 15);

        // commutative
        let mut alt = c.clone();
        alt.merge(&a);
        alt.merge(&b);
        assert_eq!(alt.value(), 15);

        // idempotent
        all.merge(&b);
        assert_eq!(all.value(), 15);
    }

    #[test]
    fn or_set_observed_remove_semantics() {
        let mut x = OrSet::<&'static str>::new(CrdtId::new(1));
        let mut y = OrSet::<&'static str>::new(CrdtId::new(1));

        let tag_x = UniqueTag {
            hlc: ts(1, 0, r(1)),
            replica: r(1),
            nonce: 1,
        };
        x.add("a", tag_x);

        // y removes "a" concurrently — it never observed tag_x.
        y.remove(&"a");

        // Now merge: y observed no add-tags for "a", so its removed_tags
        // set is empty for "a". x's tag survives.
        let mut merged = x.clone();
        merged.merge(&y);
        assert!(merged.contains(&"a"));
    }

    #[test]
    fn rga_concurrent_inserts_at_same_position_both_present() {
        let mut a = RgaList::<char>::new(CrdtId::new(1));
        let parent = a.insert_after(None, 'A', ts(1, 0, r(0)), r(0), 1);

        let mut b = a.clone();
        let _pa = a.insert_after(Some(parent), 'X', ts(2, 0, r(1)), r(1), 2);
        let _pb = b.insert_after(Some(parent), 'Y', ts(2, 0, r(2)), r(2), 2);

        let mut merged = a.clone();
        merged.merge(&b);

        let chars = merged.to_vec();
        assert!(chars.contains(&'X'));
        assert!(chars.contains(&'Y'));
        assert!(chars.contains(&'A'));
        assert_eq!(chars.len(), 3);

        // Order is deterministic across the two merge directions.
        let mut alt = b.clone();
        alt.merge(&a);
        assert_eq!(merged.to_vec(), alt.to_vec());
    }

    #[test]
    fn lww_map_nested_writes_resolve_by_hlc() {
        let mut m: LwwMap<String, i32> = LwwMap::new(CrdtId::new(1));
        m.set("x".into(), 1, ts(1, 0, r(1)), r(1));
        let mut n = m.clone();
        n.set("x".into(), 2, ts(2, 0, r(2)), r(2));
        m.set("x".into(), 9, ts(1, 5, r(1)), r(1));

        m.merge(&n);
        assert_eq!(m.get(&"x".into()), Some(&2));
    }

    #[test]
    fn pn_counter_signed_value() {
        let mut a = PnCounter::new(CrdtId::new(1));
        a.increment(r(1), 10);
        a.decrement(r(1), 3);
        let mut b = PnCounter::new(CrdtId::new(1));
        b.increment(r(2), 5);
        b.decrement(r(2), 8);
        a.merge(&b);
        assert_eq!(a.value(), 4);
    }

    #[test]
    fn two_phase_set_no_readd_after_remove() {
        let mut s: TwoPhaseSet<i32> = TwoPhaseSet::new(CrdtId::new(1));
        s.add(1);
        s.remove(&1);
        s.add(1);
        assert!(!s.contains(&1));
    }
}
