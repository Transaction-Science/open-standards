//! Vector clocks for causal-history comparisons.
//!
//! A vector clock maps each replica to the highest sequence number this
//! replica has *observed* (locally or remotely). Two clocks compare as
//! happens-before, happens-after, equal, or concurrent.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::hlc::ReplicaId;

/// A vector clock keyed by replica.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VectorClock(pub BTreeMap<ReplicaId, u64>);

impl VectorClock {
    /// Construct an empty vector clock.
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Read the sequence number recorded for `replica` (zero if absent).
    pub fn get(&self, replica: ReplicaId) -> u64 {
        self.0.get(&replica).copied().unwrap_or(0)
    }

    /// Record an observation: ensure the entry for `replica` is at
    /// least `seq`.
    pub fn observe(&mut self, replica: ReplicaId, seq: u64) {
        let entry = self.0.entry(replica).or_insert(0);
        if seq > *entry {
            *entry = seq;
        }
    }

    /// Bump the local replica's own counter and return the new value.
    pub fn tick(&mut self, replica: ReplicaId) -> u64 {
        let entry = self.0.entry(replica).or_insert(0);
        *entry += 1;
        *entry
    }

    /// True iff `self` strictly happens-before `other`: every entry in
    /// `self` is `<= other`'s entry, with at least one strict `<`.
    pub fn happens_before(&self, other: &Self) -> bool {
        let mut at_least_one_strict = false;
        // Check all entries in self are <= other.
        for (k, v) in &self.0 {
            let ov = other.get(*k);
            if *v > ov {
                return false;
            }
            if *v < ov {
                at_least_one_strict = true;
            }
        }
        // Any extra entries in other are strict gains.
        for (k, v) in &other.0 {
            if !self.0.contains_key(k) && *v > 0 {
                at_least_one_strict = true;
            }
        }
        at_least_one_strict
    }

    /// True iff `self` and `other` are concurrent (neither precedes the
    /// other, and they are not equal).
    pub fn concurrent(&self, other: &Self) -> bool {
        !self.happens_before(other) && !other.happens_before(self) && self != other
    }

    /// Pointwise maximum of two vector clocks.
    pub fn merge(&mut self, other: &Self) {
        for (k, v) in &other.0 {
            self.observe(*k, *v);
        }
    }

    /// Iterate (replica, seq) entries.
    pub fn iter(&self) -> impl Iterator<Item = (&ReplicaId, &u64)> {
        self.0.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: u128) -> ReplicaId {
        ReplicaId::new(n)
    }

    #[test]
    fn empty_clocks_are_equal_not_concurrent() {
        let a = VectorClock::new();
        let b = VectorClock::new();
        assert!(!a.happens_before(&b));
        assert!(!a.concurrent(&b));
    }

    #[test]
    fn happens_before_basic() {
        let mut a = VectorClock::new();
        a.tick(r(1));
        let mut b = a.clone();
        b.tick(r(1));
        assert!(a.happens_before(&b));
        assert!(!b.happens_before(&a));
        assert!(!a.concurrent(&b));
    }

    #[test]
    fn concurrent_when_disjoint_ticks() {
        let mut a = VectorClock::new();
        a.tick(r(1));
        let mut b = VectorClock::new();
        b.tick(r(2));
        assert!(a.concurrent(&b));
        assert!(b.concurrent(&a));
    }

    #[test]
    fn merge_is_pointwise_max() {
        let mut a = VectorClock::new();
        a.observe(r(1), 5);
        a.observe(r(2), 1);
        let mut b = VectorClock::new();
        b.observe(r(1), 3);
        b.observe(r(2), 7);
        a.merge(&b);
        assert_eq!(a.get(r(1)), 5);
        assert_eq!(a.get(r(2)), 7);
    }
}
