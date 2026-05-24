//! Sorted-vector interval tree for BIN range lookup.
//!
//! Card-network ranges form a set of **disjoint** half-open
//! intervals over `0..=99_999_999` (8-digit prefix space). A
//! plain sorted `Vec<BinRange>` with binary search by `low` gives
//! `O(log N)` point lookups, which is the right complexity for
//! the ~few-thousand published ranges. We do not need a full
//! augmented interval tree because BIN ranges do not overlap by
//! construction (ISO/IEC 7812 assigns them exclusively).
//!
//! ## Invariant
//!
//! The `ranges` vector is **sorted by `low`** and contains no
//! overlapping intervals. [`RangeTree::insert`] enforces both.

use crate::bin::{Bin, BinRange};
use crate::error::{Error, Result};

/// Sorted, non-overlapping collection of [`BinRange`] entries
/// optimized for point lookups.
#[derive(Debug, Default, Clone)]
pub struct RangeTree {
    ranges: Vec<BinRange>,
}

impl RangeTree {
    /// Empty tree.
    pub const fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    /// Build from an iterator of ranges. Ranges are inserted in
    /// order; later insertions that overlap an earlier range are
    /// rejected.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidRange`] if any pair overlaps.
    pub fn from_ranges<I: IntoIterator<Item = BinRange>>(iter: I) -> Result<Self> {
        let mut tree = Self::new();
        for r in iter {
            tree.insert(r)?;
        }
        Ok(tree)
    }

    /// Insert a range, maintaining sorted order and disjointness.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidRange`] if the new range overlaps an
    ///   existing one.
    pub fn insert(&mut self, r: BinRange) -> Result<()> {
        // Find insertion point by `low`.
        let pos = self
            .ranges
            .binary_search_by(|probe| probe.low.cmp(&r.low))
            .unwrap_or_else(|i| i);

        // Check overlap with predecessor.
        if pos > 0 {
            let prev = &self.ranges[pos - 1];
            if prev.high > r.low {
                return Err(Error::InvalidRange {
                    low: r.low,
                    high: r.high,
                });
            }
        }
        // Check overlap with successor.
        if pos < self.ranges.len() {
            let next = &self.ranges[pos];
            if r.high > next.low {
                return Err(Error::InvalidRange {
                    low: r.low,
                    high: r.high,
                });
            }
        }
        self.ranges.insert(pos, r);
        Ok(())
    }

    /// `O(log N)` point lookup. Returns the unique range
    /// containing this BIN, or `None`.
    pub fn lookup(&self, bin: &Bin) -> Option<&BinRange> {
        let p = bin.prefix_8();
        // Find rightmost range with `low <= p`.
        let idx = match self.ranges.binary_search_by(|probe| probe.low.cmp(&p)) {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };
        let candidate = &self.ranges[idx];
        if candidate.contains(bin) {
            Some(candidate)
        } else {
            None
        }
    }

    /// Total number of ranges.
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// True iff empty.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Read-only access to the underlying sorted slice.
    pub fn ranges(&self) -> &[BinRange] {
        &self.ranges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_type::CardType;
    use crate::network::CardNetwork;

    fn mk(low: u32, high: u32, net: CardNetwork) -> BinRange {
        BinRange::new(low, high, net, CardType::Credit, None, false).expect("valid")
    }

    #[test]
    fn insert_and_lookup() {
        let mut t = RangeTree::new();
        t.insert(mk(40_000_000, 50_000_000, CardNetwork::Visa))
            .expect("ok");
        t.insert(mk(51_000_000, 56_000_000, CardNetwork::Mastercard))
            .expect("ok");

        let v = Bin::parse("411111").expect("ok");
        let m = Bin::parse("520000").expect("ok");
        let n = Bin::parse("600000").expect("ok");

        assert_eq!(t.lookup(&v).map(|r| r.network), Some(CardNetwork::Visa));
        assert_eq!(
            t.lookup(&m).map(|r| r.network),
            Some(CardNetwork::Mastercard)
        );
        assert!(t.lookup(&n).is_none());
    }

    #[test]
    fn overlap_rejected() {
        let mut t = RangeTree::new();
        t.insert(mk(40_000_000, 50_000_000, CardNetwork::Visa))
            .expect("ok");
        let err = t.insert(mk(45_000_000, 55_000_000, CardNetwork::Mastercard));
        assert!(matches!(err, Err(Error::InvalidRange { .. })));
    }

    #[test]
    fn adjacent_ranges_ok() {
        // Half-open: [40M, 50M) and [50M, 60M) do NOT overlap.
        let mut t = RangeTree::new();
        t.insert(mk(40_000_000, 50_000_000, CardNetwork::Visa))
            .expect("ok");
        t.insert(mk(50_000_000, 60_000_000, CardNetwork::Mastercard))
            .expect("ok");
        assert_eq!(t.len(), 2);
    }
}
