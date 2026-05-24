//! [`EntityBloom`] — a fixed-size, double-hashing Bloom filter keyed by
//! [`EntityKey`].
//!
//! ## Why a Bloom filter here
//!
//! The fraud-graph ingest path is the hot loop. Every payment looks up a
//! handful of identifiers. A negative answer ("we've never seen this
//! card") is the common case, and we can short-circuit `HashMap` lookups
//! by checking a contiguous bit-array first. The Bloom filter is sized
//! to the operator's expected entity cardinality at construction time;
//! resizing on overflow is the operator's job (rebuild from the
//! authoritative `FraudGraph::vertices()` iterator).
//!
//! ## Math
//!
//! For target false-positive rate `p` and capacity `n`:
//! `m = -n * ln(p) / (ln(2))^2`,  `k = (m / n) * ln(2)`.
//! We round `m` up to a multiple of 64.
//!
//! ## What this is NOT
//!
//! A counting Bloom filter. We do not support deletion. To "remove" an
//! entity, rebuild the filter from the surviving vertex set.

use crate::entity::EntityKey;
use crate::error::{Error, Result};

/// Fixed-size Bloom filter over [`EntityKey`].
#[derive(Debug, Clone)]
pub struct EntityBloom {
    /// Bit array, packed into `u64` words.
    bits: Vec<u64>,
    /// Total number of bits = `bits.len() * 64`.
    m: u64,
    /// Hash functions to apply per insertion / query.
    k: u32,
    /// Approximate number of distinct items inserted.
    /// Bumped on every insert; only an estimate (no dedup).
    inserted: u64,
}

impl EntityBloom {
    /// Construct a filter sized for `capacity` entries at false-positive
    /// rate `fpr` (in `(0.0, 1.0)`).
    pub fn with_capacity(capacity: usize, fpr: f64) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::InvalidConfig("bloom capacity must be > 0"));
        }
        if !(fpr > 0.0 && fpr < 1.0) {
            return Err(Error::InvalidConfig("bloom fpr must be in (0,1)"));
        }
        let ln2 = core::f64::consts::LN_2;
        let m_real = -(capacity as f64) * fpr.ln() / (ln2 * ln2);
        // Round up to a multiple of 64 so we own complete words.
        let m_bits = ((m_real.ceil() as u64) + 63) & !63;
        let m_bits = m_bits.max(64);
        let k_real = (m_bits as f64 / capacity as f64) * ln2;
        let k = (k_real.round() as u32).clamp(1, 32);
        let words = (m_bits / 64) as usize;
        Ok(Self {
            bits: vec![0u64; words],
            m: m_bits,
            k,
            inserted: 0,
        })
    }

    /// Insert a key.
    pub fn insert(&mut self, key: &EntityKey) {
        let bits: Vec<u64> = self.indices(key).collect();
        for bit in bits {
            let word = (bit / 64) as usize;
            let mask = 1u64 << (bit % 64);
            self.bits[word] |= mask;
        }
        self.inserted = self.inserted.saturating_add(1);
    }

    /// `true` if the key *may* be present, `false` if definitely absent.
    pub fn contains(&self, key: &EntityKey) -> bool {
        for bit in self.indices(key) {
            let word = (bit / 64) as usize;
            let mask = 1u64 << (bit % 64);
            if self.bits[word] & mask == 0 {
                return false;
            }
        }
        true
    }

    /// Approximate insertion count (no deduplication).
    pub fn approx_count(&self) -> u64 {
        self.inserted
    }

    /// Current size in bits.
    pub fn bit_capacity(&self) -> u64 {
        self.m
    }

    /// Hash-function count (`k`).
    pub fn hash_count(&self) -> u32 {
        self.k
    }

    /// Double-hashing scheme à la Kirsch & Mitzenmacher: derive two 64-bit
    /// hashes from the SHA-256 digest already in the key, then combine
    /// `h1 + i * h2` for each of the `k` indices.
    fn indices(&self, key: &EntityKey) -> impl Iterator<Item = u64> + '_ {
        let h1 = u64::from_le_bytes(key.digest[0..8].try_into().unwrap_or([0u8; 8]));
        let h2 = u64::from_le_bytes(key.digest[8..16].try_into().unwrap_or([0u8; 8]));
        // Mix the kind tag into h2 so different kinds with the same raw
        // value land on different bits.
        let kind_byte = key.kind as u8 as u64;
        let h2 = h2.wrapping_add(kind_byte.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let m = self.m;
        (0..self.k).map(move |i| {
            let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
            combined % m
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::{EntityKey, EntityKind};

    #[test]
    fn contains_after_insert() {
        let mut b = EntityBloom::with_capacity(1024, 0.01).expect("ok");
        let key = EntityKey::from_raw(EntityKind::EmailHash, "a@b");
        assert!(!b.contains(&key));
        b.insert(&key);
        assert!(b.contains(&key));
    }

    #[test]
    fn no_false_negatives() {
        let mut b = EntityBloom::with_capacity(2048, 0.01).expect("ok");
        let mut keys = Vec::new();
        for i in 0..1000 {
            let key = EntityKey::from_raw(EntityKind::Account, &format!("acc-{i}"));
            b.insert(&key);
            keys.push(key);
        }
        for key in &keys {
            assert!(b.contains(key));
        }
    }

    #[test]
    fn rejects_zero_capacity() {
        assert!(EntityBloom::with_capacity(0, 0.01).is_err());
    }

    #[test]
    fn rejects_silly_fpr() {
        assert!(EntityBloom::with_capacity(100, 0.0).is_err());
        assert!(EntityBloom::with_capacity(100, 1.0).is_err());
    }
}
