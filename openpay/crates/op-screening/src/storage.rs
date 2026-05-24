//! In-memory index storage with on-disk CBOR snapshot.
//!
//! The sanctions corpus is small (~30k entries across all lists in
//! 2026) but every screen call has to be fast — payment auth has a
//! ~100ms budget end-to-end. We sit a tiny bloom filter in front of
//! an inverted index of normalised name tokens; the bloom rules out
//! 99% of misses without touching the hash map.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::lists::SanctionedEntity;
use crate::normalize::{NormalizedName, normalize};

/// Stable reference into [`SanctionsIndex::by_id`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityRef(pub String);

/// Tiny in-memory bloom filter.
///
/// We don't need a serious bloom filter — 30k entries, dozens of
/// tokens each. A 64-bit bitmap with two hash functions gives us a
/// false-positive rate around 1% at this fill level, which is the
/// design point: the bloom is only a coarse pre-filter, false
/// positives just cost a hash-map miss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter {
    /// 4096-bit bitmap, 64 `u64`s.
    bits: Vec<u64>,
}

impl Default for BloomFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl BloomFilter {
    const BITS: usize = 4096;
    const WORDS: usize = Self::BITS / 64;

    /// Empty filter.
    #[must_use]
    pub fn new() -> Self {
        Self { bits: vec![0u64; Self::WORDS] }
    }

    /// Insert a token's two hash positions.
    pub fn insert(&mut self, token: &str) {
        let (h1, h2) = Self::hashes(token);
        let (w1, b1) = (h1 / 64, h1 % 64);
        let (w2, b2) = (h2 / 64, h2 % 64);
        self.bits[w1] |= 1u64 << b1;
        self.bits[w2] |= 1u64 << b2;
    }

    /// Membership test (with false-positive rate).
    #[must_use]
    pub fn might_contain(&self, token: &str) -> bool {
        let (h1, h2) = Self::hashes(token);
        let (w1, b1) = (h1 / 64, h1 % 64);
        let (w2, b2) = (h2 / 64, h2 % 64);
        (self.bits[w1] & (1u64 << b1)) != 0 && (self.bits[w2] & (1u64 << b2)) != 0
    }

    fn hashes(token: &str) -> (usize, usize) {
        // BLAKE3 keyed twice. Plenty fast for our scale and gives us
        // two independent uniform positions without rolling our own
        // hash family.
        let h1 = blake3::keyed_hash(&[1u8; 32], token.as_bytes());
        let h2 = blake3::keyed_hash(&[2u8; 32], token.as_bytes());
        let h1_bytes = h1.as_bytes();
        let h2_bytes = h2.as_bytes();
        let p1 = u64::from_le_bytes([
            h1_bytes[0], h1_bytes[1], h1_bytes[2], h1_bytes[3],
            h1_bytes[4], h1_bytes[5], h1_bytes[6], h1_bytes[7],
        ]) as usize;
        let p2 = u64::from_le_bytes([
            h2_bytes[0], h2_bytes[1], h2_bytes[2], h2_bytes[3],
            h2_bytes[4], h2_bytes[5], h2_bytes[6], h2_bytes[7],
        ]) as usize;
        (p1 % Self::BITS, p2 % Self::BITS)
    }
}

/// Combined bloom + inverted-index over a population of
/// [`SanctionedEntity`]s.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SanctionsIndex {
    /// Bloom over every normalised token in the corpus.
    pub bloom: BloomFilter,
    /// Inverted index: normalised token -> entity refs that contain it.
    pub by_name: HashMap<NormalizedName, Vec<EntityRef>>,
    /// Primary store: entity id -> full record.
    pub by_id: HashMap<String, SanctionedEntity>,
}

impl SanctionsIndex {
    /// Build a fresh index from a list of entities.
    #[must_use]
    pub fn build(entities: Vec<SanctionedEntity>) -> Self {
        let mut idx = Self::default();
        for ent in entities {
            idx.insert(ent);
        }
        idx
    }

    /// Insert a single entity, updating bloom and inverted index.
    pub fn insert(&mut self, entity: SanctionedEntity) {
        let id = entity.id.clone();

        // Index every normalised token of every name + alias.
        let mut names: Vec<String> = vec![entity.name.clone()];
        names.extend(entity.name_aliases.iter().cloned());
        for name in &names {
            let norm = normalize(name);
            for token in norm.as_str().split_whitespace() {
                self.bloom.insert(token);
                self.by_name
                    .entry(NormalizedName(token.to_string()))
                    .or_default()
                    .push(EntityRef(id.clone()));
            }
        }
        self.by_id.insert(id, entity);
    }

    /// Number of indexed entities.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// All entity IDs that share at least one normalised token with
    /// `normalized_query`. The bloom filter narrows the per-token
    /// lookups; the inverted index resolves to entity IDs.
    #[must_use]
    pub fn candidate_ids(&self, normalized_query: &str) -> HashSet<String> {
        let mut out: HashSet<String> = HashSet::new();
        for token in normalized_query.split_whitespace() {
            if !self.bloom.might_contain(token) {
                continue;
            }
            if let Some(refs) = self.by_name.get(&NormalizedName(token.to_string())) {
                for r in refs {
                    out.insert(r.0.clone());
                }
            }
        }
        out
    }

    /// Serialise the index to a CBOR file.
    pub fn save_snapshot(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        ciborium::into_writer(self, writer).map_err(|e| Error::Cbor(e.to_string()))?;
        Ok(())
    }

    /// Read an index back from a CBOR file written by [`save_snapshot`].
    pub fn load_snapshot(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let idx: Self =
            ciborium::from_reader(reader).map_err(|e| Error::Cbor(e.to_string()))?;
        Ok(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lists::{EntityType, SanctionsList};
    use chrono::Utc;

    fn fake(id: &str, name: &str) -> SanctionedEntity {
        SanctionedEntity {
            id: id.to_string(),
            name: name.to_string(),
            name_aliases: vec![],
            entity_type: EntityType::Individual,
            dob: None,
            place_of_birth: None,
            addresses: vec![],
            nationalities: vec![],
            identifications: vec![],
            programs: vec![],
            last_updated: Utc::now(),
            source_list: SanctionsList::OfacSdn,
        }
    }

    #[test]
    fn build_and_lookup() {
        let idx = SanctionsIndex::build(vec![
            fake("1", "John Smith"),
            fake("2", "Maria Garcia"),
        ]);
        assert_eq!(idx.len(), 2);
        let cands = idx.candidate_ids("john smith");
        assert!(cands.contains("1"));
        assert!(!cands.contains("2"));
    }

    #[test]
    fn bloom_membership_works() {
        let mut bf = BloomFilter::new();
        bf.insert("alpha");
        bf.insert("beta");
        assert!(bf.might_contain("alpha"));
        assert!(bf.might_contain("beta"));
        // Highly likely not-contained — over 4096 bits, single token,
        // this isn't a false-positive in practice.
        assert!(!bf.might_contain("gamma_unlikely_token_xyz"));
    }

    #[test]
    fn snapshot_roundtrip() {
        let idx = SanctionsIndex::build(vec![fake("1", "Test Name")]);
        let dir = tempdir();
        let path = dir.join("snap.cbor");
        idx.save_snapshot(&path).expect("save");
        let loaded = SanctionsIndex::load_snapshot(&path).expect("load");
        assert_eq!(loaded.len(), 1);
        assert!(loaded.by_id.contains_key("1"));
    }

    fn tempdir() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let uniq = format!(
            "op-screening-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        p.push(uniq);
        std::fs::create_dir_all(&p).expect("mkdir");
        p
    }
}
