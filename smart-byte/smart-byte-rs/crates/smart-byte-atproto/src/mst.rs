//! Merkle Search Tree — sketch implementation.
//!
//! AT Protocol stores a repository as a Merkle Search Tree keyed by
//! `<collection>/<rkey>` strings, with each leaf pointing at the CID of
//! a record block. This module provides a *sketch* MST: it implements
//! insert / delete / lookup with deterministic structure but does not
//! attempt to match the upstream wire format byte-for-byte. It is
//! intended for in-process repo assembly and tests, not interop.
//!
//! The deterministic structure means two MSTs containing the same set
//! of `(key, cid)` entries hash to the same [`root_hash`][Mst::root_hash]
//! regardless of insert order. The hash is BLAKE3 over the sorted
//! serialised entries.

use std::collections::BTreeMap;

use crate::car::Cid;
use crate::error::AtprotoError;

/// A single MST leaf entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MstEntry {
    /// Record key (`<collection>/<rkey>` in AT Protocol).
    pub key: String,
    /// CID of the record's block.
    pub cid: Cid,
}

/// An in-memory Merkle Search Tree (sketch).
#[derive(Debug, Clone, Default)]
pub struct Mst {
    entries: BTreeMap<String, Cid>,
}

impl Mst {
    /// Construct an empty tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace `(key, cid)`. Returns the previous CID if any.
    pub fn insert(&mut self, key: impl Into<String>, cid: Cid) -> Option<Cid> {
        self.entries.insert(key.into(), cid)
    }

    /// Remove `key` from the tree. Returns the previous CID if any.
    pub fn delete(&mut self, key: &str) -> Option<Cid> {
        self.entries.remove(key)
    }

    /// Look up `key`'s CID, if present.
    pub fn lookup(&self, key: &str) -> Option<&Cid> {
        self.entries.get(key)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate entries in key order.
    pub fn iter(&self) -> impl Iterator<Item = MstEntry> + '_ {
        self.entries.iter().map(|(k, c)| MstEntry {
            key: k.clone(),
            cid: c.clone(),
        })
    }

    /// Deterministic root hash of the tree.
    ///
    /// The hash is BLAKE3 over the concatenation of
    /// `len(key) || key || cid_bytes` for every entry in key-sorted
    /// order. Two trees with the same `(key, cid)` set always hash to
    /// the same value.
    pub fn root_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for (key, cid) in &self.entries {
            let kbytes = key.as_bytes();
            let klen = (kbytes.len() as u32).to_le_bytes();
            hasher.update(&klen);
            hasher.update(kbytes);
            let cid_bytes = cid.to_bytes();
            let clen = (cid_bytes.len() as u32).to_le_bytes();
            hasher.update(&clen);
            hasher.update(&cid_bytes);
        }
        *hasher.finalize().as_bytes()
    }

    /// Hex-encoded root hash, useful for logging.
    pub fn root_hex(&self) -> String {
        let h = self.root_hash();
        let mut s = String::with_capacity(64);
        for b in h.iter() {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    /// Build a CID-shaped pointer to the tree (the root hash wrapped as a
    /// `dag-cbor` CID). This is what gets embedded in the commit's
    /// `data` field.
    pub fn root_cid(&self) -> Cid {
        Cid {
            codec: crate::car::CODEC_DAG_CBOR,
            digest: self.root_hash(),
        }
    }

    /// Validate the AT-Protocol key shape `<collection>/<rkey>`.
    pub fn check_key(key: &str) -> Result<(), AtprotoError> {
        let (collection, rkey) = key.split_once('/').ok_or_else(|| {
            AtprotoError::Mst(format!(
                "mst key must be <collection>/<rkey>: {key}"
            ))
        })?;
        if collection.is_empty() || rkey.is_empty() {
            return Err(AtprotoError::Mst(format!(
                "mst key has empty side: {key}"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid_for(label: &[u8]) -> Cid {
        Cid::dag_cbor(label)
    }

    #[test]
    fn insert_and_lookup() {
        let mut mst = Mst::new();
        mst.insert("app.bsky.feed.post/abc", cid_for(b"a"));
        mst.insert("app.bsky.feed.post/def", cid_for(b"b"));
        assert_eq!(mst.len(), 2);
        assert_eq!(mst.lookup("app.bsky.feed.post/abc"), Some(&cid_for(b"a")));
    }

    #[test]
    fn delete_works() {
        let mut mst = Mst::new();
        mst.insert("k/1", cid_for(b"x"));
        let prev = mst.delete("k/1");
        assert_eq!(prev, Some(cid_for(b"x")));
        assert!(mst.lookup("k/1").is_none());
    }

    #[test]
    fn hash_is_order_independent() {
        let mut a = Mst::new();
        a.insert("c/2", cid_for(b"two"));
        a.insert("c/1", cid_for(b"one"));
        let mut b = Mst::new();
        b.insert("c/1", cid_for(b"one"));
        b.insert("c/2", cid_for(b"two"));
        assert_eq!(a.root_hash(), b.root_hash());
    }

    #[test]
    fn key_validation() {
        assert!(Mst::check_key("app.bsky.feed.post/abc").is_ok());
        assert!(Mst::check_key("no-slash").is_err());
        assert!(Mst::check_key("/empty-left").is_err());
    }
}
