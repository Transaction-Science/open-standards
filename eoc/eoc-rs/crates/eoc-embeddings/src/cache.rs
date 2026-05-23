//! Optional in-memory embedding cache.
//!
//! Embedders that hit the network (or run expensive local ONNX) benefit
//! from a content-addressed cache so the same string isn't re-embedded.
//! Keys are blake3 hashes of the (model, text) pair; values are the
//! produced vectors.

use std::num::NonZeroUsize;
use std::sync::Mutex;

use lru::LruCache;

/// Content-addressed cache key: blake3 hash of `(model, text)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(pub [u8; 32]);

impl ContentHash {
    /// Hash a `(model, text)` pair.
    pub fn of(model: &str, text: &str) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(model.as_bytes());
        h.update(&[0u8]);
        h.update(text.as_bytes());
        let mut out = [0u8; 32];
        out.copy_from_slice(h.finalize().as_bytes());
        Self(out)
    }
}

/// Bounded LRU cache of embedding vectors.
pub struct EmbeddingCache {
    inner: Mutex<LruCache<ContentHash, Vec<f32>>>,
}

impl EmbeddingCache {
    /// Construct with the given capacity. Panics only at construction time
    /// if `capacity == 0` (impossible at runtime since the type enforces it).
    pub fn new(capacity: usize) -> Self {
        // `NonZeroUsize::new(1)` is infallible at construction.
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap_or_else(|| {
            // Provably unreachable — the above max(1) guarantees ≥ 1.
            #[allow(clippy::expect_used)]
            NonZeroUsize::new(1).expect("1 is non-zero")
        });
        Self {
            inner: Mutex::new(LruCache::new(cap)),
        }
    }

    /// Look up an embedding by `(model, text)`.
    pub fn get(&self, model: &str, text: &str) -> Option<Vec<f32>> {
        let key = ContentHash::of(model, text);
        let mut guard = self.inner.lock().ok()?;
        guard.get(&key).cloned()
    }

    /// Store an embedding for `(model, text)`.
    pub fn put(&self, model: &str, text: &str, vector: Vec<f32>) {
        let key = ContentHash::of(model, text);
        if let Ok(mut guard) = self.inner.lock() {
            guard.put(key, vector);
        }
    }

    /// Current number of entries.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic() {
        let a = ContentHash::of("m", "hello");
        let b = ContentHash::of("m", "hello");
        let c = ContentHash::of("m", "world");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn cache_round_trip() {
        let c = EmbeddingCache::new(8);
        assert!(c.get("m", "hi").is_none());
        c.put("m", "hi", vec![0.1, 0.2, 0.3]);
        assert_eq!(c.get("m", "hi"), Some(vec![0.1, 0.2, 0.3]));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn cache_zero_capacity_clamps_to_one() {
        let c = EmbeddingCache::new(0);
        c.put("m", "x", vec![1.0]);
        // Size 1: store then retrieve.
        assert!(c.get("m", "x").is_some());
    }
}
