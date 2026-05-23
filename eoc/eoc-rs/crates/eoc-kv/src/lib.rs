//! EOC stage 2 — key-value + embedding match.
//!
//! The KV stage handles two retrieval modes:
//!
//! 1. **Exact key match.** The prompt is used as a string key against a
//!    `KvBackend`. A hit returns the stored payload directly.
//! 2. **Embedding-similarity match.** If the query carries an embedding,
//!    the stage scans known embeddings and returns the closest match whose
//!    cosine similarity exceeds the configured threshold.
//!
//! Joule cost at this stage is small but non-zero — we report a synthetic
//! estimated cost. Production deployments swap in a real disk-backed KV
//! (e.g. RocksDB, fjall) and a real vector index.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use eoc_cache::Stage;
use eoc_core::{JouleCost, Query, Response, Stage as StageKind};

/// A KV backend — string-keyed byte storage.
pub trait KvBackend: Send + Sync {
    /// Get the value for a key.
    fn get(&self, key: &str) -> Option<Vec<u8>>;
    /// Put a key/value pair.
    fn put(&self, key: &str, value: Vec<u8>);
    /// List `(embedding, key)` pairs for similarity search.
    fn embeddings(&self) -> Vec<(Vec<f32>, String)>;
    /// Store an embedding alongside a key.
    fn put_embedding(&self, embedding: Vec<f32>, key: String);
}

/// In-memory reference `KvBackend`.
#[derive(Default)]
pub struct MemoryKvBackend {
    store: Mutex<HashMap<String, Vec<u8>>>,
    embeddings: Mutex<Vec<(Vec<f32>, String)>>,
}

impl MemoryKvBackend {
    /// Construct an empty in-memory backend.
    pub fn new() -> Self {
        Self::default()
    }
}

impl KvBackend for MemoryKvBackend {
    fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.store
            .lock()
            .expect("kv lock poisoned")
            .get(key)
            .cloned()
    }
    fn put(&self, key: &str, value: Vec<u8>) {
        self.store
            .lock()
            .expect("kv lock poisoned")
            .insert(key.to_string(), value);
    }
    fn embeddings(&self) -> Vec<(Vec<f32>, String)> {
        self.embeddings
            .lock()
            .expect("kv lock poisoned")
            .clone()
    }
    fn put_embedding(&self, embedding: Vec<f32>, key: String) {
        self.embeddings
            .lock()
            .expect("kv lock poisoned")
            .push((embedding, key));
    }
}

/// Configuration for the KV stage.
#[derive(Debug, Clone)]
pub struct KvConfig {
    /// Cosine-similarity threshold for embedding match.
    pub similarity_threshold: f32,
    /// Synthetic estimated cost in micro-joules per KV resolution.
    pub estimated_microjoules: u64,
}

impl Default for KvConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.95,
            estimated_microjoules: 50,
        }
    }
}

/// The KV stage.
pub struct KvStage {
    backend: Box<dyn KvBackend>,
    config: KvConfig,
}

impl KvStage {
    /// Construct a `KvStage` from a backend.
    pub fn new(backend: Box<dyn KvBackend>) -> Self {
        Self {
            backend,
            config: KvConfig::default(),
        }
    }

    /// Override the config.
    pub fn with_config(mut self, config: KvConfig) -> Self {
        self.config = config;
        self
    }

    /// Borrow the underlying backend (for tests / wiring).
    pub fn backend(&self) -> &dyn KvBackend {
        self.backend.as_ref()
    }

    /// Borrow the config.
    pub fn config(&self) -> &KvConfig {
        &self.config
    }
}

#[async_trait]
impl Stage for KvStage {
    async fn try_resolve(&self, q: &Query) -> Option<Response> {
        // 1) Exact-key match on prompt.
        if let Some(raw) = self.backend.get(&q.prompt) {
            let payload = String::from_utf8_lossy(&raw).into_owned();
            return Some(Response::new(
                q.id,
                payload,
                StageKind::Kv,
                JouleCost::estimated(self.config.estimated_microjoules),
            ));
        }

        // 2) Embedding-similarity match.
        if let Some(query_emb) = q.embedding.as_ref() {
            let mut best: Option<(f32, String)> = None;
            for (stored_emb, key) in self.backend.embeddings() {
                if let Some(sim) = cosine_similarity(query_emb, &stored_emb)
                    && sim >= self.config.similarity_threshold
                    && best.as_ref().is_none_or(|(s, _)| sim > *s)
                {
                    best = Some((sim, key));
                }
            }
            if let Some((_sim, key)) = best
                && let Some(raw) = self.backend.get(&key) {
                    let payload = String::from_utf8_lossy(&raw).into_owned();
                    return Some(Response::new(
                        q.id,
                        payload,
                        StageKind::Kv,
                        JouleCost::estimated(self.config.estimated_microjoules),
                    ));
                }
        }

        None
    }
}

/// Cosine similarity between two equal-length vectors. Returns `None` if
/// either vector has zero norm or the lengths differ.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    Some(dot / (na.sqrt() * nb.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_known_pairs() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];
        let c = [0.0f32, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&a, &c).unwrap()).abs() < 1e-6);
    }

    #[tokio::test]
    async fn kv_exact_key_hits() {
        let backend = Box::new(MemoryKvBackend::new());
        backend.put("hello", b"world".to_vec());
        let stage = KvStage::new(backend);
        let q = Query::new("hello");
        let r = stage.try_resolve(&q).await.expect("hit");
        assert_eq!(r.payload, "world");
        assert_eq!(r.stage, StageKind::Kv);
    }

    #[tokio::test]
    async fn kv_embedding_similarity_hits() {
        let backend = Box::new(MemoryKvBackend::new());
        // Store a payload keyed by "canonical" with a unit-x embedding.
        backend.put("canonical", b"canonical answer".to_vec());
        backend.put_embedding(vec![1.0, 0.0, 0.0], "canonical".to_string());

        let stage = KvStage::new(backend).with_config(KvConfig {
            similarity_threshold: 0.99,
            estimated_microjoules: 50,
        });

        // A query whose embedding is essentially the same vector (rotated
        // by ~1e-3 rad) should resolve via similarity.
        let q = Query::new("paraphrased prompt").with_embedding(vec![0.9999, 0.0141, 0.0]);
        let r = stage.try_resolve(&q).await.expect("hit");
        assert_eq!(r.payload, "canonical answer");
        assert_eq!(r.stage, StageKind::Kv);
    }

    #[tokio::test]
    async fn kv_misses_when_far() {
        let backend = Box::new(MemoryKvBackend::new());
        backend.put("canonical", b"x".to_vec());
        backend.put_embedding(vec![1.0, 0.0, 0.0], "canonical".to_string());
        let stage = KvStage::new(backend);
        let q = Query::new("unrelated").with_embedding(vec![0.0, 1.0, 0.0]);
        assert!(stage.try_resolve(&q).await.is_none());
    }
}
