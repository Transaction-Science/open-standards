//! `SemanticHistory` — wraps a base `HistoryLayer` with semantic
//! retrieval backed by an `EmbeddingService`.
//!
//! Architecture:
//!   - On `record()`: compute and store the embedding alongside the
//!     entry. Delegates the actual write to the inner layer.
//!   - On `lookup_exact()`: pass-through.
//!   - On `lookup_semantic()`: compute the embedding for the query,
//!     compute cosine similarity against all stored embeddings, return
//!     top-k above `min_sim`.
//!
//! The inner layer must support iterating its entries — for in-memory
//! that's free; for disk-backed it requires the index to expose
//! `entries()` (which both `InMemoryHistory` and `DiskHistory` do).
//!
//! Cost model:
//!   - `lookup_semantic`: 1 embed call (~µJ) + N cosine-sim
//!     operations (each ~10 nJ for d_model ≤ 128). Cheap relative to
//!     L3/L4.
//!   - `record`: 1 embed call (~µJ) + inner record. The embed cost
//!     is paid at write time so reads are fast.
//!
//! This is the seam where L0 stops being just exact match. A query
//! "What is the capital of France?" exact-misses but
//! semantic-hits "capital of France?" — same answer, no model call.

use jouleclaw_cascade::*;
use std::collections::HashMap;

pub trait IndexedHistory: HistoryLayer {
    /// Iterate all entries currently in the layer. Used by semantic
    /// retrieval to scan candidates.
    fn iter_entries(&self) -> Box<dyn Iterator<Item = HistoryEntry> + '_>;

    /// Update the embedding for an existing entry.
    fn set_embedding(&mut self, key: &EntryKey, embedding: Vec<f32>)
        -> Result<(), HistoryError>;
}

/// A history wrapper that adds embedding-backed semantic retrieval.
pub struct SemanticHistory<H: IndexedHistory> {
    inner: H,
    embedder: Box<dyn EmbeddingService>,
    /// Cached embeddings indexed by entry key. Filled lazily from the
    /// inner layer on first `lookup_semantic` call.
    embedding_cache: HashMap<EntryKey, Vec<f32>>,
    warmed: bool,
}

impl<H: IndexedHistory> SemanticHistory<H> {
    pub fn new(inner: H, embedder: Box<dyn EmbeddingService>) -> Self {
        Self {
            inner, embedder,
            embedding_cache: HashMap::new(),
            warmed: false,
        }
    }

    pub fn inner(&self) -> &H { &self.inner }
    pub fn inner_mut(&mut self) -> &mut H { &mut self.inner }
    pub fn embedder(&self) -> &dyn EmbeddingService { self.embedder.as_ref() }

    /// Number of embeddings in the warm cache.
    pub fn cached_embeddings(&self) -> usize { self.embedding_cache.len() }

    /// Warm the cache by computing/loading embeddings for all entries.
    /// Called automatically on first semantic lookup.
    pub fn warm(&mut self) -> Result<(), HistoryError> {
        if self.warmed { return Ok(()); }
        // Collect text inputs first to avoid borrowing issues.
        let pending: Vec<(EntryKey, String, Vec<f32>)> = self.inner.iter_entries()
            .filter_map(|e| {
                let text = match &e.query_input {
                    QueryInput::Text(s) => s.clone(),
                    _ => return None,
                };
                Some((e.key, text, e.embedding.clone()))
            })
            .collect();
        for (key, text, existing) in pending {
            if !existing.is_empty() {
                self.embedding_cache.insert(key, existing);
                continue;
            }
            // Compute on demand.
            let budget = self.embedder.estimate_cost(&text) * 2.0;
            match self.embedder.embed(&text, budget) {
                Ok(r) => {
                    let v = r.vector;
                    self.embedding_cache.insert(key, v.clone());
                    let _ = self.inner.set_embedding(&key, v);
                }
                Err(_) => continue,
            }
        }
        self.warmed = true;
        Ok(())
    }
}

impl<H: IndexedHistory> HistoryLayer for SemanticHistory<H> {
    fn lookup_exact(&mut self, key: &EntryKey) -> Result<Option<HistoryAnswer>, HistoryError> {
        self.inner.lookup_exact(key)
    }

    fn lookup_semantic(
        &mut self,
        embedding: &[f32],
        k: usize,
        min_sim: f32,
    ) -> Result<Vec<(HistoryAnswer, f32)>, HistoryError> {
        if !self.warmed {
            self.warm()?;
        }
        // Score all cached embeddings.
        let mut scored: Vec<(EntryKey, f32)> = self.embedding_cache.iter()
            .map(|(k, v)| (*k, cosine_sim(embedding, v)))
            .filter(|(_, s)| *s >= min_sim)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        let mut out = Vec::new();
        for (key, sim) in scored {
            if let Some(ha) = self.inner.lookup_exact(&key)? {
                out.push((ha, sim));
            }
        }
        Ok(out)
    }

    fn record(&mut self, q: &Query, a: &Answer) -> Result<EntryKey, HistoryError> {
        let key = self.inner.record(q, a)?;
        // Compute embedding on record if the query is text.
        if let QueryInput::Text(s) = &q.input {
            let budget = self.embedder.estimate_cost(s) * 2.0;
            if let Ok(r) = self.embedder.embed(s, budget) {
                self.embedding_cache.insert(key, r.vector.clone());
                let _ = self.inner.set_embedding(&key, r.vector);
            }
        }
        Ok(key)
    }

    fn estimate_lookup_cost(&self, q: &Query) -> f64 {
        // Exact lookup cost from inner.
        self.inner.estimate_lookup_cost(q)
    }

    fn stats(&self) -> &HistoryStats {
        self.inner.stats()
    }
}

/// Estimate the joule cost of a semantic lookup given the embedder and
/// the current entry count.
pub fn estimate_semantic_lookup_cost(
    embedder: &dyn EmbeddingService,
    text: &str,
    n_entries: usize,
) -> f64 {
    let embed_cost = embedder.estimate_cost(text);
    // Each cosine sim: ~10 nJ per pair (d_model fmadd ≈ 10 nJ).
    let scan_cost = (n_entries as f64) * 1e-8;
    embed_cost + scan_cost
}
