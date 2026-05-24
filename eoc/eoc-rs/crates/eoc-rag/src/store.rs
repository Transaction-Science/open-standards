//! Document store — the corpus the RAG pipeline retrieves from.
//!
//! [`DocumentStore`] is the trait every retrieval backend implements.
//! [`InMemoryStore`] is a deterministic reference implementation built
//! on Jaccard token overlap. Production backends (BM25, dense HNSW,
//! hybrid) live in `eoc-rerank` and plug in through this trait.

use std::collections::{BTreeMap, HashSet};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{RagError, RagResult};

/// A content-addressed chunk identifier (BLAKE3 of doc-id + offset + text).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChunkId(pub [u8; 32]);

impl ChunkId {
    /// Derive a `ChunkId` from `(doc_id, offset, text)`.
    pub fn new(doc_id: &str, offset: usize, text: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(doc_id.as_bytes());
        hasher.update(&offset.to_le_bytes());
        hasher.update(text.as_bytes());
        Self(*hasher.finalize().as_bytes())
    }

    /// Hex encoding for logging.
    pub fn to_hex(&self) -> String {
        let mut out = String::with_capacity(64);
        for b in &self.0 {
            out.push_str(&format!("{b:02x}"));
        }
        out
    }
}

/// A retrievable chunk — a span of some source document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Stable identifier.
    pub id: ChunkId,
    /// Source document identifier.
    pub doc_id: String,
    /// Byte offset of the chunk's first character in the source document.
    pub offset: usize,
    /// Length of the chunk in bytes.
    pub length: usize,
    /// Chunk text.
    pub text: String,
    /// Free-form metadata.
    pub metadata: BTreeMap<String, String>,
}

impl Chunk {
    /// Construct a chunk and derive its id.
    pub fn new(doc_id: impl Into<String>, offset: usize, text: impl Into<String>) -> Self {
        let doc_id = doc_id.into();
        let text = text.into();
        let id = ChunkId::new(&doc_id, offset, &text);
        let length = text.len();
        Self {
            id,
            doc_id,
            offset,
            length,
            text,
            metadata: BTreeMap::new(),
        }
    }

    /// Insert a metadata key (consumes `self`).
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A chunk with a relevance score (higher = better).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedChunk {
    /// The chunk.
    pub chunk: Chunk,
    /// Retriever score.
    pub score: f32,
    /// 1-based rank from the originating retriever.
    pub rank: usize,
}

/// Abstract retrieval surface.
///
/// `query` is the literal query text. `top_k` is the number of chunks
/// the pipeline wants back. Implementations are expected to be
/// deterministic for a given snapshot of the underlying corpus.
#[async_trait]
pub trait DocumentStore: Send + Sync {
    /// Run a retrieval and return up to `top_k` chunks.
    async fn retrieve(&self, query: &str, top_k: usize) -> RagResult<Vec<RetrievedChunk>>;

    /// Look up a chunk by id (used for citation back-resolution).
    async fn lookup(&self, id: &ChunkId) -> RagResult<Option<Chunk>>;

    /// Number of chunks in the corpus.
    fn len(&self) -> usize;

    /// Is the corpus empty?
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Human-readable store name.
    fn name(&self) -> &str;
}

/// A simple, deterministic in-memory store that scores chunks by
/// Jaccard token overlap against the query.
///
/// This is the reference store used by tests and for wiring up
/// pipelines without pulling in a heavy index. For production, swap
/// in BM25 or a dense index from `eoc-rerank` — the trait surface is
/// identical.
pub struct InMemoryStore {
    chunks: Vec<Chunk>,
    name: String,
}

impl InMemoryStore {
    /// Construct an empty store.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            chunks: Vec::new(),
            name: name.into(),
        }
    }

    /// Build from an iterator of chunks.
    pub fn from_chunks<I: IntoIterator<Item = Chunk>>(name: impl Into<String>, chunks: I) -> Self {
        Self {
            chunks: chunks.into_iter().collect(),
            name: name.into(),
        }
    }

    /// Insert a single chunk.
    pub fn insert(&mut self, chunk: Chunk) {
        self.chunks.push(chunk);
    }

    /// Borrow the chunks in the store.
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }
}

/// Lowercase alphanumeric tokeniser.
fn tokens(s: &str) -> HashSet<String> {
    let mut out: HashSet<String> = HashSet::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            for low in ch.to_lowercase() {
                cur.push(low);
            }
        } else if !cur.is_empty() {
            out.insert(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.insert(cur);
    }
    out
}

/// Jaccard similarity between two token sets. 0.0 for either-empty.
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 { 0.0 } else { inter / union }
}

#[async_trait]
impl DocumentStore for InMemoryStore {
    async fn retrieve(&self, query: &str, top_k: usize) -> RagResult<Vec<RetrievedChunk>> {
        if top_k == 0 {
            return Err(RagError::Config("top_k must be >= 1".into()));
        }
        let qt = tokens(query);
        let mut scored: Vec<(f32, &Chunk)> = self
            .chunks
            .iter()
            .map(|c| (jaccard(&qt, &tokens(&c.text)), c))
            .filter(|(s, _)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.id.0.cmp(&b.1.id.0))
        });
        scored.truncate(top_k);
        Ok(scored
            .into_iter()
            .enumerate()
            .map(|(i, (score, c))| RetrievedChunk {
                chunk: c.clone(),
                score,
                rank: i + 1,
            })
            .collect())
    }

    async fn lookup(&self, id: &ChunkId) -> RagResult<Option<Chunk>> {
        Ok(self.chunks.iter().find(|c| &c.id == id).cloned())
    }

    fn len(&self) -> usize {
        self.chunks.len()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn store_retrieves_overlapping_chunk_first() {
        let store = InMemoryStore::from_chunks(
            "test",
            vec![
                Chunk::new("doc1", 0, "apples are sweet fruits"),
                Chunk::new("doc2", 0, "bicycles have two wheels"),
                Chunk::new("doc3", 0, "oranges are citrus fruits"),
            ],
        );
        let hits = store.retrieve("citrus fruits", 2).await.expect("retrieve");
        assert_eq!(hits[0].chunk.doc_id, "doc3");
        assert!(hits.len() <= 2);
    }

    #[tokio::test]
    async fn store_top_k_zero_is_error() {
        let store = InMemoryStore::from_chunks("e", vec![Chunk::new("d", 0, "hi")]);
        assert!(store.retrieve("hi", 0).await.is_err());
    }
}
