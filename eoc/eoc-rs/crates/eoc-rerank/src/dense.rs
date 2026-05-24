//! Dense (embedding-based) retrieval with an in-process HNSW index.
//!
//! [`DenseIndex`] wraps any [`Embedder`] from [`eoc_embeddings`] with a
//! Hierarchical Navigable Small World (HNSW) graph for approximate
//! nearest-neighbour search.
//!
//! The HNSW implementation is a faithful, single-threaded port of
//! Malkov & Yashunin 2018 ("Efficient and Robust Approximate Nearest
//! Neighbor Search using Hierarchical Navigable Small World Graphs",
//! arXiv:1603.09320). It uses cosine distance (`1 - cos_sim`) so vectors
//! are L2-normalised at insert time.
//!
//! For million-scale corpora swap this in for `hnsw_rs` or `usearch`;
//! the trait surface is identical.

use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use eoc_embeddings::Embedder;

use crate::DocId;
use crate::error::{RerankError, RerankResult};
use crate::reranker::Retriever;

/// HNSW build / query parameters.
#[derive(Debug, Clone)]
pub struct HnswParams {
    /// Max neighbours per node at layer > 0. Typical 16.
    pub m: usize,
    /// Max neighbours per node at layer 0. Typical 2*m.
    pub m_max0: usize,
    /// Size of the dynamic candidate list at build time. Typical 200.
    pub ef_construction: usize,
    /// Size of the dynamic candidate list at search time. Typical 50-200.
    pub ef_search: usize,
    /// Level-generation multiplier: `mL = 1 / ln(m)`.
    pub level_mult: f32,
    /// Seed for the level generator (deterministic).
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m_max0: m * 2,
            ef_construction: 200,
            ef_search: 50,
            level_mult: 1.0 / (m as f32).ln(),
            seed: 0x4544_4F43_5249_4E47, // "EOCRING"
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Ord32(f32);

impl Eq for Ord32 {}

impl PartialOrd for Ord32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ord32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// In-process HNSW index over `Vec<f32>` vectors (L2-normalised, cosine).
#[derive(Default)]
pub struct HnswIndex {
    params: HnswParams,
    vectors: Vec<Vec<f32>>,
    /// `nodes[node_id][layer]` -> neighbour ids.
    nodes: Vec<Vec<Vec<usize>>>,
    /// Entry point (highest-layer node).
    entry: Option<usize>,
    /// Highest layer occupied.
    max_layer: usize,
    /// Deterministic RNG state for level generation.
    rng_state: u64,
}

impl HnswIndex {
    /// Construct an empty index.
    pub fn new(params: HnswParams) -> Self {
        let seed = params.seed;
        Self {
            params,
            vectors: Vec::new(),
            nodes: Vec::new(),
            entry: None,
            max_layer: 0,
            rng_state: seed,
        }
    }

    /// Number of inserted vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Is the index empty?
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Insert a vector and return its index. The caller is responsible
    /// for maintaining a `(index -> DocId)` mapping.
    pub fn insert(&mut self, mut v: Vec<f32>) -> usize {
        l2_normalise(&mut v);
        let id = self.vectors.len();
        let layer = self.random_level();

        // Initialise empty neighbour lists per layer.
        let mut layers = Vec::with_capacity(layer + 1);
        for _ in 0..=layer {
            layers.push(Vec::new());
        }

        if self.entry.is_none() {
            self.vectors.push(v);
            self.nodes.push(layers);
            self.entry = Some(id);
            self.max_layer = layer;
            return id;
        }

        // Phase 1 — greedy walk from the entry point down to layer+1.
        let mut curr = self.entry.expect("entry set");
        for lc in ((layer + 1)..=self.max_layer).rev() {
            curr = self.greedy_search(&v, curr, lc);
        }

        self.vectors.push(v.clone());
        self.nodes.push(layers);

        // Phase 2 — layer-by-layer ef-construction neighbour selection.
        for lc in (0..=layer.min(self.max_layer)).rev() {
            let m_max = if lc == 0 { self.params.m_max0 } else { self.params.m };
            let ef = self.params.ef_construction;
            let candidates = self.search_layer(&v, curr, ef, lc);
            let neighbours = select_neighbours_heuristic(&self.vectors, &v, candidates, m_max);

            // Bidirectional edges.
            self.nodes[id][lc] = neighbours.clone();
            for &n in &neighbours {
                self.nodes[n][lc].push(id);
                // Prune n's neighbours back to m_max if exceeded.
                if self.nodes[n][lc].len() > m_max {
                    let cand: Vec<(usize, f32)> = self.nodes[n][lc]
                        .iter()
                        .map(|&j| (j, distance(&self.vectors[n], &self.vectors[j])))
                        .collect();
                    let pruned = select_neighbours_heuristic(
                        &self.vectors,
                        &self.vectors[n].clone(),
                        cand,
                        m_max,
                    );
                    self.nodes[n][lc] = pruned;
                }
            }

            if !neighbours.is_empty() {
                curr = neighbours[0];
            }
        }

        if layer > self.max_layer {
            self.max_layer = layer;
            self.entry = Some(id);
        }

        id
    }

    /// k-NN search — returns `(node_id, distance)` pairs sorted ascending
    /// by distance (smaller = closer).
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        if self.vectors.is_empty() || top_k == 0 {
            return Vec::new();
        }
        let mut q = query.to_vec();
        l2_normalise(&mut q);

        let mut curr = self.entry.expect("non-empty -> entry set");
        for lc in (1..=self.max_layer).rev() {
            curr = self.greedy_search(&q, curr, lc);
        }
        let ef = self.params.ef_search.max(top_k);
        let mut results = self.search_layer(&q, curr, ef, 0);
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    fn greedy_search(&self, q: &[f32], entry: usize, layer: usize) -> usize {
        let mut curr = entry;
        let mut curr_d = distance(q, &self.vectors[curr]);
        loop {
            let mut improved = false;
            if let Some(layer_neighbours) = self.nodes[curr].get(layer) {
                for &n in layer_neighbours {
                    let d = distance(q, &self.vectors[n]);
                    if d < curr_d {
                        curr_d = d;
                        curr = n;
                        improved = true;
                    }
                }
            }
            if !improved {
                return curr;
            }
        }
    }

    fn search_layer(
        &self,
        q: &[f32],
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited: HashSet<usize> = HashSet::new();
        visited.insert(entry);
        let entry_d = distance(q, &self.vectors[entry]);

        // `candidates` is a min-heap (closest first) via Reverse.
        let mut candidates: BinaryHeap<std::cmp::Reverse<(Ord32, usize)>> = BinaryHeap::new();
        candidates.push(std::cmp::Reverse((Ord32(entry_d), entry)));
        // `top` is a max-heap (furthest first) bounded at ef.
        let mut top: BinaryHeap<(Ord32, usize)> = BinaryHeap::new();
        top.push((Ord32(entry_d), entry));

        while let Some(std::cmp::Reverse((cd, c))) = candidates.pop() {
            if let Some(&(furthest, _)) = top.peek()
                && cd.0 > furthest.0
            {
                break;
            }
            if let Some(layer_neighbours) = self.nodes[c].get(layer) {
                for &n in layer_neighbours {
                    if !visited.insert(n) {
                        continue;
                    }
                    let d = distance(q, &self.vectors[n]);
                    let should_push = top.len() < ef
                        || top.peek().map(|&(f, _)| d < f.0).unwrap_or(true);
                    if should_push {
                        candidates.push(std::cmp::Reverse((Ord32(d), n)));
                        top.push((Ord32(d), n));
                        if top.len() > ef {
                            top.pop();
                        }
                    }
                }
            }
        }

        top.into_iter().map(|(d, i)| (i, d.0)).collect()
    }

    fn random_level(&mut self) -> usize {
        // xorshift64* for determinism without an extra dep.
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x.max(1);
        let u = (self.rng_state as f64) / (u64::MAX as f64);
        // Clamp to a positive non-zero value before taking ln.
        let u = u.clamp(1.0e-12, 1.0 - 1.0e-12);
        (-u.ln() * self.params.level_mult as f64).floor() as usize
    }
}

fn select_neighbours_heuristic(
    vectors: &[Vec<f32>],
    _q: &[f32],
    mut candidates: Vec<(usize, f32)>,
    m: usize,
) -> Vec<usize> {
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<usize> = Vec::with_capacity(m);
    for (c, d_cq) in candidates {
        if out.len() >= m {
            break;
        }
        // Heuristic: include `c` only if it's closer to `q` than to any
        // already-selected neighbour.
        let dominated = out.iter().any(|&e| {
            let d_ce = distance(&vectors[c], &vectors[e]);
            d_ce < d_cq
        });
        if !dominated {
            out.push(c);
        }
    }
    out
}

fn l2_normalise(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn distance(a: &[f32], b: &[f32]) -> f32 {
    // Cosine distance over L2-normalised vectors == 1 - <a,b>.
    let mut dot = 0.0f32;
    let n = a.len().min(b.len());
    for i in 0..n {
        dot += a[i] * b[i];
    }
    1.0 - dot
}

/// Dense retriever — embeds + HNSW.
pub struct DenseIndex {
    embedder: Arc<dyn Embedder>,
    hnsw: Mutex<HnswIndex>,
    /// `(hnsw_id -> DocId)` plus document text.
    doc_id_map: Mutex<BTreeMap<usize, DocId>>,
    doc_text: Mutex<BTreeMap<DocId, String>>,
}

impl DenseIndex {
    /// Construct a fresh dense index.
    pub fn new(embedder: Arc<dyn Embedder>, params: HnswParams) -> Self {
        Self {
            embedder,
            hnsw: Mutex::new(HnswIndex::new(params)),
            doc_id_map: Mutex::new(BTreeMap::new()),
            doc_text: Mutex::new(BTreeMap::new()),
        }
    }

    /// Embed and insert a single document.
    pub async fn insert(&self, id: DocId, text: String) -> RerankResult<()> {
        let v = self
            .embedder
            .embed(&[text.as_str()])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| RerankError::Index("embedder returned no vectors".into()))?;
        let hnsw_id = self.hnsw.lock().expect("hnsw lock").insert(v);
        self.doc_id_map
            .lock()
            .expect("map lock")
            .insert(hnsw_id, id.clone());
        self.doc_text.lock().expect("text lock").insert(id, text);
        Ok(())
    }

    /// Embed and insert a batch.
    pub async fn insert_batch(&self, docs: Vec<(DocId, String)>) -> RerankResult<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let texts: Vec<&str> = docs.iter().map(|(_, t)| t.as_str()).collect();
        let vectors = self.embedder.embed(&texts).await?;
        if vectors.len() != docs.len() {
            return Err(RerankError::Index(format!(
                "embedder returned {} vectors for {} docs",
                vectors.len(),
                docs.len()
            )));
        }
        let mut hnsw = self.hnsw.lock().expect("hnsw lock");
        let mut id_map = self.doc_id_map.lock().expect("map lock");
        let mut text_map = self.doc_text.lock().expect("text lock");
        for ((id, text), v) in docs.into_iter().zip(vectors) {
            let hnsw_id = hnsw.insert(v);
            id_map.insert(hnsw_id, id.clone());
            text_map.insert(id, text);
        }
        Ok(())
    }

    /// Search by query text.
    pub async fn search(&self, query: &str, top_k: usize) -> RerankResult<Vec<(DocId, f32)>> {
        let q = self
            .embedder
            .embed(&[query])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| RerankError::Index("embedder returned no query vector".into()))?;
        let hits = self.hnsw.lock().expect("hnsw lock").search(&q, top_k);
        let id_map = self.doc_id_map.lock().expect("map lock");
        Ok(hits
            .into_iter()
            .filter_map(|(i, d)| id_map.get(&i).map(|id| (id.clone(), 1.0 - d)))
            .collect())
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.hnsw.lock().expect("hnsw lock").len()
    }

    /// Is the index empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl Retriever for DenseIndex {
    async fn retrieve(&self, query: &str, top_k: usize) -> RerankResult<Vec<(DocId, f32)>> {
        self.search(query, top_k).await
    }

    fn document_text(&self, id: &DocId) -> Option<String> {
        self.doc_text.lock().expect("text lock").get(id).cloned()
    }

    fn name(&self) -> &str {
        "dense-hnsw"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hnsw_round_trip_with_unit_vectors() {
        let mut idx = HnswIndex::new(HnswParams::default());
        // Three orthogonal-ish unit vectors.
        let a = idx.insert(vec![1.0, 0.0, 0.0]);
        let b = idx.insert(vec![0.0, 1.0, 0.0]);
        let c = idx.insert(vec![0.0, 0.0, 1.0]);
        let d = idx.insert(vec![0.99, 0.01, 0.0]);

        let hits = idx.search(&[1.0, 0.0, 0.0], 2);
        assert!(!hits.is_empty());
        let ids: Vec<usize> = hits.iter().map(|h| h.0).collect();
        // The two closest to (1,0,0) are `a` and `d` (cosine ~ 1.0).
        assert!(ids.contains(&a));
        assert!(ids.contains(&d));
        let _ = (b, c);
    }

    #[test]
    fn hnsw_empty_query_no_panic() {
        let idx = HnswIndex::new(HnswParams::default());
        assert!(idx.search(&[1.0, 0.0], 5).is_empty());
    }

    #[test]
    fn random_level_bounded() {
        let mut idx = HnswIndex::new(HnswParams::default());
        let levels: Vec<usize> = (0..1000).map(|_| idx.random_level()).collect();
        // Should be capped well under 64 for typical params.
        assert!(levels.iter().all(|&l| l < 64));
        // Should produce *some* level-0 nodes (most common).
        assert!(levels.contains(&0));
    }
}
