//! `Corpus` + brute-force nearest-neighbor lookup using an MRL-style
//! truncated embedding.
//!
//! R30.0 shipped the dim-picking math. R30.1.0 wires a real-shape
//! retrieval pipeline: a corpus of `(doc_id, text, embedding)` triples,
//! a query path that embeds the query, truncates to a picked dim, and
//! returns the top-k nearest documents by cosine similarity.
//!
//! Outputs are still synthetic in spirit: the [`IdentityEmbedder`] is
//! deterministic but not semantically meaningful (it's a prefix
//! reflector of the input). R30.1.1 will swap in a real trained
//! embedder (e.g. gte-small, all-MiniLM) and a meaningful corpus.

use crate::embedder::Embedder;
use crate::matryoshka::MatryoshkaEmbedder;
use crate::picker::DimPicker;

#[derive(Debug, Clone)]
pub struct CorpusDoc {
    pub id: u32,
    pub text: String,
    /// Full-dim embedding, length `full_dim`.
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct RetrievalHit {
    pub doc_id: u32,
    pub doc_text: String,
    /// Cosine similarity at the picked dim.
    pub score: f32,
    /// Dim used for this query.
    pub dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RetrievalError {
    EmbedderFailed(String),
    Empty,
    PickFailed(String),
}

impl std::fmt::Display for RetrievalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmbedderFailed(s) => write!(f, "embedder failed: {}", s),
            Self::Empty => write!(f, "empty corpus"),
            Self::PickFailed(s) => write!(f, "dim pick failed: {}", s),
        }
    }
}

impl std::error::Error for RetrievalError {}

pub struct Corpus<E: Embedder> {
    matryoshka: MatryoshkaEmbedder<E>,
    docs: Vec<CorpusDoc>,
}

impl<E: Embedder> Corpus<E> {
    pub fn new(matryoshka: MatryoshkaEmbedder<E>) -> Self {
        Self { matryoshka, docs: Vec::new() }
    }

    pub fn full_dim(&self) -> usize {
        self.matryoshka.full_dim()
    }

    pub fn matryoshka(&self) -> &MatryoshkaEmbedder<E> {
        &self.matryoshka
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Add a document, embedding it with the full-dim path. Returns the
    /// assigned `doc_id` (monotonic from 0).
    pub fn add(&mut self, text: impl Into<String>) -> Result<u32, RetrievalError> {
        let text = text.into();
        // Build a deterministic input vector from the text bytes, then embed.
        let input = text_to_embed_input(&text, self.matryoshka.full_dim());
        let embedding = self
            .matryoshka
            .embed(&input)
            .map_err(|e| RetrievalError::EmbedderFailed(e.to_string()))?;
        let id = self.docs.len() as u32;
        self.docs.push(CorpusDoc { id, text, embedding });
        Ok(id)
    }

    /// Find the top-k most similar documents to `query` at a picked dim.
    /// The pick uses [`DimPicker`] with the supplied `quality_floor` and
    /// `retrieval_budget_joules`; if the picker rejects, falls back to
    /// the full dim. Cosine similarity is used after L2-normalization.
    pub fn retrieve(
        &self,
        query: &str,
        k: usize,
        quality_floor: f32,
        retrieval_budget_joules: f64,
    ) -> Result<Vec<RetrievalHit>, RetrievalError> {
        if self.docs.is_empty() {
            return Err(RetrievalError::Empty);
        }
        // Embed the query at full dim.
        let q_input = text_to_embed_input(query, self.matryoshka.full_dim());
        let q_full = self
            .matryoshka
            .embed(&q_input)
            .map_err(|e| RetrievalError::EmbedderFailed(e.to_string()))?;

        // Pick a truncation dim.
        let dim = match DimPicker::new(&self.matryoshka)
            .pick(self.docs.len(), quality_floor, retrieval_budget_joules)
        {
            Ok(d) => d,
            Err(_) => self.matryoshka.full_dim(),
        };

        let q_trunc = &q_full[..dim];
        let q_norm = l2_norm(q_trunc);

        // Score every doc by cosine similarity at the picked dim.
        let mut scored: Vec<(usize, f32)> = self
            .docs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let d_trunc = &d.embedding[..dim];
                let dot: f32 = q_trunc
                    .iter()
                    .zip(d_trunc.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let d_norm = l2_norm(d_trunc);
                let score = if q_norm > 0.0 && d_norm > 0.0 {
                    dot / (q_norm * d_norm)
                } else {
                    0.0
                };
                (i, score)
            })
            .collect();

        // Top-k by descending score, stable for ties (preserves doc_id order).
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k.max(1));

        Ok(scored
            .into_iter()
            .map(|(i, score)| RetrievalHit {
                doc_id: self.docs[i].id,
                doc_text: self.docs[i].text.clone(),
                score,
                dim,
            })
            .collect())
    }

    /// Joule estimate for one retrieve(): embed pass + retrieval at picked dim.
    pub fn retrieve_joules(&self, picked_dim: usize) -> f64 {
        self.matryoshka.embed_joules()
            + self.matryoshka.retrieval_joules(picked_dim, self.docs.len())
    }
}

/// Deterministic text → embedder-input. Hashes the text bytes into a
/// fixed-length vector. Used to give the synthetic IdentityEmbedder
/// something structured to project; real embedders take token IDs and
/// run their own tokenizer.
pub fn text_to_embed_input(text: &str, len: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; len];
    let bytes = text.as_bytes();
    // Splat bytes into the buffer, then mix with a small xorshift so
    // different strings produce different vectors.
    for (i, &b) in bytes.iter().enumerate() {
        let pos = i % len;
        let mut h = (b as u64).wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
        h ^= h >> 27;
        h = h.wrapping_mul(0x94D049BB133111EB);
        h ^= h >> 31;
        let f = ((h & 0xFFFF) as f32) / 32768.0 - 1.0; // [-1, 1)
        out[pos] += f;
    }
    // Light normalization to keep magnitudes reasonable.
    let n = l2_norm(&out).max(1e-6);
    for v in out.iter_mut() {
        *v /= n;
    }
    out
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::IdentityEmbedder;
    use crate::matryoshka::MatryoshkaEmbedder;

    fn small_corpus() -> Corpus<IdentityEmbedder> {
        let m = MatryoshkaEmbedder::with_powers_of_two(IdentityEmbedder::new(64));
        let mut c = Corpus::new(m);
        c.add("hello world").unwrap();
        c.add("the lawful synthesizer").unwrap();
        c.add("ternary quantization").unwrap();
        c.add("matryoshka embeddings").unwrap();
        c
    }

    #[test]
    fn corpus_size_grows_with_add() {
        let mut c = Corpus::new(MatryoshkaEmbedder::with_powers_of_two(
            IdentityEmbedder::new(32),
        ));
        assert_eq!(c.len(), 0);
        c.add("a").unwrap();
        c.add("b").unwrap();
        assert_eq!(c.len(), 2);
    }

    // Quality floor that forces the picker to choose a meaningful dim.
    // With QualityModel::default_for(64), q(d) = 1 - 0.022 * ln(64/d);
    // 0.99 requires d ≈ 41 → d = 64 in the power-of-2 ladder, which is
    // enough resolution for cosine similarity to discriminate.
    const HIGH_Q: f32 = 0.99;
    const BIG_BUDGET: f64 = 1.0;

    #[test]
    fn retrieve_returns_at_most_k_hits() {
        let c = small_corpus();
        let hits = c.retrieve("hello", 2, HIGH_Q, BIG_BUDGET).unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn retrieve_returns_self_match_at_top_when_query_matches_doc() {
        let c = small_corpus();
        // Querying a doc's text should put that doc at the top.
        let hits = c.retrieve("the lawful synthesizer", 1, HIGH_Q, BIG_BUDGET).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].score > 0.95, "self-match score too low: {}", hits[0].score);
        assert_eq!(hits[0].doc_text, "the lawful synthesizer");
    }

    #[test]
    fn retrieve_is_deterministic() {
        let c = small_corpus();
        let a = c.retrieve("hello", 4, HIGH_Q, BIG_BUDGET).unwrap();
        let b = c.retrieve("hello", 4, HIGH_Q, BIG_BUDGET).unwrap();
        assert_eq!(a.len(), b.len());
        for (ha, hb) in a.iter().zip(b.iter()) {
            assert_eq!(ha.doc_id, hb.doc_id);
            assert_eq!(ha.score, hb.score);
        }
    }

    #[test]
    fn retrieve_on_empty_corpus_errors() {
        let c = Corpus::new(MatryoshkaEmbedder::with_powers_of_two(
            IdentityEmbedder::new(32),
        ));
        assert!(matches!(c.retrieve("x", 1, HIGH_Q, BIG_BUDGET), Err(RetrievalError::Empty)));
    }

    #[test]
    fn picked_dim_is_recorded_in_hits() {
        let c = small_corpus();
        let hits = c.retrieve("hello", 1, HIGH_Q, BIG_BUDGET).unwrap();
        assert!(hits[0].dim > 0);
        assert!(hits[0].dim <= 64);
    }

    #[test]
    fn text_to_embed_input_is_deterministic() {
        let a = text_to_embed_input("hello", 16);
        let b = text_to_embed_input("hello", 16);
        assert_eq!(a, b);
        let c = text_to_embed_input("world", 16);
        assert_ne!(a, c, "different text should give different input vector");
    }

    #[test]
    fn lower_quality_floor_picks_smaller_dim() {
        // High quality floor (0.99) forces ~full dim; low quality (0.92)
        // allows a much smaller dim. Picked dim is recorded on each hit.
        let c = small_corpus();
        let strict = c.retrieve("x", 1, 0.99, BIG_BUDGET).unwrap();
        let loose = c.retrieve("x", 1, 0.92, BIG_BUDGET).unwrap();
        assert!(
            loose[0].dim <= strict[0].dim,
            "looser quality should pick a smaller dim: loose={} strict={}",
            loose[0].dim, strict[0].dim
        );
    }
}
