//! Memory retrieval: similarity + recency + frequency triangulation.
//!
//! Following the classical SAM / ACT-R activation model (Anderson
//! 1983), an item's "activation" combines three signals:
//!
//! * **Similarity** — cosine-similarity to the query embedding.
//! * **Recency** — Ebbinghaus retention given the last access time.
//! * **Frequency** — saturating function of access count.
//!
//! The final score is a weighted linear combination. Weights are
//! configurable and default to `(0.6, 0.25, 0.15)`.

use serde::{Deserialize, Serialize};

use crate::episodic::EpisodicLog;
use crate::error::{MemoryError, MemoryResult};
use crate::forget::EbbinghausScorer;
use crate::memory::{EpisodeId, Memory, MemoryItem};

/// Configuration for [`Retriever`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Weight on cosine similarity in `[0.0, 1.0]`.
    pub w_similarity: f32,
    /// Weight on Ebbinghaus recency in `[0.0, 1.0]`.
    pub w_recency: f32,
    /// Weight on frequency in `[0.0, 1.0]`.
    pub w_frequency: f32,
    /// Number of items to return.
    pub top_k: usize,
}

impl RetrievalConfig {
    /// Construct + validate.
    pub fn new(w_sim: f32, w_rec: f32, w_freq: f32, top_k: usize) -> MemoryResult<Self> {
        for (name, w) in [("w_sim", w_sim), ("w_rec", w_rec), ("w_freq", w_freq)] {
            if !(0.0..=1.0).contains(&w) {
                return Err(MemoryError::Config(format!("{name} must be in [0,1]")));
            }
        }
        if top_k == 0 {
            return Err(MemoryError::Config("top_k must be > 0".into()));
        }
        Ok(Self {
            w_similarity: w_sim,
            w_recency: w_rec,
            w_frequency: w_freq,
            top_k,
        })
    }
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            w_similarity: 0.6,
            w_recency: 0.25,
            w_frequency: 0.15,
            top_k: 8,
        }
    }
}

/// One scored retrieval candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalScore {
    /// Episode id of the retrieved record.
    pub episode_id: EpisodeId,
    /// Final blended score in `[0.0, 1.0]`.
    pub score: f32,
    /// Similarity sub-score (cosine).
    pub sim: f32,
    /// Recency sub-score (Ebbinghaus retention).
    pub recency: f32,
    /// Frequency sub-score (saturating).
    pub frequency: f32,
}

/// Triangulating retriever over an [`EpisodicLog`].
pub struct Retriever<'a> {
    log: &'a EpisodicLog,
    cfg: RetrievalConfig,
    scorer: EbbinghausScorer,
}

impl<'a> Retriever<'a> {
    /// New retriever over `log` with `cfg` and `scorer`.
    #[must_use]
    pub fn new(log: &'a EpisodicLog, cfg: RetrievalConfig, scorer: EbbinghausScorer) -> Self {
        Self { log, cfg, scorer }
    }

    /// Rank episodes against the supplied query embedding at `now_ms`.
    pub fn rank(&self, query: &[f32], now_ms: u64) -> MemoryResult<Vec<RetrievalScore>> {
        let mut scored: Vec<RetrievalScore> = Vec::with_capacity(self.log.len());
        for ep in self.log.all() {
            let sim = ep
                .embedding
                .as_deref()
                .map(|v| cosine(query, v))
                .unwrap_or(0.0);
            let recency =
                self.scorer
                    .retention(ep.timestamp_ms, now_ms, ep.access_count) as f32;
            let frequency = freq_score(ep.access_count);
            let score = self.cfg.w_similarity * sim
                + self.cfg.w_recency * recency
                + self.cfg.w_frequency * frequency;
            scored.push(RetrievalScore {
                episode_id: ep.id,
                score,
                sim,
                recency,
                frequency,
            });
        }
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.episode_id.0.cmp(&a.episode_id.0))
        });
        scored.truncate(self.cfg.top_k);
        Ok(scored)
    }

    /// Materialise the top-K ranked episodes as [`MemoryItem`]s.
    pub fn retrieve(&self, query: &[f32], now_ms: u64) -> MemoryResult<Vec<MemoryItem>> {
        let scores = self.rank(query, now_ms)?;
        let mut items = Vec::with_capacity(scores.len());
        for s in scores {
            if let Some(ep) = self.log.get(&s.episode_id) {
                items.push(ep.as_item());
            }
        }
        Ok(items)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt()) * (nb.sqrt());
    if denom == 0.0 {
        0.0
    } else {
        let c = dot / denom;
        if c < 0.0 {
            0.0
        } else if c > 1.0 {
            1.0
        } else {
            c
        }
    }
}

fn freq_score(access_count: u32) -> f32 {
    // Saturating: 1 - 1/(1+n)
    let n = access_count as f32;
    1.0 - 1.0 / (1.0 + n)
}
