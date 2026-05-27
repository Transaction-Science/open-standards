//! Fusion strategies for federated search.
//!
//! A federation dispatch produces N hit lists (one per provider). The
//! [`Fuser`] trait reduces those to a single ranked list, deduplicated
//! by URL.
//!
//! Two fusers ship in-tree:
//!
//! - [`LinearFuser`] — sums per-provider scores (optionally weighted)
//!   for each URL. Cheap; good for providers whose scores are already
//!   normalised on `[0, 1]`.
//! - [`RrfFuser`] — Reciprocal Rank Fusion. Ignores raw scores and
//!   fuses on rank position; the canonical robust fuser for hetero-
//!   scored providers (Cormack/Clarke/Buettcher SIGIR 2009).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::provider::SearchHit;

/// One row in the fused output list.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FusedHit {
    /// Canonical URL — the dedup key.
    pub url: String,
    /// Title from the first provider that surfaced this URL.
    pub title: String,
    /// Snippet from the first provider that surfaced this URL.
    pub snippet: String,
    /// Combined fused score in `[0.0, 1.0]`. Higher = better.
    pub score: f32,
    /// All providers that returned this URL, in surface order. Diversity
    /// is a load-bearing signal for the federation tier's confidence.
    pub sources: Vec<String>,
}

/// Summary of one fusion pass — fed into the federation tier's
/// confidence calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FuseReport {
    /// Top fused score in the output, or 0.0 when empty.
    pub top_score: f32,
    /// How many distinct providers returned at least one hit.
    pub successful_providers: usize,
    /// Total fused-hit count after dedup + truncation.
    pub hit_count: usize,
}

/// The fusion trait. Implementations consume a slice of
/// `(provider_name, hit_list)` pairs and return a fused ranked list
/// plus a [`FuseReport`].
pub trait Fuser: Send + Sync {
    /// Run the fusion. Implementations MUST be deterministic — the
    /// federation tier's confidence is sensitive to fuser stability.
    fn fuse(
        &self,
        per_provider: &[(String, Vec<SearchHit>)],
        k: usize,
    ) -> (Vec<FusedHit>, FuseReport);

    /// Human-readable name for logs and receipts.
    fn name(&self) -> &str;
}

// ─── LinearFuser ─────────────────────────────────────────────────────

/// Linear-combination fuser. For each provider, multiplies hits by the
/// provider's weight (default 1.0), then sums per-URL.
///
/// Final scores are min-max normalised to `[0, 1]` so downstream
/// confidence is comparable across queries.
#[derive(Debug, Clone)]
pub struct LinearFuser {
    /// Per-provider weights — providers absent from the map get `1.0`.
    weights: HashMap<String, f32>,
}

impl Default for LinearFuser {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearFuser {
    /// Equal-weighted linear fuser.
    pub fn new() -> Self {
        Self { weights: HashMap::new() }
    }

    /// Override the weight for one provider. Weights above 1.0 emphasise
    /// that provider; weights below 1.0 deemphasise it. Negative weights
    /// are clamped to 0.0.
    pub fn with_weight(mut self, provider: impl Into<String>, w: f32) -> Self {
        self.weights.insert(provider.into(), w.max(0.0));
        self
    }

    fn weight(&self, provider: &str) -> f32 {
        self.weights.get(provider).copied().unwrap_or(1.0)
    }
}

impl Fuser for LinearFuser {
    fn name(&self) -> &str {
        "linear"
    }

    fn fuse(
        &self,
        per_provider: &[(String, Vec<SearchHit>)],
        k: usize,
    ) -> (Vec<FusedHit>, FuseReport) {
        fuse_dedup(per_provider, k, |hit, provider| {
            hit.score.max(0.0) * self.weight(provider)
        })
    }
}

// ─── RrfFuser ────────────────────────────────────────────────────────

/// Reciprocal-Rank Fusion. Ignores provider-reported scores; sums
/// `1 / (rrf_k + rank)` across providers, where `rank` is the 1-based
/// position within that provider's hit list.
///
/// The donor `verity-federation` used a `rrf_k` of 60 (Cormack et al.).
/// The same constant is the default here; consumers can override.
#[derive(Debug, Clone)]
pub struct RrfFuser {
    /// RRF damping constant — donor default is 60.
    pub rrf_k: f32,
}

impl Default for RrfFuser {
    fn default() -> Self {
        Self { rrf_k: 60.0 }
    }
}

impl RrfFuser {
    /// Build an RRF fuser with the given damping constant.
    pub fn new(rrf_k: f32) -> Self {
        Self { rrf_k: rrf_k.max(0.0) }
    }
}

impl Fuser for RrfFuser {
    fn name(&self) -> &str {
        "rrf"
    }

    fn fuse(
        &self,
        per_provider: &[(String, Vec<SearchHit>)],
        k: usize,
    ) -> (Vec<FusedHit>, FuseReport) {
        // RRF needs per-hit rank; rebuild per-provider with rank-derived score.
        let mut per_provider_rrf: Vec<(String, Vec<SearchHit>)> = Vec::with_capacity(per_provider.len());
        for (name, hits) in per_provider {
            let mut rebuilt: Vec<SearchHit> = Vec::with_capacity(hits.len());
            for (rank, hit) in hits.iter().enumerate() {
                let rrf_score = 1.0 / (self.rrf_k + (rank as f32) + 1.0);
                let mut h = hit.clone();
                h.score = rrf_score;
                rebuilt.push(h);
            }
            per_provider_rrf.push((name.clone(), rebuilt));
        }
        fuse_dedup(&per_provider_rrf, k, |hit, _provider| hit.score)
    }
}

// ─── Shared dedup + normalise pipeline ───────────────────────────────

/// Common dedup-by-URL + sort + normalise pipeline used by every fuser.
/// `score_of(hit, provider_name)` returns the per-provider contribution
/// for a single hit.
fn fuse_dedup<F>(
    per_provider: &[(String, Vec<SearchHit>)],
    k: usize,
    score_of: F,
) -> (Vec<FusedHit>, FuseReport)
where
    F: Fn(&SearchHit, &str) -> f32,
{
    // Order-preserving accumulator keyed by URL.
    let mut order: Vec<String> = Vec::new();
    let mut by_url: HashMap<String, FusedHit> = HashMap::new();
    let mut successful_providers: usize = 0;

    for (provider, hits) in per_provider {
        if hits.is_empty() {
            continue;
        }
        successful_providers += 1;
        for hit in hits {
            let contribution = score_of(hit, provider);
            match by_url.get_mut(&hit.url) {
                Some(existing) => {
                    existing.score += contribution;
                    if !existing.sources.contains(provider) {
                        existing.sources.push(provider.clone());
                    }
                }
                None => {
                    order.push(hit.url.clone());
                    by_url.insert(
                        hit.url.clone(),
                        FusedHit {
                            url: hit.url.clone(),
                            title: hit.title.clone(),
                            snippet: hit.snippet.clone(),
                            score: contribution,
                            sources: vec![provider.clone()],
                        },
                    );
                }
            }
        }
    }

    let mut out: Vec<FusedHit> = order
        .into_iter()
        .filter_map(|url| by_url.remove(&url))
        .collect();

    // Sort by raw score descending, then by URL for determinism.
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.url.cmp(&b.url))
    });
    out.truncate(k);

    // Normalise to [0, 1].
    let max_score = out.iter().map(|h| h.score).fold(0.0f32, f32::max);
    if max_score > 0.0 {
        for h in &mut out {
            h.score = (h.score / max_score).clamp(0.0, 1.0);
        }
    }

    let top_score = out.first().map(|h| h.score).unwrap_or(0.0);
    let hit_count = out.len();
    (
        out,
        FuseReport {
            top_score,
            successful_providers,
            hit_count,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(provider: &str, url: &str, score: f32) -> SearchHit {
        SearchHit {
            title: format!("title for {url}"),
            url: url.into(),
            snippet: format!("snippet for {url}"),
            score,
            source: provider.into(),
        }
    }

    fn three_providers() -> Vec<(String, Vec<SearchHit>)> {
        vec![
            (
                "brave".into(),
                vec![
                    hit("brave", "https://a", 0.9),
                    hit("brave", "https://b", 0.5),
                ],
            ),
            (
                "wikipedia".into(),
                vec![
                    hit("wikipedia", "https://a", 0.8),
                    hit("wikipedia", "https://c", 0.7),
                ],
            ),
            (
                "arxiv".into(),
                vec![hit("arxiv", "https://b", 0.6)],
            ),
        ]
    }

    #[test]
    fn linear_fuser_sums_scores_and_dedups() {
        let (out, report) = LinearFuser::default().fuse(&three_providers(), 10);
        // a is in brave + wikipedia, b is in brave + arxiv, c only in wikipedia.
        assert_eq!(out.len(), 3);
        assert_eq!(report.successful_providers, 3);
        // a should win: 0.9 + 0.8 = 1.7 (max)
        assert_eq!(out[0].url, "https://a");
        assert_eq!(out[0].sources.len(), 2);
        // Normalised: top is exactly 1.0
        assert!((out[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linear_fuser_respects_weights() {
        let fuser = LinearFuser::new().with_weight("wikipedia", 3.0);
        let providers = vec![
            ("brave".into(), vec![hit("brave", "https://a", 0.9)]),
            ("wikipedia".into(), vec![hit("wikipedia", "https://b", 0.5)]),
        ];
        let (out, _r) = fuser.fuse(&providers, 10);
        // weighted wikipedia: 0.5 * 3.0 = 1.5; brave: 0.9 * 1.0 = 0.9.
        assert_eq!(out[0].url, "https://b");
    }

    #[test]
    fn rrf_fuser_uses_rank() {
        let (out, _r) = RrfFuser::default().fuse(&three_providers(), 10);
        // RRF: a is rank-0 in brave + rank-0 in wikipedia
        //      → 1/61 + 1/61 = ~0.0328 (highest)
        assert_eq!(out[0].url, "https://a");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn rrf_respects_custom_k() {
        let f = RrfFuser::new(10.0);
        let (out, _r) = f.fuse(&three_providers(), 10);
        assert_eq!(out.len(), 3);
        // Sanity: top is normalised to 1.0
        assert!((out[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fuser_truncates_to_k() {
        let providers = vec![
            (
                "brave".into(),
                vec![
                    hit("brave", "https://a", 0.9),
                    hit("brave", "https://b", 0.8),
                    hit("brave", "https://c", 0.7),
                    hit("brave", "https://d", 0.6),
                ],
            ),
        ];
        let (out, report) = LinearFuser::default().fuse(&providers, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(report.hit_count, 2);
    }

    #[test]
    fn empty_provider_lists_yield_empty_fuse() {
        let providers: Vec<(String, Vec<SearchHit>)> = vec![
            ("brave".into(), vec![]),
            ("wikipedia".into(), vec![]),
        ];
        let (out, report) = LinearFuser::default().fuse(&providers, 10);
        assert!(out.is_empty());
        assert_eq!(report.successful_providers, 0);
        assert_eq!(report.top_score, 0.0);
    }

    #[test]
    fn fused_hit_serdes_roundtrip() {
        let h = FusedHit {
            url: "u".into(),
            title: "t".into(),
            snippet: "s".into(),
            score: 0.5,
            sources: vec!["a".into(), "b".into()],
        };
        let bytes = serde_json::to_vec(&h).expect("ser");
        let back: FusedHit = serde_json::from_slice(&bytes).expect("deser");
        assert_eq!(back, h);
    }
}
