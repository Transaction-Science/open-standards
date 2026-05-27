//! The [`SearchProvider`] trait — the boundary between the federation
//! tier and the consumer's actual search adapters.
//!
//! JouleClaw deliberately ships **no live providers** in this crate.
//! Downstream consumers supply their own adapters (Brave, Bing,
//! Wikipedia, an internal Elastic, a local sled index, …) by
//! implementing this trait. The donor's `verity-federation` crate
//! contains 28 such adapters; porting them is out-of-scope for the L2
//! orchestrator and would drag in `reqwest` / `async` / vendor
//! credentials this layer should not know about.

use serde::{Deserialize, Serialize};

/// A single search hit produced by one provider.
///
/// Hits are the unit of fusion: the federation tier collects one
/// `Vec<SearchHit>` per provider, then a [`Fuser`](crate::Fuser)
/// reduces them to a single ranked list.
///
/// `score` is the provider's own per-hit relevance; the fuser is
/// responsible for normalising across providers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchHit {
    /// Human-readable title of the result.
    pub title: String,
    /// Canonical URL — the dedup key inside the fuser.
    pub url: String,
    /// Short snippet / abstract.
    pub snippet: String,
    /// Provider-local relevance score in `[0.0, 1.0]`. Higher = better.
    /// The fuser normalises across providers.
    pub score: f32,
    /// The provider's [`SearchProvider::name`]. Set by the federation
    /// orchestrator when collecting hits; provider impls may leave
    /// this empty.
    pub source: String,
}

impl SearchHit {
    /// Construct a [`SearchHit`] with sensible defaults — useful for
    /// tests and mock providers.
    pub fn new(
        title: impl Into<String>,
        url: impl Into<String>,
        snippet: impl Into<String>,
        score: f32,
    ) -> Self {
        Self {
            title: title.into(),
            url: url.into(),
            snippet: snippet.into(),
            score: score.clamp(0.0, 1.0),
            source: String::new(),
        }
    }
}

/// Errors a provider may report. Provider failures are isolated: one
/// failing provider does NOT fail the whole federation dispatch.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// Transient transport error (timeout, DNS, refused).
    #[error("provider transport error: {0}")]
    Transport(String),
    /// Provider returned a structured error (HTTP 4xx/5xx, JSON error
    /// envelope, rate limit). Callers SHOULD include a short reason.
    #[error("provider rejected query: {0}")]
    Rejected(String),
    /// Provider authentication failure (missing or expired credentials).
    #[error("provider auth failure: {0}")]
    Auth(String),
    /// Provider exceeded the orchestrator's deadline.
    #[error("provider timeout")]
    Timeout,
    /// Catch-all for adapter-specific failures.
    #[error("provider error: {0}")]
    Other(String),
}

/// The provider trait.
///
/// Consumer crates implement this once per search backend. The trait is
/// **synchronous** — the federation orchestrator runs each provider on
/// its own OS thread via `std::thread::scope`, which keeps the
/// federation crate's dependency footprint minimal and works on the
/// embedded / no-async targets JouleClaw must support.
///
/// Implementations MUST be `Send + Sync` so the orchestrator can hand
/// them to scoped threads.
pub trait SearchProvider: Send + Sync {
    /// Stable identifier for this provider — used as a dedup key in
    /// the fuser and surfaced in receipts. Convention: lower-snake
    /// (`brave`, `wikipedia`, `local_arxiv`).
    fn name(&self) -> &str;

    /// Run a query and return at most `k` hits. On error, the
    /// orchestrator counts this provider as a refusal but continues
    /// dispatching the others.
    fn search(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchHit>, ProviderError>;

    /// Self-reported "typical" energy cost per call, in joules. Used
    /// by [`Federation::estimate_cost`](crate::Federation::estimate_cost) to advertise the federation's
    /// total estimated joule spend up the cascade. Implementations
    /// SHOULD report an order-of-magnitude honest value; over-claiming
    /// will make the tier look more expensive than it is and pushes
    /// the cascade past it; under-claiming risks budget violations.
    fn typical_joules_per_call(&self) -> f64;
}

// ─── Mock provider for tests + smoke runs ────────────────────────────

/// A deterministic mock provider for unit tests and smoke runs.
///
/// Returns `k` hits whose titles encode the provider name, the query,
/// and the hit index, with linearly decreasing scores. Energy is
/// reported as a configurable constant (default 100 µJ).
///
/// `MockProvider` is part of the public API because the donor's tests
/// relied on the equivalent mock fixtures; downstream consumers benefit
/// from the same affordance when wiring fakes.
#[derive(Debug, Clone)]
pub struct MockProvider {
    name: String,
    typical_joules: f64,
    /// When set, [`MockProvider::search`] returns this error instead
    /// of hits — used by tests to exercise the per-provider failure
    /// isolation path.
    force_error: Option<String>,
    /// When set, returns this many hits regardless of `k`. `None`
    /// (the default) respects the caller's `k`.
    fixed_hit_count: Option<usize>,
}

impl MockProvider {
    /// Build a mock with a default 100 µJ-per-call cost.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            typical_joules: 100e-6,
            force_error: None,
            fixed_hit_count: None,
        }
    }

    /// Override the self-reported energy cost. Useful for tests that
    /// want to exercise the federation's joule-sum.
    pub fn with_joules(mut self, joules: f64) -> Self {
        self.typical_joules = joules;
        self
    }

    /// Configure this provider to always return [`ProviderError::Rejected`]
    /// with the given reason — for testing failure-isolation.
    pub fn with_forced_error(mut self, reason: impl Into<String>) -> Self {
        self.force_error = Some(reason.into());
        self
    }

    /// Pin the hit count returned, ignoring the caller's `k`.
    pub fn with_fixed_hit_count(mut self, n: usize) -> Self {
        self.fixed_hit_count = Some(n);
        self
    }
}

impl SearchProvider for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn search(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchHit>, ProviderError> {
        if let Some(reason) = &self.force_error {
            return Err(ProviderError::Rejected(reason.clone()));
        }
        let n = self.fixed_hit_count.unwrap_or(k).min(k.max(1));
        let mut hits = Vec::with_capacity(n);
        for i in 0..n {
            // Score decays linearly from 0.9 to 0.1 over the hit list.
            let score = if n == 1 {
                0.9
            } else {
                (0.9 - (i as f32) * 0.8 / (n.saturating_sub(1) as f32))
                    .clamp(0.0, 1.0)
            };
            hits.push(SearchHit {
                title: format!("[{}] {} #{i}", self.name, query),
                url: format!("https://{}.example/{i}", self.name),
                snippet: format!("mock snippet for {query} from {}", self.name),
                score,
                source: self.name.clone(),
            });
        }
        Ok(hits)
    }

    fn typical_joules_per_call(&self) -> f64 {
        self.typical_joules
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_returns_k_hits() {
        let p = MockProvider::named("brave");
        let hits = p.search("rust", 3).expect("ok");
        assert_eq!(hits.len(), 3);
        assert!(hits[0].score >= hits[1].score);
        assert!(hits[1].score >= hits[2].score);
        assert_eq!(hits[0].source, "brave");
    }

    #[test]
    fn mock_forced_error() {
        let p = MockProvider::named("brave").with_forced_error("rate limit");
        let r = p.search("rust", 3);
        assert!(matches!(r, Err(ProviderError::Rejected(_))));
    }

    #[test]
    fn mock_with_joules_override() {
        let p = MockProvider::named("brave").with_joules(2e-3);
        assert!((p.typical_joules_per_call() - 2e-3).abs() < 1e-12);
    }

    #[test]
    fn search_hit_new_clamps_score() {
        let h = SearchHit::new("t", "u", "s", 2.0);
        assert_eq!(h.score, 1.0);
        let h2 = SearchHit::new("t", "u", "s", -1.0);
        assert_eq!(h2.score, 0.0);
    }

    #[test]
    fn search_hit_roundtrips_serde() {
        let h = SearchHit {
            title: "T".into(),
            url: "u".into(),
            snippet: "s".into(),
            score: 0.5,
            source: "p".into(),
        };
        let bytes = serde_json::to_vec(&h).expect("ser");
        let back: SearchHit = serde_json::from_slice(&bytes).expect("deser");
        assert_eq!(back, h);
    }
}
