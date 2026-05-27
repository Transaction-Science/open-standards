//! L2 — Federation tier.
//!
//! Dispatches one query to N providers in parallel via
//! `std::thread::scope`, fuses the per-provider hit lists with a
//! consumer-chosen [`Fuser`](crate::Fuser), and returns the fused list as
//! `AnswerOutput::Structured(json)`.
//!
//! ## Mapping to the `Tier` trait
//!
//! - `id` → `TierId::L2(<configured L2ModelId>)` (default `L2ModelId(0)`).
//! - `estimate_cost` → `Σ provider.typical_joules_per_call()` joules,
//!   ~300 ms latency, confidence floor 0.5. `None` for non-text inputs
//!   or when zero providers are registered.
//! - `try_answer` → fans out, fuses, returns
//!   `AnswerOutput::Structured(FederationOutput)` on success, or
//!   `Refused(Inapplicable)` when *every* provider failed (or all
//!   returned empty).

use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, L2ModelId, Query,
    QueryInput, RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;
use jouleclaw_energy::Provenance;

use crate::fuser::{FusedHit, FuseReport, Fuser, LinearFuser};
use crate::provider::{ProviderError, SearchHit, SearchProvider};

// ─── Cost model ──────────────────────────────────────────────────────

/// Donor envelope target latency: ~300 ms wall-clock for the slowest
/// provider in a parallel fanout.
pub const DEFAULT_FEDERATION_LATENCY: Duration = Duration::from_millis(300);

/// Confidence floor advertised to the runtime — the lowest confidence
/// we are willing to claim before refusing the dispatch. Mirrors the
/// donor's `min_confidence` default for the federation layer.
pub const FEDERATION_CONFIDENCE_FLOOR: f32 = 0.5;

/// Hard ceiling on hits surfaced in the structured output. Mirrors the
/// donor's `limit` argument; consumers can override via
/// [`Federation::with_max_hits`].
pub const MAX_HITS_OUT: usize = 25;

/// Default per-provider hit count requested when the consumer does not
/// override. Mirrors the donor's default-10.
pub const DEFAULT_PER_PROVIDER_K: usize = 10;

// ─── Errors ──────────────────────────────────────────────────────────

/// Errors specific to the federation tier.
#[derive(Debug, thiserror::Error)]
pub enum FederationError {
    /// Failed to serialise the structured output payload.
    #[error("failed to serialise federation output: {0}")]
    Serialise(#[from] serde_json::Error),
}

// ─── Public output shape ─────────────────────────────────────────────

/// Per-provider outcome surfaced in the federation output. Downstream
/// tiers (reranker, model, observability) use this to attribute fused
/// hits back to their source providers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FederationProviderReport {
    /// `SearchProvider::name` of the provider.
    pub name: String,
    /// Number of hits this provider returned (0 if it failed).
    pub hits_returned: usize,
    /// Joules this provider self-reported as its typical cost.
    pub typical_joules: f64,
    /// Outcome — empty string on success; on failure, the
    /// [`ProviderError`] message.
    pub error: String,
}

/// Structured payload returned in `AnswerOutput::Structured`. Downstream
/// tiers (L2.5 reranker, L3 reader) deserialise this to recover the
/// fused hit list without re-issuing network calls.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FederationOutput {
    /// Echoes the original query text.
    pub query: String,
    /// Names of providers consulted, in registration order.
    pub providers_used: Vec<String>,
    /// Per-provider outcome (success or error).
    pub provider_reports: Vec<FederationProviderReport>,
    /// Fused, deduped, top-k hit list.
    pub hits: Vec<FusedHit>,
    /// Name of the fuser used (`"linear"`, `"rrf"`, …).
    pub fuser: String,
    /// Joules spent — sum of all providers' self-reported costs.
    pub joules_spent: f64,
}

// ─── The tier ────────────────────────────────────────────────────────

/// L2 Federation tier. Holds a [`Vec<Box<dyn SearchProvider>>`] plus a
/// pluggable [`Fuser`].
pub struct Federation {
    /// L2 sub-id surfaced via `TierId::L2(...)`. Consumers can register
    /// multiple federation tiers with different model ids.
    model_id: L2ModelId,
    /// Provider adapters. Order is preserved in `providers_used` and
    /// in the per-provider report list.
    providers: Vec<Box<dyn SearchProvider>>,
    /// Fusion strategy.
    fuser: Box<dyn Fuser>,
    /// Per-provider hit count requested.
    per_provider_k: usize,
    /// Hard ceiling on output hits.
    max_hits: usize,
}

impl Federation {
    /// Build a federation tier.
    pub fn new(
        model_id: L2ModelId,
        providers: Vec<Box<dyn SearchProvider>>,
        fuser: Box<dyn Fuser>,
    ) -> Self {
        Self {
            model_id,
            providers,
            fuser,
            per_provider_k: DEFAULT_PER_PROVIDER_K,
            max_hits: MAX_HITS_OUT,
        }
    }

    /// Convenience: build a default federation with the linear fuser
    /// and `L2ModelId(0)`.
    pub fn with_providers(providers: Vec<Box<dyn SearchProvider>>) -> Self {
        Self::new(L2ModelId(0), providers, Box::new(LinearFuser::default()))
    }

    /// Override the per-provider hit count requested.
    pub fn with_per_provider_k(mut self, k: usize) -> Self {
        self.per_provider_k = k.max(1);
        self
    }

    /// Override the cap on fused output hits.
    pub fn with_max_hits(mut self, n: usize) -> Self {
        self.max_hits = n.max(1);
        self
    }

    /// Number of providers registered.
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    /// Names of providers, in registration order.
    pub fn provider_names(&self) -> Vec<String> {
        self.providers.iter().map(|p| p.name().to_string()).collect()
    }

    /// Provenance tag for any energy spend reported by this tier. The
    /// L2 envelope is provider-self-reported (no hardware shunt), so
    /// [`Provenance::Estimator`] is the honest label.
    pub const fn provenance() -> Provenance {
        Provenance::Estimator
    }

    /// Sum of providers' `typical_joules_per_call` — the federation's
    /// pre-call energy estimate.
    fn total_typical_joules(&self) -> f64 {
        self.providers
            .iter()
            .map(|p| p.typical_joules_per_call().max(0.0))
            .sum()
    }

    // ─── Internal pipeline ───────────────────────────────────────────

    /// Dispatch every provider in parallel on its own OS thread and
    /// collect per-provider outcomes.
    ///
    /// Provider failures are isolated: a single panicking or erroring
    /// provider does not poison the rest of the fanout.
    fn dispatch_parallel(
        &self,
        query: &str,
    ) -> Vec<(usize, Result<Vec<SearchHit>, ProviderError>)> {
        // Pre-size the output slots so we can return outcomes in
        // registration order regardless of thread completion order.
        let slots: Vec<Mutex<Option<Result<Vec<SearchHit>, ProviderError>>>> =
            (0..self.providers.len()).map(|_| Mutex::new(None)).collect();

        std::thread::scope(|scope| {
            for (idx, provider) in self.providers.iter().enumerate() {
                let slot = &slots[idx];
                let k = self.per_provider_k;
                let q = query;
                let prov: &dyn SearchProvider = provider.as_ref();
                scope.spawn(move || {
                    let result = prov.search(q, k);
                    // Lock-poisoning here means a previous handler on the
                    // same Mutex panicked — impossible because we just
                    // created it — but handle gracefully anyway.
                    if let Ok(mut g) = slot.lock() {
                        *g = Some(result);
                    }
                });
            }
        });

        slots
            .into_iter()
            .enumerate()
            .map(|(idx, m)| {
                let inner = m
                    .into_inner()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                (
                    idx,
                    inner.unwrap_or_else(|| {
                        Err(ProviderError::Other(
                            "provider task did not complete".into(),
                        ))
                    }),
                )
            })
            .collect()
    }

    /// Build the per-provider report list and the
    /// `(provider_name, hits)` shape the fuser consumes.
    fn assemble(
        &self,
        outcomes: Vec<(usize, Result<Vec<SearchHit>, ProviderError>)>,
    ) -> (
        Vec<FederationProviderReport>,
        Vec<(String, Vec<SearchHit>)>,
        usize,
    ) {
        let mut reports = Vec::with_capacity(outcomes.len());
        let mut per_provider: Vec<(String, Vec<SearchHit>)> =
            Vec::with_capacity(outcomes.len());
        let mut successes: usize = 0;

        for (idx, outcome) in outcomes {
            let p = &self.providers[idx];
            let name = p.name().to_string();
            let typical = p.typical_joules_per_call();
            match outcome {
                Ok(hits) => {
                    if !hits.is_empty() {
                        successes += 1;
                    }
                    reports.push(FederationProviderReport {
                        name: name.clone(),
                        hits_returned: hits.len(),
                        typical_joules: typical,
                        error: String::new(),
                    });
                    // Tag each hit with the provider's name in case the
                    // adapter left `source` empty.
                    let tagged: Vec<SearchHit> = hits
                        .into_iter()
                        .map(|mut h| {
                            if h.source.is_empty() {
                                h.source = name.clone();
                            }
                            h
                        })
                        .collect();
                    per_provider.push((name, tagged));
                }
                Err(e) => {
                    reports.push(FederationProviderReport {
                        name: name.clone(),
                        hits_returned: 0,
                        typical_joules: typical,
                        error: e.to_string(),
                    });
                    per_provider.push((name, Vec::new()));
                }
            }
        }
        (reports, per_provider, successes)
    }

    /// Donor confidence formula, ported to `[0.0, 1.0]`:
    ///
    /// `min(top_score + 0.05 * min(successful_providers, 10), 1.0)`
    ///
    /// The donor used a u16 score with +200 per provider (max +2000 on
    /// a 10_000 scale = +0.20). JouleClaw scales to a 1.0 cap; +0.05 per
    /// provider, cap +0.50, matches the donor's relative shape.
    fn compute_confidence(&self, report: &FuseReport) -> f32 {
        if report.hit_count == 0 {
            return 0.0;
        }
        let diversity = (report.successful_providers.min(10) as f32) * 0.05;
        (report.top_score + diversity).clamp(0.0, 1.0)
    }

    /// End-to-end pipeline: dispatch, fuse, build answer.
    fn federate(&self, query: &str) -> Result<Answer, FederationError> {
        let outcomes = self.dispatch_parallel(query);
        let (reports, per_provider, successes) = self.assemble(outcomes);
        let joules_spent = self.total_typical_joules();

        // All providers failed → refuse the whole tier.
        if successes == 0 {
            return Ok(refused(self.tier_id(), joules_spent));
        }

        let (hits, fuse_report) = self.fuser.fuse(&per_provider, self.max_hits);
        if hits.is_empty() {
            return Ok(refused(self.tier_id(), joules_spent));
        }

        let confidence = self.compute_confidence(&fuse_report);
        let providers_used: Vec<String> =
            self.providers.iter().map(|p| p.name().to_string()).collect();

        let payload = FederationOutput {
            query: query.to_string(),
            providers_used,
            provider_reports: reports,
            hits,
            fuser: self.fuser.name().to_string(),
            joules_spent,
        };
        let bytes = serde_json::to_vec(&payload)?;

        Ok(Answer {
            output: AnswerOutput::Structured(bytes),
            tier_used: self.tier_id(),
            joules_spent,
            confidence,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        })
    }

    /// The `TierId` this federation reports — convenience for tests and
    /// downstream wiring.
    pub fn tier_id(&self) -> TierId {
        TierId::L2(self.model_id)
    }
}

// ─── Answer helpers ──────────────────────────────────────────────────

fn refused(tier: TierId, joules_spent: f64) -> Answer {
    Answer {
        output: AnswerOutput::Refused(RefusalReason::Inapplicable),
        tier_used: tier,
        joules_spent,
        confidence: 0.0,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

// ─── Tier impl ───────────────────────────────────────────────────────

impl Tier for Federation {
    fn id(&self) -> TierId {
        self.tier_id()
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        match &q.input {
            QueryInput::Text(_) => {}
            _ => return None,
        };
        if self.providers.is_empty() {
            return None;
        }
        Some(TierEstimate {
            joules: self.total_typical_joules(),
            latency: DEFAULT_FEDERATION_LATENCY,
            confidence_floor: FEDERATION_CONFIDENCE_FLOOR,
        })
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget_remaining: f64,
    ) -> Result<Answer, AnswerError> {
        let text = match &q.input {
            QueryInput::Text(s) => s.clone(),
            _ => return Ok(refused(self.tier_id(), 0.0)),
        };
        self.federate(&text).map_err(|e| AnswerError::TierFailed {
            tier: self.tier_id(),
            cause: e.to_string(),
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuser::RrfFuser;
    use crate::provider::MockProvider;
    use jouleclaw_cascade::tier::Cascade;
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, QualityFloor, Query, QueryInput,
    };

    fn three_mocks() -> Vec<Box<dyn SearchProvider>> {
        vec![
            Box::new(MockProvider::named("brave").with_joules(50e-6)),
            Box::new(MockProvider::named("wikipedia").with_joules(20e-6)),
            Box::new(MockProvider::named("arxiv").with_joules(30e-6)),
        ]
    }

    fn text_query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.into()),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn tier_id_is_l2() {
        let f = Federation::with_providers(three_mocks());
        assert_eq!(f.id(), TierId::L2(L2ModelId(0)));
    }

    #[test]
    fn tier_id_custom_model_id() {
        let f = Federation::new(
            L2ModelId(7),
            three_mocks(),
            Box::new(LinearFuser::default()),
        );
        assert_eq!(f.id(), TierId::L2(L2ModelId(7)));
    }

    #[test]
    fn estimate_cost_sums_provider_joules() {
        let f = Federation::with_providers(three_mocks());
        let est = f.estimate_cost(&text_query("rust")).expect("applicable");
        // 50 + 20 + 30 = 100 µJ
        assert!((est.joules - 100e-6).abs() < 1e-12);
        assert_eq!(est.confidence_floor, FEDERATION_CONFIDENCE_FLOOR);
        assert_eq!(est.latency, DEFAULT_FEDERATION_LATENCY);
    }

    #[test]
    fn estimate_cost_non_text_is_none() {
        let f = Federation::with_providers(three_mocks());
        let q = Query {
            input: QueryInput::Binary(vec![1, 2, 3]),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(f.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_cost_empty_providers_is_none() {
        let f = Federation::with_providers(vec![]);
        assert!(f.estimate_cost(&text_query("rust")).is_none());
    }

    #[test]
    fn parallel_dispatch_fuses_three_providers() {
        let mut f = Federation::with_providers(three_mocks());
        let a = f.try_answer(&text_query("rust"), 1.0).expect("ok");
        let bytes = match a.output {
            AnswerOutput::Structured(b) => b,
            other => panic!("expected structured, got {other:?}"),
        };
        let payload: FederationOutput =
            serde_json::from_slice(&bytes).expect("deser");
        assert_eq!(payload.providers_used.len(), 3);
        assert!(payload.hits.len() >= 3);
        assert_eq!(payload.fuser, "linear");
        // Three providers each report success → all 3 in provider_reports.
        assert_eq!(payload.provider_reports.len(), 3);
        for r in &payload.provider_reports {
            assert!(r.hits_returned > 0);
            assert!(r.error.is_empty());
        }
        // Confidence: top fused score + 3 * 0.05 diversity = within [0,1].
        assert!(a.confidence > 0.0);
        assert!(a.confidence <= 1.0);
    }

    #[test]
    fn one_failing_provider_does_not_poison_the_rest() {
        let providers: Vec<Box<dyn SearchProvider>> = vec![
            Box::new(
                MockProvider::named("brave")
                    .with_forced_error("rate-limited")
                    .with_joules(50e-6),
            ),
            Box::new(MockProvider::named("wikipedia").with_joules(20e-6)),
            Box::new(MockProvider::named("arxiv").with_joules(30e-6)),
        ];
        let mut f = Federation::with_providers(providers);
        let a = f.try_answer(&text_query("rust"), 1.0).expect("ok");
        let bytes = match a.output {
            AnswerOutput::Structured(b) => b,
            other => panic!("expected structured, got {other:?}"),
        };
        let payload: FederationOutput =
            serde_json::from_slice(&bytes).expect("deser");
        // brave failed; wikipedia + arxiv succeeded → 2 successes recorded.
        let failed = payload
            .provider_reports
            .iter()
            .filter(|r| !r.error.is_empty())
            .count();
        let ok = payload
            .provider_reports
            .iter()
            .filter(|r| r.hits_returned > 0)
            .count();
        assert_eq!(failed, 1);
        assert_eq!(ok, 2);
        // Fused list still has hits.
        assert!(!payload.hits.is_empty());
    }

    #[test]
    fn all_providers_failing_refuses() {
        let providers: Vec<Box<dyn SearchProvider>> = vec![
            Box::new(MockProvider::named("a").with_forced_error("x")),
            Box::new(MockProvider::named("b").with_forced_error("y")),
        ];
        let mut f = Federation::with_providers(providers);
        let a = f.try_answer(&text_query("rust"), 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable),
        ));
        assert_eq!(a.confidence, 0.0);
    }

    #[test]
    fn try_answer_non_text_refuses() {
        let mut f = Federation::with_providers(three_mocks());
        let q = Query {
            input: QueryInput::Binary(vec![1]),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let a = f.try_answer(&q, 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable),
        ));
    }

    #[test]
    fn rrf_fuser_works_end_to_end() {
        let mut f = Federation::new(
            L2ModelId(0),
            three_mocks(),
            Box::new(RrfFuser::default()),
        );
        let a = f.try_answer(&text_query("rust"), 1.0).expect("ok");
        let bytes = match a.output {
            AnswerOutput::Structured(b) => b,
            _ => panic!("expected structured"),
        };
        let payload: FederationOutput =
            serde_json::from_slice(&bytes).expect("deser");
        assert_eq!(payload.fuser, "rrf");
        assert!(!payload.hits.is_empty());
    }

    #[test]
    fn provenance_is_estimator() {
        assert_eq!(Federation::provenance(), Provenance::Estimator);
    }

    #[test]
    fn registers_in_a_cascade() {
        let mut c = Cascade::new();
        c.register(Box::new(Federation::with_providers(three_mocks())));
        let ids = c.tier_ids();
        assert!(matches!(ids.first(), Some(TierId::L2(_))));
    }

    #[test]
    fn diversity_bonus_lifts_confidence() {
        let f = Federation::with_providers(three_mocks());
        let r_three = FuseReport {
            top_score: 0.5,
            successful_providers: 3,
            hit_count: 5,
        };
        let r_one = FuseReport {
            top_score: 0.5,
            successful_providers: 1,
            hit_count: 5,
        };
        let c_three = f.compute_confidence(&r_three);
        let c_one = f.compute_confidence(&r_one);
        // +0.05 per provider, capped at 10 — so 3 should add 0.15.
        assert!(c_three > c_one);
        assert!((c_three - 0.65).abs() < 1e-6);
        assert!((c_one - 0.55).abs() < 1e-6);
    }

    #[test]
    fn confidence_caps_at_one() {
        let f = Federation::with_providers(three_mocks());
        let r = FuseReport {
            top_score: 0.95,
            successful_providers: 20,
            hit_count: 10,
        };
        let c = f.compute_confidence(&r);
        assert!((c - 1.0).abs() < 1e-6);
    }

    #[test]
    fn output_serialises_roundtrip() {
        let out = FederationOutput {
            query: "q".into(),
            providers_used: vec!["a".into(), "b".into()],
            provider_reports: vec![
                FederationProviderReport {
                    name: "a".into(),
                    hits_returned: 2,
                    typical_joules: 50e-6,
                    error: String::new(),
                },
                FederationProviderReport {
                    name: "b".into(),
                    hits_returned: 0,
                    typical_joules: 20e-6,
                    error: "boom".into(),
                },
            ],
            hits: vec![FusedHit {
                url: "u".into(),
                title: "t".into(),
                snippet: "s".into(),
                score: 0.7,
                sources: vec!["a".into()],
            }],
            fuser: "linear".into(),
            joules_spent: 70e-6,
        };
        let bytes = serde_json::to_vec(&out).expect("ser");
        let back: FederationOutput =
            serde_json::from_slice(&bytes).expect("deser");
        assert_eq!(back, out);
    }

    #[test]
    fn with_per_provider_k_pins_request_size() {
        let providers: Vec<Box<dyn SearchProvider>> = vec![Box::new(
            MockProvider::named("brave").with_fixed_hit_count(2),
        )];
        let mut f = Federation::with_providers(providers).with_per_provider_k(2);
        let a = f.try_answer(&text_query("rust"), 1.0).expect("ok");
        let bytes = match a.output {
            AnswerOutput::Structured(b) => b,
            _ => panic!("expected structured"),
        };
        let payload: FederationOutput =
            serde_json::from_slice(&bytes).expect("deser");
        assert_eq!(payload.provider_reports[0].hits_returned, 2);
        assert!(f.provider_count() == 1);
    }

    #[test]
    fn with_max_hits_truncates_output() {
        // Build 5 providers each returning 5 distinct hits — fuser sees
        // 25 candidates; cap output at 3.
        let providers: Vec<Box<dyn SearchProvider>> = (0..5)
            .map(|i| {
                Box::new(MockProvider::named(format!("p{i}")))
                    as Box<dyn SearchProvider>
            })
            .collect();
        let mut f = Federation::with_providers(providers).with_max_hits(3);
        let a = f.try_answer(&text_query("rust"), 1.0).expect("ok");
        let bytes = match a.output {
            AnswerOutput::Structured(b) => b,
            _ => panic!("expected structured"),
        };
        let payload: FederationOutput =
            serde_json::from_slice(&bytes).expect("deser");
        assert!(payload.hits.len() <= 3);
    }

    #[test]
    fn provider_names_listed_in_order() {
        let f = Federation::with_providers(three_mocks());
        assert_eq!(f.provider_names(), vec!["brave", "wikipedia", "arxiv"]);
    }
}
