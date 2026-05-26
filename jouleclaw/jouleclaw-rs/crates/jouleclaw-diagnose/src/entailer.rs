//! Entailment (spec §6.4).
//!
//! [`Entailer`] is the contract the diagnose pillar uses to judge
//! claim ↔ retrieved-item pairs. It returns full
//! [`EntailmentResult`]s (schema-shaped, with calibrated
//! probabilities and joule accounting), so the verification report
//! can carry them in its `entailments_consulted` list.
//!
//! Two impls ship with this module:
//!
//! - [`DebertaEntailer`] — wraps a `jouleclaw_deberta::NliEngine`, which
//!   is the production-grade entailment backend verified
//!   end-to-end in phase 4f. Slow on CPU (~22s per call for
//!   17-token sequences); good enough to demonstrate the
//!   verification chain.
//! - [`FixtureEntailer`] — a programmable mock keyed by `(premise,
//!   hypothesis)` tuple, used in unit tests of the conflict search
//!   so the tests don't need to load DeBERTa.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use uuid::Uuid;

use jouleclaw_schema::{
    AtomicClaim, EntailmentLabel, EntailmentProbabilities, EntailmentResult, RetrievedItem,
};

use jouleclaw_deberta::{NliEngine, NliInference, NliInferenceError};

#[derive(Debug)]
pub enum EntailError {
    Backend(String),
    InputTooLong { actual: usize, max: usize },
}

impl std::fmt::Display for EntailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(s) => write!(f, "entailer backend: {s}"),
            Self::InputTooLong { actual, max } => write!(f, "input {actual} > max {max}"),
        }
    }
}

impl std::error::Error for EntailError {}

impl From<NliInferenceError> for EntailError {
    fn from(e: NliInferenceError) -> Self {
        match e {
            NliInferenceError::InputTooLong { actual, max } => {
                Self::InputTooLong { actual, max }
            }
            other => Self::Backend(other.to_string()),
        }
    }
}

/// Run entailment against (claim, item) pairs and return full
/// [`EntailmentResult`]s.
pub trait Entailer: Send + Sync {
    /// Run NLI for a single (premise text, hypothesis text) pair.
    /// Higher-level helpers in this module construct premises from
    /// retrieved items and hypotheses from atomic claims.
    fn entail_raw(&self, premise: &str, hypothesis: &str) -> Result<RawEntailment, EntailError>;

    /// Stable model id surfaced in `EntailmentResult.model_id`.
    fn model_id(&self) -> &str;

    /// Wrap a raw entailment for (claim, item) in a schema-shaped
    /// [`EntailmentResult`] with attribution. Implementations
    /// usually don't override this; the default builds the result
    /// from `entail_raw`.
    fn entail_claim_against(
        &self,
        claim: &AtomicClaim,
        item: &RetrievedItem,
    ) -> Result<EntailmentResult, EntailError> {
        let premise = item
            .content
            .text
            .clone()
            .unwrap_or_else(|| format!("[{} non-text source]", item.source_id));
        let hypothesis = claim.text.clone();
        let raw = self.entail_raw(&premise, &hypothesis)?;
        Ok(EntailmentResult {
            schema_version: "2.0".into(),
            result_id: Uuid::new_v4(),
            claim_id: claim.claim_id,
            premise_item_ids: vec![item.item_id],
            label: raw.label,
            label_probabilities: raw.probabilities,
            running_e_value: None,
            model_id: self.model_id().to_string(),
            joules_spent: raw.joules_spent,
            metadata: Default::default(),
        })
    }
}

/// Output shape of `entail_raw` — the categorical label, full
/// probability distribution, and measured joules. Anything else the
/// trait needs (`claim_id`, `premise_item_ids`, etc.) is added by
/// the default `entail_claim_against`.
#[derive(Debug, Clone)]
pub struct RawEntailment {
    pub label: EntailmentLabel,
    pub probabilities: EntailmentProbabilities,
    pub joules_spent: f64,
}

// ──────────────────────────────────────────────────────────────────
// DeBERTa adapter
// ──────────────────────────────────────────────────────────────────

/// Bridges the diagnose pillar to `jouleclaw_deberta::NliEngine`.
pub struct DebertaEntailer {
    engine: NliEngine,
    /// Static per-call joule estimate. For the 17-token NLI pair
    /// the actual measured wall-clock energy is much higher (~22s @
    /// laptop TDP), but we surface a per-token cost model so the
    /// runtime energy ledger has something honest to add up.
    /// Update once we add real per-step measurement.
    joules_per_call: f64,
}

impl DebertaEntailer {
    pub fn new(engine: NliEngine) -> Self {
        // Rough estimate: 22s wall × ~25 W laptop CPU draw ≈ 550 J.
        // This is illustrative — real deployments replace this with
        // a measured value from the energy accounting infrastructure.
        Self {
            engine,
            joules_per_call: 550.0,
        }
    }

    pub fn with_joules_per_call(mut self, j: f64) -> Self {
        self.joules_per_call = j;
        self
    }
}

impl Entailer for DebertaEntailer {
    fn entail_raw(&self, premise: &str, hypothesis: &str) -> Result<RawEntailment, EntailError> {
        let pred = self.engine.predict(premise, hypothesis)?;
        let label: EntailmentLabel = pred.label.into();
        Ok(RawEntailment {
            label,
            probabilities: pred.probabilities,
            joules_spent: if pred.joules_spent > 0.0 {
                pred.joules_spent
            } else {
                self.joules_per_call
            },
        })
    }

    fn model_id(&self) -> &str {
        self.engine.model_id()
    }
}

// ──────────────────────────────────────────────────────────────────
// Fixture adapter (for tests)
// ──────────────────────────────────────────────────────────────────

/// Programmable entailer keyed by `(premise, hypothesis)`. Use in
/// tests where loading DeBERTa is not desirable.
pub struct FixtureEntailer {
    map: Mutex<HashMap<(String, String), RawEntailment>>,
    default: Mutex<RawEntailment>,
    model_id: String,
}

impl FixtureEntailer {
    pub fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
            default: Mutex::new(RawEntailment {
                label: EntailmentLabel::Neutral,
                probabilities: EntailmentProbabilities {
                    entails: 0.34,
                    neutral: 0.33,
                    contradicts: 0.33,
                },
                joules_spent: 1.0,
            }),
            model_id: "fixture-entailer".into(),
        }
    }

    pub fn set(&self, premise: &str, hypothesis: &str, raw: RawEntailment) {
        self.map
            .lock()
            .unwrap()
            .insert((premise.into(), hypothesis.into()), raw);
    }

    pub fn set_default(&self, raw: RawEntailment) {
        *self.default.lock().unwrap() = raw;
    }

    /// Convenience: program an entailment label by name. `_e` =
    /// entails, `_n` = neutral, `_c` = contradicts.
    pub fn set_entails(&self, premise: &str, hypothesis: &str) {
        self.set(
            premise,
            hypothesis,
            RawEntailment {
                label: EntailmentLabel::Entails,
                probabilities: EntailmentProbabilities {
                    entails: 0.95,
                    neutral: 0.04,
                    contradicts: 0.01,
                },
                joules_spent: 1.0,
            },
        );
    }

    pub fn set_neutral(&self, premise: &str, hypothesis: &str) {
        self.set(
            premise,
            hypothesis,
            RawEntailment {
                label: EntailmentLabel::Neutral,
                probabilities: EntailmentProbabilities {
                    entails: 0.10,
                    neutral: 0.80,
                    contradicts: 0.10,
                },
                joules_spent: 1.0,
            },
        );
    }

    pub fn set_contradicts(&self, premise: &str, hypothesis: &str) {
        self.set(
            premise,
            hypothesis,
            RawEntailment {
                label: EntailmentLabel::Contradicts,
                probabilities: EntailmentProbabilities {
                    entails: 0.01,
                    neutral: 0.04,
                    contradicts: 0.95,
                },
                joules_spent: 1.0,
            },
        );
    }
}

impl Default for FixtureEntailer {
    fn default() -> Self {
        Self::new()
    }
}

impl Entailer for FixtureEntailer {
    fn entail_raw(&self, premise: &str, hypothesis: &str) -> Result<RawEntailment, EntailError> {
        let guard = self.map.lock().unwrap();
        if let Some(r) = guard.get(&(premise.to_string(), hypothesis.to_string())) {
            return Ok(r.clone());
        }
        Ok(self.default.lock().unwrap().clone())
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// Run [`Entailer::entail_claim_against`] for the cross product of
/// `claims` × `items` in parallel via `std::thread::scope`. Each
/// (claim, item) pair runs on its own OS thread so Accelerate's
/// sgemm calls can interleave across cores via GCD's shared
/// pool. Sequential ordering of results is preserved (matches the
/// row-major flatten of the cross product).
///
/// On macOS Accelerate's threading is libdispatch-based — submitting
/// N concurrent sgemm calls doesn't over-subscribe; libdispatch
/// queues work onto a shared global pool and load-balances. For our
/// 14-pair Wikidata+Wikipedia workload this drops the verify stage
/// from ~2.5 s sequential to roughly the cost of the longest single
/// call (the long Wikipedia summary) plus dispatch.
///
/// **Why not a single padded batched forward?** `NliEngine::predict_batch`
/// exists, and on a homogeneous batch it wins ~1.7× over this path.
/// But on our actual workload (short Wikidata claims mixed with long
/// Wikipedia summaries) padding every pair to L_max makes the batched
/// path ~16% *slower* than parallel-per-pair. See
/// `jouleclaw_deberta::engine::tests::bench_heterogeneous_batch`. If a
/// future caller has a known-homogeneous batch (e.g., classify N
/// similar-length passages), call `predict_batch` directly.
pub fn entail_batch<E: Entailer + ?Sized>(
    entailer: &E,
    claims: &[AtomicClaim],
    items: &[RetrievedItem],
) -> Result<(Vec<EntailmentResult>, f64), EntailError> {
    let start = Instant::now();

    // No work → fast path.
    if claims.is_empty() || items.is_empty() {
        let _ = start.elapsed();
        return Ok((Vec::new(), 0.0));
    }

    // Flatten the cross product into an indexed list so the
    // post-join sort puts results back in row-major order.
    let pairs: Vec<(usize, &AtomicClaim, &RetrievedItem)> = claims
        .iter()
        .enumerate()
        .flat_map(|(ci, c)| {
            items
                .iter()
                .enumerate()
                .map(move |(ii, it)| (ci * items.len() + ii, c, it))
        })
        .collect();
    let n = pairs.len();

    // Fast path: single pair → no thread scope overhead.
    if n == 1 {
        let (_, claim, item) = pairs[0];
        let r = entailer.entail_claim_against(claim, item)?;
        let joules = r.joules_spent;
        let _ = start.elapsed();
        return Ok((vec![r], joules));
    }

    // Parallel path: one OS thread per pair, joined in order.
    let mut indexed_results: Vec<(usize, Result<EntailmentResult, EntailError>)> =
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(n);
            for (idx, claim, item) in pairs {
                let h =
                    scope.spawn(move || (idx, entailer.entail_claim_against(claim, item)));
                handles.push(h);
            }
            handles
                .into_iter()
                .map(|h| h.join().expect("entailment thread panicked"))
                .collect()
        });

    // Preserve cross-product order so callers can match results
    // against the (claim, item) indexing they passed in.
    indexed_results.sort_by_key(|(idx, _)| *idx);

    // First error short-circuits — surfaces a single failure
    // cleanly without losing the rest's work for diagnostics.
    let mut results = Vec::with_capacity(n);
    let mut joules_total = 0.0;
    for (_, r) in indexed_results {
        let r = r?;
        joules_total += r.joules_spent;
        results.push(r);
    }
    let _ = start.elapsed();
    Ok((results, joules_total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use jouleclaw_schema::{
        Attribution, ClaimStakes, Content, FreshnessClass, GranularityClass, KnowledgeAxes,
        Modality, RetrievalContext, RetrievalMethod, RetrievedItem, ScopeClass, ScoreType,
        SourceType, TemporalStabilityClass, Temporal,
    };

    fn axes() -> KnowledgeAxes {
        KnowledgeAxes {
            schema_version: "5.0".into(),
            valid_time_start: None,
            valid_time_end: None,
            transaction_time: None,
            reference_time: Utc::now(),
            temporal_stability: TemporalStabilityClass::Slow,
            granularity: GranularityClass::Coarse,
            granularity_notes: None,
            scope: ScopeClass::Particular,
            scope_domain: None,
            certainty: 1.0,
            certainty_basis: "test".into(),
            source_uri: None,
            source_authority_tier: 1,
            extraction_method: None,
            citation_chain: vec![],
            metadata: Default::default(),
        }
    }

    fn item(text: &str) -> RetrievedItem {
        RetrievedItem {
            schema_version: "2.0".into(),
            item_id: Uuid::new_v4(),
            source_id: "wikidata:Q90".into(),
            source_url: None,
            source_type: SourceType::StructuredKb,
            content: Content {
                modality: Modality::Text,
                text: Some(text.into()),
                media_ref: None,
                structured: None,
                excerpt_span: None,
            },
            retrieval_context: RetrievalContext {
                retriever_id: "wikidata".into(),
                matched_against: "test".into(),
                sub_id: "q0".into(),
                raw_score: 1.0,
                score_type: ScoreType::Exact,
                normalized_score: Some(1.0),
                rank_in_store: 0,
                retrieval_method: RetrievalMethod::Sparql,
                hop_quality: None,
                hop_path: None,
                rap_step: "primary".into(),
                rap_attempts: 1,
            },
            temporal: Temporal {
                content_timestamp: None,
                retrieval_timestamp: Utc::now(),
                last_modified: None,
                freshness_class: FreshnessClass::Timeless,
            },
            attribution: Attribution::default(),
            knowledge_axes: axes(),
            metadata: Default::default(),
        }
    }

    fn claim(text: &str) -> AtomicClaim {
        AtomicClaim {
            schema_version: "2.0".into(),
            claim_id: Uuid::new_v4(),
            text: text.into(),
            segment_id: "s0".into(),
            stakes: ClaimStakes::Medium,
            knowledge_axes: axes(),
            atomization_notes: None,
            metadata: Default::default(),
        }
    }

    #[test]
    fn fixture_returns_default_for_unknown_pair() {
        let e = FixtureEntailer::new();
        let r = e.entail_raw("p", "h").unwrap();
        assert!(matches!(r.label, EntailmentLabel::Neutral));
    }

    #[test]
    fn fixture_set_overrides_default() {
        let e = FixtureEntailer::new();
        e.set_entails("Paris is the capital of France.", "France's capital is Paris.");
        let r = e
            .entail_raw("Paris is the capital of France.", "France's capital is Paris.")
            .unwrap();
        assert!(matches!(r.label, EntailmentLabel::Entails));
        assert!(r.probabilities.entails > 0.9);
    }

    #[test]
    fn entail_claim_against_attaches_ids_and_model() {
        let e = FixtureEntailer::new();
        let it = item("Paris is the capital of France.");
        let cl = claim("Paris is the capital of France.");
        e.set_entails(
            "Paris is the capital of France.",
            "Paris is the capital of France.",
        );
        let r = e.entail_claim_against(&cl, &it).unwrap();
        assert_eq!(r.claim_id, cl.claim_id);
        assert_eq!(r.premise_item_ids, vec![it.item_id]);
        assert_eq!(r.model_id, "fixture-entailer");
        assert!(matches!(r.label, EntailmentLabel::Entails));
    }

    #[test]
    fn entail_batch_runs_cross_product() {
        let e = FixtureEntailer::new();
        let it1 = item("Paris is in France.");
        let it2 = item("France is in Europe.");
        let cl1 = claim("Paris is in France.");
        let cl2 = claim("France is in Europe.");
        e.set_entails("Paris is in France.", "Paris is in France.");
        e.set_entails("France is in Europe.", "France is in Europe.");

        let (results, joules) =
            entail_batch(&e, &[cl1, cl2], &[it1, it2]).unwrap();
        assert_eq!(results.len(), 4);
        assert!(joules >= 4.0); // 4 calls * 1.0 per call default
    }
}
