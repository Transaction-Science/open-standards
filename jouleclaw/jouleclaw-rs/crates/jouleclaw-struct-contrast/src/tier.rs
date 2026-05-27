//! L1.375 — structural-contrast tier with graph-enriched input.
//!
//! Runs after L1.25 GraphRag. Consumes the graph-resolved entity list,
//! looks each name up in a [`KnowledgeStore`], computes pairwise
//! contrast, and emits a per-dimension breakdown — every dimension
//! tagged `Align` / `Oppose` / `Partial` / `Unknown`.
//!
//! ## Mapping to the `Tier` trait
//!
//! The donor returned a `LayerOutcome` enum (`Resolved` / `Partial` /
//! `Skipped`). JouleClaw's `Tier` returns an `Answer` whose `output` is:
//!
//! - `AnswerOutput::Structured(json_bytes)` — when the formula resolves
//!   with coverage ≥ 70% (the donor's `confidence ≥ 7000/10000` gate).
//!   The structured payload carries the per-dimension verdict array
//!   plus a human-readable text summary.
//! - `AnswerOutput::Refused(LowConfidence(c))` — Partial / Skipped
//!   cases, with the formula's confidence so the cascade can continue.
//! - `AnswerOutput::Refused(Inapplicable)` — when the input isn't a
//!   well-formed [`StructContrastInput`] envelope.

use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, Query, QueryInput,
    RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;
use jouleclaw_energy::Provenance;
use jouleclaw_formula::knowledge::{
    Concept, ContrastMap, ContrastRelation, KnowledgeStore,
};
use serde::{Deserialize, Serialize};

use crate::context::StructContrastInput;

// ─── Cost model ──────────────────────────────────────────────────

/// Donor's joule envelope: ~200 µJ per query. The second formula pass
/// is slightly more work than L0.25 (per-dimension breakdown emitted
/// verbatim) but well within the same coarse class.
pub const CONTRAST_JOULES: f64 = 200e-6;
/// Wall-clock latency estimate.
pub const CONTRAST_LATENCY: Duration = Duration::from_micros(10);
/// Confidence floor advertised to the runtime.
pub const CONTRAST_CONFIDENCE_FLOOR: f32 = 0.75;

/// Donor confidence gate (q14.0) above which the tier claims full
/// resolution. `7_000 / 10_000 = 0.7`. We still advertise a
/// `CONTRAST_CONFIDENCE_FLOOR` of `0.75` because the gate is hit only
/// when coverage is well above 70 %.
const CONTRAST_RESOLUTION_CONFIDENCE: u16 = 7_000;

// ─── Per-dimension verdict ───────────────────────────────────────

/// Coarse verdict for a single dimension in a pairwise contrast.
///
/// Mirrors the donor's "Known/Partial/Unknown" classification (axiom 4
/// in the donor) but renames the two `Known` cases — `Align` and
/// `Oppose` — to surface the *direction* of agreement explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContrastVerdict {
    /// Both entities point the same way on this dimension.
    Align,
    /// Both entities point opposite ways on this dimension.
    Oppose,
    /// Only one entity has a measurement on this dimension.
    Partial,
    /// Neither entity has a measurement on this dimension.
    Unknown,
}

impl ContrastVerdict {
    /// Project a [`ContrastRelation`] from `jouleclaw-formula` into the
    /// L1.375 verdict alphabet. The two enums are intentionally
    /// shape-identical so the bridge is a four-arm match.
    pub fn from_relation(r: ContrastRelation) -> Self {
        match r {
            ContrastRelation::Align => Self::Align,
            ContrastRelation::Oppose => Self::Oppose,
            ContrastRelation::Partial => Self::Partial,
            ContrastRelation::Unknown => Self::Unknown,
        }
    }

    /// Wire-stable string tag. Used in the structured JSON payload.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Align => "align",
            Self::Oppose => "oppose",
            Self::Partial => "partial",
            Self::Unknown => "unknown",
        }
    }
}

// ─── Public output shapes ────────────────────────────────────────

/// One dimension's contribution within a pairwise contrast.
///
/// Serialised under the `dim` key in the structured payload to match
/// the user-facing schema:
/// `{"dim": "heat", "verdict": "oppose", "score": -1.0}`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DimensionVerdict {
    /// Dimension name, falling back to `dim_<index>` when the store
    /// does not expose names.
    pub dim: String,
    /// Coarse alignment verdict.
    pub verdict: ContrastVerdict,
    /// Signed score from the formula (positive = align, negative =
    /// oppose, zero = partial/unknown).
    pub score: f32,
}

/// One pairwise contrast surfaced in the structured payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairContrast {
    /// Display name of the first entity.
    pub entity_a: String,
    /// Display name of the second entity.
    pub entity_b: String,
    /// Aggregate similarity in `[0.0, 1.0]`.
    pub similarity: f32,
    /// Aggregate coverage in `[0.0, 1.0]`.
    pub coverage: f32,
    /// Per-dimension verdicts.
    pub contrast: Vec<DimensionVerdict>,
}

/// Sidecar payload emitted on every dispatch. Lets observability and
/// downstream tiers see the per-dimension verdicts even when the tier
/// itself refuses to claim the answer.
#[derive(Debug, Default, Clone)]
pub struct StructContrastSidecar {
    /// Per-pair contrasts from the most recent dispatch.
    pub pairs: Vec<PairContrast>,
    /// Human-readable analysis text.
    pub explanation: String,
    /// Confidence the tier assigned.
    pub confidence: f32,
}

/// Receives the L1.375 side-channel on every dispatch.
pub trait StructContrastSidecarSink: Send {
    /// Called after every dispatch with the per-pair analysis.
    fn observe(&mut self, sidecar: StructContrastSidecar);
}

impl<F> StructContrastSidecarSink for F
where
    F: FnMut(StructContrastSidecar) + Send,
{
    fn observe(&mut self, sidecar: StructContrastSidecar) {
        (self)(sidecar);
    }
}

// ─── Errors ──────────────────────────────────────────────────────

/// Errors specific to the L1.375 tier.
#[derive(Debug, thiserror::Error)]
pub enum StructContrastError {
    /// Structured payload failed to deserialise as a
    /// [`StructContrastInput`] envelope.
    #[error("malformed struct-contrast envelope: {0}")]
    Decode(#[from] serde_json::Error),
}

// ─── The tier ────────────────────────────────────────────────────

/// L1.375 structural-contrast tier. Generic over the same
/// [`KnowledgeStore`] trait as L0.25 so a single store can drive both
/// passes.
pub struct StructContrastTier<K: KnowledgeStore> {
    knowledge: K,
    sidecar: Option<Box<dyn StructContrastSidecarSink>>,
}

impl<K: KnowledgeStore> StructContrastTier<K> {
    /// Build a new L1.375 tier over a knowledge store.
    pub fn new(knowledge: K) -> Self {
        Self { knowledge, sidecar: None }
    }

    /// Install a sidecar sink for the per-pair analysis. Useful for
    /// observability and when downstream tiers want the Partial output
    /// even though the tier itself refused.
    pub fn with_sidecar(
        mut self,
        sink: Box<dyn StructContrastSidecarSink>,
    ) -> Self {
        self.sidecar = Some(sink);
        self
    }

    /// Borrow the underlying knowledge store.
    pub fn knowledge(&self) -> &K {
        &self.knowledge
    }

    /// Provenance for any energy spend recorded by this tier — same as
    /// L0.25, this is a model-based estimator.
    pub const fn provenance() -> Provenance {
        Provenance::Estimator
    }

    /// Try to extract a [`StructContrastInput`] from a [`Query`].
    /// Returns `None` for any other input shape.
    fn decode_input(q: &Query) -> Option<Result<StructContrastInput, serde_json::Error>> {
        match &q.input {
            QueryInput::Structured(bytes) => {
                Some(serde_json::from_slice::<StructContrastInput>(bytes))
            }
            _ => None,
        }
    }

    /// Whether the query input is a well-formed envelope with the
    /// canonical [`crate::STRUCT_CONTRAST_KIND`] tag. Used by
    /// `estimate_cost` to decide applicability without committing to a
    /// full dispatch.
    fn is_applicable(&self, q: &Query) -> bool {
        if self.knowledge.is_empty() {
            return false;
        }
        match Self::decode_input(q) {
            Some(Ok(env)) => env.kind_matches(),
            _ => false,
        }
    }

    fn resolve(
        &mut self,
        envelope: StructContrastInput,
    ) -> Result<Answer, StructContrastError> {
        // Resolve entity names against the knowledge store.
        let dim_names = self.knowledge.dimension_names();
        let mut matched: Vec<Concept> = Vec::with_capacity(envelope.entities.len());
        for name in &envelope.entities {
            let hits = self.knowledge.search_by_name(name, 1);
            if let Some(concept) = hits.into_iter().next() {
                if !matched.iter().any(|c| c.id == concept.id) {
                    matched.push(concept);
                }
            }
        }

        if matched.len() < 2 {
            self.emit_sidecar(StructContrastSidecar {
                pairs: Vec::new(),
                explanation: format!(
                    "fewer than 2 entities resolved in knowledge store for: \"{}\"",
                    envelope.query
                ),
                confidence: 0.0,
            });
            return Ok(refused_inapplicable());
        }

        // Compute pairwise contrasts.
        let mut pairs: Vec<PairContrast> = Vec::new();
        let mut text_lines: Vec<String> = Vec::new();
        let mut total_coverage: u64 = 0;

        for i in 0..matched.len() {
            for j in (i + 1)..matched.len() {
                let ca = &matched[i];
                let cb = &matched[j];
                let Some(map) = self.knowledge.contrast(&ca.id, &cb.id) else {
                    continue;
                };
                total_coverage += map.coverage.0 as u64;
                let verdicts = decompose_dimensions(&map, &dim_names);
                text_lines.push(format!(
                    "{} <-> {}: similarity={}/10000, coverage={}/10000",
                    ca.name, cb.name, map.similarity.0, map.coverage.0
                ));
                pairs.push(PairContrast {
                    entity_a: ca.name.clone(),
                    entity_b: cb.name.clone(),
                    similarity: map.similarity.as_unit(),
                    coverage: map.coverage.as_unit(),
                    contrast: verdicts,
                });
            }
        }

        if pairs.is_empty() {
            self.emit_sidecar(StructContrastSidecar {
                pairs: Vec::new(),
                explanation: format!(
                    "no pairwise contrasts available for: \"{}\"",
                    envelope.query
                ),
                confidence: 0.0,
            });
            return Ok(refused_inapplicable());
        }

        let pair_count = pairs.len() as u64;
        let avg_coverage_q14 = (total_coverage / pair_count) as u16;
        let confidence_q14 = avg_coverage_q14.min(9_000);
        let confidence = q14_to_unit(confidence_q14);

        let explanation = format!(
            "Structural contrast for: \"{}\"\n\
             Matched {} concepts, {} pairwise contrasts:\n\n{}",
            envelope.query,
            matched.len(),
            pair_count,
            text_lines.join("\n---\n"),
        );

        let sidecar = StructContrastSidecar {
            pairs: pairs.clone(),
            explanation: explanation.clone(),
            confidence,
        };
        self.emit_sidecar(sidecar);

        if confidence_q14 >= CONTRAST_RESOLUTION_CONFIDENCE {
            // Structured payload — JSON-encoded.
            let payload = serde_json::json!({
                "kind": "jouleclaw.struct_contrast.result/v1",
                "query": envelope.query,
                "confidence": confidence,
                "text": explanation,
                "contrast": pairs,
            });
            let bytes = serde_json::to_vec(&payload)?;
            Ok(resolved_structured(bytes, confidence))
        } else {
            Ok(refused_low_confidence(confidence))
        }
    }

    fn emit_sidecar(&mut self, sidecar: StructContrastSidecar) {
        if let Some(sink) = self.sidecar.as_mut() {
            sink.observe(sidecar);
        }
    }
}

// ─── Per-dimension decomposition ─────────────────────────────────

/// Project a [`ContrastMap`] from `jouleclaw-formula` into the L1.375
/// per-dimension verdict array. Exposed publicly so callers that
/// already have a `ContrastMap` (e.g. from a federated source) can
/// decompose it without going through the full tier.
pub fn decompose_dimensions(
    map: &ContrastMap,
    dim_names: &[String],
) -> Vec<DimensionVerdict> {
    map.dimensions
        .iter()
        .map(|d| {
            let name = dim_names
                .get(d.dimension as usize)
                .cloned()
                .unwrap_or_else(|| format!("dim_{}", d.dimension));
            DimensionVerdict {
                dim: name,
                verdict: ContrastVerdict::from_relation(d.relation),
                score: d.contribution,
            }
        })
        .collect()
}

// ─── Answer-shape helpers ────────────────────────────────────────

fn resolved_structured(bytes: Vec<u8>, confidence: f32) -> Answer {
    Answer {
        output: AnswerOutput::Structured(bytes),
        tier_used: TierId::L1_375StructContrast,
        joules_spent: CONTRAST_JOULES,
        confidence,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

fn refused_inapplicable() -> Answer {
    Answer {
        output: AnswerOutput::Refused(RefusalReason::Inapplicable),
        tier_used: TierId::L1_375StructContrast,
        joules_spent: CONTRAST_JOULES,
        confidence: 0.0,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

fn refused_low_confidence(confidence: f32) -> Answer {
    Answer {
        output: AnswerOutput::Refused(RefusalReason::low_confidence(confidence)),
        tier_used: TierId::L1_375StructContrast,
        joules_spent: CONTRAST_JOULES,
        confidence,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

fn q14_to_unit(q: u16) -> f32 {
    (q as f32 / 10_000.0).clamp(0.0, 1.0)
}

// ─── Tier-trait impl ─────────────────────────────────────────────

impl<K: KnowledgeStore + Send> Tier for StructContrastTier<K> {
    fn id(&self) -> TierId {
        TierId::L1_375StructContrast
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        if !self.is_applicable(q) {
            return None;
        }
        Some(TierEstimate {
            joules: CONTRAST_JOULES,
            latency: CONTRAST_LATENCY,
            confidence_floor: CONTRAST_CONFIDENCE_FLOOR,
        })
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget_remaining: f64,
    ) -> Result<Answer, AnswerError> {
        let envelope = match Self::decode_input(q) {
            Some(Ok(env)) if env.kind_matches() => env,
            Some(Ok(_)) => return Ok(refused_inapplicable()),
            Some(Err(e)) => {
                return Err(AnswerError::TierFailed {
                    tier: TierId::L1_375StructContrast,
                    cause: format!("envelope decode: {e}"),
                });
            }
            None => return Ok(refused_inapplicable()),
        };
        self.resolve(envelope).map_err(|e| AnswerError::TierFailed {
            tier: TierId::L1_375StructContrast,
            cause: e.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::tier::Cascade;
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, QualityFloor, Query, QueryInput,
    };
    use jouleclaw_formula::knowledge::{Concept, InMemoryKnowledgeStore};

    /// Fully-covered fixture — every dimension measured on every concept.
    /// Coverage is 100 % so the contrast tier hits its `>= 7_000` gate
    /// and emits a `Structured` payload. Used by the resolves-structured
    /// path tests.
    fn store() -> InMemoryKnowledgeStore {
        let mut k = InMemoryKnowledgeStore::new()
            .with_dimension_names(["heat", "wet"]);
        k.insert(Concept {
            id: "urn:fire".into(),
            name: "fire".into(),
            traits: vec![1.0, -1.0],
        });
        k.insert(Concept {
            id: "urn:water".into(),
            name: "water".into(),
            traits: vec![-1.0, 1.0],
        });
        k.insert(Concept {
            id: "urn:steam".into(),
            name: "steam".into(),
            traits: vec![1.0, 1.0],
        });
        k
    }

    /// Mixed-coverage fixture — one dimension is unknown on both sides
    /// (`alive=0` for everything), so per-dimension verdicts include an
    /// `unknown` entry. Used to exercise the verdict-classification
    /// path; coverage drops below the structured gate so callers expect
    /// `Refused(LowConfidence(_))`.
    fn mixed_store() -> InMemoryKnowledgeStore {
        let mut k = InMemoryKnowledgeStore::new()
            .with_dimension_names(["heat", "wet", "alive"]);
        k.insert(Concept {
            id: "urn:fire".into(),
            name: "fire".into(),
            traits: vec![1.0, -1.0, 0.0],
        });
        k.insert(Concept {
            id: "urn:water".into(),
            name: "water".into(),
            traits: vec![-1.0, 1.0, 0.0],
        });
        k
    }

    fn envelope_query(query: &str, ents: &[&str]) -> Query {
        let env = StructContrastInput::new(
            query,
            ents.iter().copied().map(str::to_string),
        );
        let bytes = serde_json::to_vec(&env).expect("serialise envelope");
        Query {
            input: QueryInput::Structured(bytes),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn text_query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.into()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn tier_id_is_l1_375() {
        let t = StructContrastTier::new(store());
        assert_eq!(t.id(), TierId::L1_375StructContrast);
    }

    #[test]
    fn estimate_cost_for_envelope_text_is_none() {
        // Plain text queries skip this tier — they belong to L0.25.
        let t = StructContrastTier::new(store());
        assert!(t.estimate_cost(&text_query("compare fire and water")).is_none());
    }

    #[test]
    fn estimate_cost_for_well_formed_envelope() {
        let t = StructContrastTier::new(store());
        let q = envelope_query("compare fire and water", &["fire", "water"]);
        let est = t.estimate_cost(&q).expect("envelope query is applicable");
        assert!((est.joules - CONTRAST_JOULES).abs() < 1e-12);
        assert_eq!(est.confidence_floor, CONTRAST_CONFIDENCE_FLOOR);
        assert_eq!(est.latency, CONTRAST_LATENCY);
    }

    #[test]
    fn estimate_cost_empty_store_is_none() {
        let t = StructContrastTier::new(InMemoryKnowledgeStore::new());
        let q = envelope_query("q", &["a", "b"]);
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn wrong_kind_estimate_is_none() {
        let t = StructContrastTier::new(store());
        let payload = serde_json::json!({
            "kind": "something.else/v1",
            "query": "q",
            "entities": ["fire", "water"],
        });
        let bytes = serde_json::to_vec(&payload).expect("ser");
        let q = Query {
            input: QueryInput::Structured(bytes),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn binary_input_estimate_is_none() {
        let t = StructContrastTier::new(store());
        let q = Query {
            input: QueryInput::Binary(vec![1, 2, 3]),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn two_entity_envelope_resolves_structured() {
        let mut t = StructContrastTier::new(store());
        let q = envelope_query("compare fire and water", &["fire", "water"]);
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert_eq!(a.tier_used, TierId::L1_375StructContrast);
        match a.output {
            AnswerOutput::Structured(bytes) => {
                let v: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("json");
                assert_eq!(v["kind"], "jouleclaw.struct_contrast.result/v1");
                let arr = v["contrast"].as_array().expect("contrast array");
                assert_eq!(arr.len(), 1, "1 pair for 2 entities");
                let dims = arr[0]["contrast"].as_array().expect("dims");
                assert!(!dims.is_empty(), "per-dimension verdicts present");
            }
            other => panic!("expected Structured, got {other:?}"),
        }
    }

    #[test]
    fn three_entity_envelope_emits_three_pairs() {
        let mut t = StructContrastTier::new(store());
        let q = envelope_query(
            "compare fire water steam",
            &["fire", "water", "steam"],
        );
        let a = t.try_answer(&q, 1.0).expect("ok");
        match a.output {
            AnswerOutput::Structured(bytes) => {
                let v: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("json");
                let arr = v["contrast"].as_array().expect("contrast array");
                // C(3,2) = 3 pairs.
                assert_eq!(arr.len(), 3);
            }
            AnswerOutput::Refused(reason) => {
                panic!("expected Structured, got Refused({reason:?})");
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn single_entity_refuses_inapplicable() {
        let mut t = StructContrastTier::new(store());
        let q = envelope_query("fire", &["fire"]);
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn unknown_entity_names_dropped_then_refuses() {
        let mut t = StructContrastTier::new(store());
        let q = envelope_query("q", &["zzz_unknown_a", "zzz_unknown_b"]);
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn dim_verdicts_classify_fire_vs_water() {
        // mixed_store has a third dimension (`alive`) with no measurements
        // → coverage drops below the 70 % structured-resolution gate, so
        // the tier `Refused(LowConfidence)`. The sidecar still carries
        // the per-dimension verdicts — that's exactly what downstream
        // tiers want to consume after a Partial outcome.
        use std::sync::{Arc, Mutex};
        let captured = Arc::new(Mutex::new(Vec::<StructContrastSidecar>::new()));
        let c2 = Arc::clone(&captured);
        struct Sink(Arc<Mutex<Vec<StructContrastSidecar>>>);
        impl StructContrastSidecarSink for Sink {
            fn observe(&mut self, s: StructContrastSidecar) {
                if let Ok(mut guard) = self.0.lock() {
                    guard.push(s);
                }
            }
        }
        let mut t = StructContrastTier::new(mixed_store())
            .with_sidecar(Box::new(Sink(c2)));
        let q = envelope_query("compare fire and water", &["fire", "water"]);
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(
            matches!(
                a.output,
                AnswerOutput::Refused(RefusalReason::LowConfidence(_))
            ),
            "mixed-coverage fixture should refuse with low confidence, got {:?}",
            a.output
        );

        let guard = captured.lock().expect("lock");
        let sidecar = guard.first().expect("sidecar observed");
        let dims = &sidecar.pairs[0].contrast;
        // heat: fire=+1, water=-1 → oppose. wet: fire=-1, water=+1 → oppose.
        // alive: fire=0, water=0 → unknown.
        let mut has_oppose = false;
        let mut has_unknown = false;
        for d in dims {
            match d.verdict {
                ContrastVerdict::Oppose => has_oppose = true,
                ContrastVerdict::Unknown => has_unknown = true,
                _ => {}
            }
        }
        assert!(has_oppose, "expected at least one oppose dimension");
        assert!(has_unknown, "expected the unknown dimension");
    }

    #[test]
    fn sidecar_observes_dispatch() {
        use std::sync::{Arc, Mutex};
        let captured = Arc::new(Mutex::new(Vec::<StructContrastSidecar>::new()));
        let c2 = Arc::clone(&captured);
        struct Sink(Arc<Mutex<Vec<StructContrastSidecar>>>);
        impl StructContrastSidecarSink for Sink {
            fn observe(&mut self, s: StructContrastSidecar) {
                if let Ok(mut guard) = self.0.lock() {
                    guard.push(s);
                }
            }
        }
        let mut t = StructContrastTier::new(store())
            .with_sidecar(Box::new(Sink(c2)));
        let q = envelope_query("compare fire and water", &["fire", "water"]);
        let _ = t.try_answer(&q, 1.0).expect("ok");
        let guard = captured.lock().expect("lock");
        assert!(!guard.is_empty(), "sidecar invoked");
        assert!(!guard[0].pairs.is_empty(), "sidecar carries pairs");
    }

    #[test]
    fn registers_in_a_cascade() {
        let mut c = Cascade::new();
        c.register(Box::new(StructContrastTier::new(store())));
        assert!(c.tier_ids().contains(&TierId::L1_375StructContrast));
    }

    #[test]
    fn provenance_is_estimator() {
        assert_eq!(
            StructContrastTier::<InMemoryKnowledgeStore>::provenance(),
            Provenance::Estimator,
        );
    }

    #[test]
    fn verdict_round_trips_through_json() {
        let v = ContrastVerdict::Oppose;
        let s = serde_json::to_string(&v).expect("ser");
        // Lowercase per the rename_all attribute.
        assert_eq!(s, "\"oppose\"");
        let back: ContrastVerdict = serde_json::from_str(&s).expect("de");
        assert_eq!(v, back);
    }

    #[test]
    fn verdict_from_relation_maps_align_oppose_partial_unknown() {
        assert_eq!(
            ContrastVerdict::from_relation(ContrastRelation::Align),
            ContrastVerdict::Align
        );
        assert_eq!(
            ContrastVerdict::from_relation(ContrastRelation::Oppose),
            ContrastVerdict::Oppose
        );
        assert_eq!(
            ContrastVerdict::from_relation(ContrastRelation::Partial),
            ContrastVerdict::Partial
        );
        assert_eq!(
            ContrastVerdict::from_relation(ContrastRelation::Unknown),
            ContrastVerdict::Unknown
        );
    }

    #[test]
    fn malformed_envelope_returns_tier_failed() {
        let mut t = StructContrastTier::new(store());
        let q = Query {
            input: QueryInput::Structured(b"not json".to_vec()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let err = t.try_answer(&q, 1.0).expect_err("should error");
        match err {
            AnswerError::TierFailed { tier, .. } => {
                assert_eq!(tier, TierId::L1_375StructContrast);
            }
            other => panic!("expected TierFailed, got {other:?}"),
        }
    }
}
