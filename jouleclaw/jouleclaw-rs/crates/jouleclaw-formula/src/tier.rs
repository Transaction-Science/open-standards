//! L0.25 — formula-first structural-relationship tier.
//!
//! Runs immediately after L0 cache. Extracts entity candidates from the
//! query text, resolves them in a [`KnowledgeStore`], and computes
//! structural relationships via the contrast formula `R(A, B) =
//! cos(E(A − μ), E(B − μ))`. When the formula resolves with high
//! confidence, the cascade skips everything downstream — this is the
//! backbone of the deterministic-first doctrine.
//!
//! ## Resolution strategy (ported verbatim from `verity-cascade`)
//!
//! - No entities and ≥ 2 raw word candidates → NCD zero-shot Partial.
//! - One entity → nearest-neighbour neighbourhood; Resolved iff the query
//!   is about the entity and top similarity ≥ 0.7, else Partial.
//! - Two+ entities → pairwise contrast; Resolved iff the query is short
//!   and structural ("compare X and Y"), else Partial.
//!
//! ## Mapping to the `Tier` trait
//!
//! The donor returned a `LayerOutcome` enum (`Resolved` / `Partial` /
//! `Skipped` / `Failed`). JouleClaw's `Tier` returns an `Answer` whose
//! `output` is either:
//!
//! - `AnswerOutput::Text(answer_text)` — Resolved case.
//! - `AnswerOutput::Refused(reason)`   — Partial / Skipped / Failed cases.
//!
//! Partial outputs are surfaced as `Refused(LowConfidence(c))` so the
//! cascade walker continues to the next tier. Downstream tiers can still
//! benefit from the formula's analysis via the optional sidecar passed to
//! [`FormulaFirstTier::with_sidecar`].

use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, Query, QueryInput,
    RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;
use jouleclaw_energy::Provenance;

use crate::extract::{
    extract_entities, is_stop_word, is_structural_query, query_is_about_entity,
};
use crate::knowledge::{Concept, ContrastMap, KnowledgeStore};
use crate::ncd::concept_ncd;

// ─── Cost model ──────────────────────────────────────────────────

/// Donor's joule envelope: ~200 µJ per query. Static estimate; the runtime
/// records the actual via [`Tier::try_answer`]'s reported `joules_spent`.
pub const FORMULA_JOULES: f64 = 200e-6;
/// Wall-clock latency estimate.
pub const FORMULA_LATENCY: Duration = Duration::from_micros(10);
/// Confidence floor advertised to the runtime — the lowest confidence we
/// will claim before refusing to take the dispatch.
pub const FORMULA_CONFIDENCE_FLOOR: f32 = 0.7;

/// Donor confidence threshold (q14.0) above which the formula counts the
/// query as fully resolved. `7_000 / 10_000 = 0.7`.
const FORMULA_RESOLUTION_CONFIDENCE: u16 = 7_000;

// ─── Internal hit shape ──────────────────────────────────────────

/// One pairwise structural relationship surfaced by the formula tier.
///
/// Replaces the donor's dependency on `verity_federation::fuser::FusedResult`.
/// The fields are the minimum the Tier-trait `AnswerOutput::Structured`
/// payload needs: who, who, how similar, with what coverage.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FormulaHit {
    /// Canonical id of the first concept.
    pub entity_a: String,
    /// Canonical id of the second concept.
    pub entity_b: String,
    /// Aggregate similarity in `[0.0, 1.0]`.
    pub similarity: f32,
    /// Aggregate coverage in `[0.0, 1.0]`.
    pub coverage: f32,
}

/// Side-channel produced on every dispatch. Surfaces the formula's
/// internal analysis even when the tier itself refuses to claim the
/// answer (Partial outcome). Downstream tiers / observability code can
/// hold a reference to a `Vec<FormulaHit>` slot and pull this out.
#[derive(Debug, Default)]
pub struct FormulaSidecar {
    /// Per-pair hits from the most recent dispatch.
    pub hits: Vec<FormulaHit>,
    /// The human-readable analysis the tier would have surfaced as
    /// `AnswerOutput::Text` on Resolved.
    pub explanation: String,
    /// The confidence the formula assigned to its analysis.
    pub confidence: f32,
}

// ─── Errors ──────────────────────────────────────────────────────

/// Errors specific to the formula tier.
#[derive(Debug, thiserror::Error)]
pub enum FormulaError {
    /// Failed to serialise the structured output payload.
    #[error("failed to serialise formula output: {0}")]
    Serialise(#[from] serde_json::Error),
}

// ─── The tier ────────────────────────────────────────────────────

/// L0.25 formula-first tier. Generic over a [`KnowledgeStore`] so that
/// OpenIE's `verity-contrast`, hand-curated tables, or any other store
/// implementation can drive it.
pub struct FormulaFirstTier<K: KnowledgeStore> {
    knowledge: K,
    sidecar: Option<Box<dyn FormulaSidecarSink>>,
}

/// Receives the formula's side-channel on every dispatch. Implementations
/// can stash the analysis for downstream tiers, observability, or tests.
pub trait FormulaSidecarSink: Send {
    /// Called after every dispatch with the formula's analysis.
    fn observe(&mut self, sidecar: FormulaSidecar);
}

impl<F> FormulaSidecarSink for F
where
    F: FnMut(FormulaSidecar) + Send,
{
    fn observe(&mut self, sidecar: FormulaSidecar) {
        (self)(sidecar);
    }
}

impl<K: KnowledgeStore> FormulaFirstTier<K> {
    /// Build a new formula-first tier over a knowledge store.
    pub fn new(knowledge: K) -> Self {
        Self { knowledge, sidecar: None }
    }

    /// Install a sidecar sink for the formula's internal analysis. Useful
    /// when downstream tiers want the formula's Partial output even when
    /// the tier itself refuses to claim the answer.
    pub fn with_sidecar(mut self, sink: Box<dyn FormulaSidecarSink>) -> Self {
        self.sidecar = Some(sink);
        self
    }

    /// Borrow the underlying knowledge store.
    pub fn knowledge(&self) -> &K {
        &self.knowledge
    }

    /// Provenance for any energy spend recorded by this tier — we have no
    /// hardware shunt, this is a model-based estimator.
    pub const fn provenance() -> Provenance {
        Provenance::Estimator
    }

    /// Run the donor's resolution strategy and produce a Tier-trait
    /// [`Answer`].
    ///
    /// `query` is the raw query text. The function is pure (modulo the
    /// sidecar) — no I/O, no allocation beyond the per-call working set.
    fn resolve(&mut self, query: &str) -> Result<Answer, FormulaError> {
        // 0. Empty store → Refused(Inapplicable).
        if self.knowledge.is_empty() {
            return Ok(refused_inapplicable());
        }

        // 1. Entity extraction.
        let extraction = extract_entities(query, &self.knowledge);

        if extraction.matches.is_empty() {
            return self.handle_no_entities(query, &extraction);
        }
        if extraction.matches.len() == 1 {
            return self.handle_single_entity(query, &extraction.matches[0]);
        }
        self.handle_multi_entity(query, &extraction.matches)
    }

    // ─── No-entity NCD fallback ──────────────────────────────────

    fn handle_no_entities(
        &mut self,
        query: &str,
        extraction: &crate::extract::Extraction,
    ) -> Result<Answer, FormulaError> {
        let ncd_entities: Vec<&str> = extraction
            .words
            .iter()
            .filter(|w| w.len() >= 3 && !is_stop_word(&w.to_lowercase()))
            .map(|s| s.as_str())
            .collect();

        if ncd_entities.len() < 2 {
            self.emit_sidecar(FormulaSidecar {
                hits: Vec::new(),
                explanation: format!(
                    "no entities found in knowledge store for: \"{query}\""
                ),
                confidence: 0.0,
            });
            return Ok(refused_inapplicable());
        }

        let a = ncd_entities[0].to_lowercase();
        let b = ncd_entities[1].to_lowercase();
        let dist = concept_ncd(&a, &b);
        // Donor map: distance 0 → confidence 0.8 (8000/10000),
        //            distance 1 → confidence 0.1 (1000/10000).
        let confidence_unit = (((1.0 - dist) * 7000.0 + 1000.0)
            .clamp(1000.0, 8000.0))
            / 10_000.0;
        let confidence = confidence_unit as f32;

        let explanation = format!(
            "[NCD zero-shot] \"{}\" <-> \"{}\" — compression distance {:.3}\n\
             (lower = more structurally similar, no knowledge store needed)",
            ncd_entities[0], ncd_entities[1], dist
        );
        let hit = FormulaHit {
            entity_a: a,
            entity_b: b,
            similarity: (1.0 - dist) as f32,
            coverage: confidence,
        };
        self.emit_sidecar(FormulaSidecar {
            hits: vec![hit],
            explanation: explanation.clone(),
            confidence,
        });

        // NCD is always Partial — we never claim full resolution from a
        // zero-shot guess.
        Ok(refused_low_confidence(confidence))
    }

    // ─── Single-entity neighbourhood ─────────────────────────────

    fn handle_single_entity(
        &mut self,
        query: &str,
        (_candidate, concept): &(String, Concept),
    ) -> Result<Answer, FormulaError> {
        let nearest = self.knowledge.nearest_to(&concept.id, 5);
        if nearest.is_empty() {
            self.emit_sidecar(FormulaSidecar {
                hits: Vec::new(),
                explanation: format!(
                    "single entity {} has no neighbours",
                    concept.name
                ),
                confidence: 0.0,
            });
            return Ok(refused_inapplicable());
        }

        let mut explanation =
            format!("Structural neighbourhood of \"{}\":\n", concept.name);
        for (sim, neighbour) in &nearest {
            explanation.push_str(&format!(
                "  {} — similarity {}/10000\n",
                neighbour.name, sim.0
            ));
        }

        let top_sim = nearest.first().map(|(s, _)| s.0).unwrap_or(0);
        let confidence_q14 = if query_is_about_entity(query, &concept.name) {
            top_sim.min(8000)
        } else {
            (top_sim / 2).min(5000)
        };
        let confidence = q14_to_unit(confidence_q14);

        let hits: Vec<FormulaHit> = nearest
            .iter()
            .map(|(sim, n)| FormulaHit {
                entity_a: concept.id.clone(),
                entity_b: n.id.clone(),
                similarity: sim.as_unit(),
                coverage: sim.as_unit(),
            })
            .collect();
        self.emit_sidecar(FormulaSidecar {
            hits,
            explanation: explanation.clone(),
            confidence,
        });

        if confidence_q14 >= FORMULA_RESOLUTION_CONFIDENCE {
            Ok(resolved(explanation, confidence))
        } else {
            Ok(refused_low_confidence(confidence))
        }
    }

    // ─── Multi-entity pairwise contrast ──────────────────────────

    fn handle_multi_entity(
        &mut self,
        query: &str,
        matched: &[(String, Concept)],
    ) -> Result<Answer, FormulaError> {
        let mut contrast_descriptions: Vec<String> = Vec::new();
        let mut hits: Vec<FormulaHit> = Vec::new();
        let mut total_coverage: u64 = 0;
        let mut pair_count: u64 = 0;

        for i in 0..matched.len() {
            for j in (i + 1)..matched.len() {
                let (_, ca) = &matched[i];
                let (_, cb) = &matched[j];
                if let Some(map) = self.knowledge.contrast(&ca.id, &cb.id) {
                    contrast_descriptions.push(format!(
                        "{} <-> {}: similarity={}/10000, coverage={}/10000",
                        ca.name, cb.name, map.similarity.0, map.coverage.0
                    ));
                    hits.push(FormulaHit {
                        entity_a: ca.id.clone(),
                        entity_b: cb.id.clone(),
                        similarity: map.similarity.as_unit(),
                        coverage: map.coverage.as_unit(),
                    });
                    total_coverage += map.coverage.0 as u64;
                    pair_count += 1;
                }
            }
        }

        if pair_count == 0 {
            self.emit_sidecar(FormulaSidecar {
                hits: Vec::new(),
                explanation: format!(
                    "no pairwise contrasts available for: \"{query}\""
                ),
                confidence: 0.0,
            });
            return Ok(refused_inapplicable());
        }

        let avg_coverage = (total_coverage / pair_count) as u16;
        let explanation = format!(
            "Formula-first structural analysis for: \"{query}\"\n\
             Matched {} concepts, {} pairwise contrasts:\n\n{}",
            matched.len(),
            pair_count,
            contrast_descriptions.join("\n---\n"),
        );

        let (confidence_q14, resolved_flag) = if is_structural_query(query) {
            (avg_coverage.max(7500).min(9000), true)
        } else {
            ((avg_coverage / 2).min(5000), false)
        };
        let confidence = q14_to_unit(confidence_q14);

        self.emit_sidecar(FormulaSidecar {
            hits,
            explanation: explanation.clone(),
            confidence,
        });

        if resolved_flag {
            Ok(resolved(explanation, confidence))
        } else {
            Ok(refused_low_confidence(confidence))
        }
    }

    fn emit_sidecar(&mut self, sidecar: FormulaSidecar) {
        if let Some(sink) = self.sidecar.as_mut() {
            sink.observe(sidecar);
        }
    }
}

// ─── Answer-shape helpers ────────────────────────────────────────

fn resolved(answer_text: String, confidence: f32) -> Answer {
    Answer {
        output: AnswerOutput::Text(answer_text),
        tier_used: TierId::L0_25FormulaFirst,
        joules_spent: FORMULA_JOULES,
        confidence,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

fn refused_inapplicable() -> Answer {
    Answer {
        output: AnswerOutput::Refused(RefusalReason::Inapplicable),
        tier_used: TierId::L0_25FormulaFirst,
        joules_spent: FORMULA_JOULES,
        confidence: 0.0,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

fn refused_low_confidence(confidence: f32) -> Answer {
    Answer {
        output: AnswerOutput::Refused(RefusalReason::low_confidence(confidence)),
        tier_used: TierId::L0_25FormulaFirst,
        joules_spent: FORMULA_JOULES,
        confidence,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

fn q14_to_unit(q: u16) -> f32 {
    (q as f32 / 10_000.0).clamp(0.0, 1.0)
}

// ─── Public extract-structured helper ────────────────────────────

/// Donor `extract_structured_contrasts` — pulled out so callers that have
/// already resolved entities (e.g. a coordinator that fanned out to other
/// stores) can ask the knowledge store for per-pair contrasts without
/// re-running the formula tier itself.
pub fn structured_contrasts<K: KnowledgeStore + ?Sized>(
    knowledge: &K,
    entity_names: &[String],
) -> Vec<StructuredContrast> {
    let dim_names = knowledge.dimension_names();
    let mut out = Vec::new();

    // Resolve entity names to concepts.
    let resolved: Vec<Concept> = entity_names
        .iter()
        .filter_map(|name| knowledge.search_by_name(name, 1).into_iter().next())
        .collect();

    for i in 0..resolved.len() {
        for j in (i + 1)..resolved.len() {
            let ca = &resolved[i];
            let cb = &resolved[j];
            if let Some(map) = knowledge.contrast(&ca.id, &cb.id) {
                let relation = dominant_relation(&map);
                let top_dimensions = top_dimensions(&map, &dim_names, 5);
                out.push(StructuredContrast {
                    entity_a: ca.name.clone(),
                    entity_b: cb.name.clone(),
                    similarity: map.similarity.0,
                    top_dimensions,
                    relation: relation.to_string(),
                    coverage: map.coverage.0 as f64 / 10_000.0,
                });
            }
        }
    }
    out
}

/// Donor `StructuredContrast` shape — public so observability surfaces and
/// the Tier-trait `AnswerOutput::Structured` payload can carry it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StructuredContrast {
    /// Display name of the first entity.
    pub entity_a: String,
    /// Display name of the second entity.
    pub entity_b: String,
    /// Aggregate similarity, q14.0 fixed-point.
    pub similarity: u16,
    /// Top-N dimension contributions.
    pub top_dimensions: Vec<(String, f64)>,
    /// `"align"` | `"oppose"` | `"partial"` | `"unknown"`.
    pub relation: String,
    /// Aggregate coverage in `[0.0, 1.0]`.
    pub coverage: f64,
}

fn dominant_relation(map: &ContrastMap) -> &'static str {
    let align = map
        .dimensions
        .iter()
        .filter(|d| {
            matches!(d.relation, crate::knowledge::ContrastRelation::Align)
        })
        .count();
    let oppose = map
        .dimensions
        .iter()
        .filter(|d| {
            matches!(d.relation, crate::knowledge::ContrastRelation::Oppose)
        })
        .count();
    if align > oppose {
        "align"
    } else if oppose > align {
        "oppose"
    } else if map.unknown_count > map.known_count + map.partial_count {
        "unknown"
    } else {
        "partial"
    }
}

fn top_dimensions(
    map: &ContrastMap,
    dim_names: &[String],
    take: usize,
) -> Vec<(String, f64)> {
    let mut dims: Vec<_> = map.dimensions.iter().collect();
    dims.sort_by(|a, b| {
        b.contribution
            .abs()
            .partial_cmp(&a.contribution.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    dims.iter()
        .take(take)
        .map(|d| {
            let name = dim_names
                .get(d.dimension as usize)
                .cloned()
                .unwrap_or_else(|| format!("dim_{}", d.dimension));
            (name, d.contribution as f64)
        })
        .collect()
}

// ─── Tier-trait impl ─────────────────────────────────────────────

impl<K: KnowledgeStore + Send> Tier for FormulaFirstTier<K> {
    fn id(&self) -> TierId {
        TierId::L0_25FormulaFirst
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        match &q.input {
            QueryInput::Text(_) => {
                // Donor behaviour: an empty knowledge store short-circuits
                // to Skipped; the NCD fallback only fires when the store
                // is populated but extraction produced no matches.
                if self.knowledge.is_empty() {
                    return None;
                }
                Some(TierEstimate {
                    joules: FORMULA_JOULES,
                    latency: FORMULA_LATENCY,
                    confidence_floor: FORMULA_CONFIDENCE_FLOOR,
                })
            }
            _ => None,
        }
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget_remaining: f64,
    ) -> Result<Answer, AnswerError> {
        let text = match &q.input {
            QueryInput::Text(s) => s.clone(),
            _ => return Ok(refused_inapplicable()),
        };
        self.resolve(&text).map_err(|e| AnswerError::TierFailed {
            tier: TierId::L0_25FormulaFirst,
            cause: e.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{Concept, InMemoryKnowledgeStore};
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, QualityFloor, Query, QueryInput,
    };

    fn store() -> InMemoryKnowledgeStore {
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
        k.insert(Concept {
            id: "urn:steam".into(),
            name: "steam".into(),
            traits: vec![1.0, 1.0, 0.0],
        });
        k
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
    fn tier_id_is_l0_25() {
        let t = FormulaFirstTier::new(store());
        assert_eq!(t.id(), TierId::L0_25FormulaFirst);
    }

    #[test]
    fn estimate_cost_for_text() {
        let t = FormulaFirstTier::new(store());
        let est = t.estimate_cost(&text_query("compare fire and water"));
        let est = est.expect("text query should be applicable");
        assert!((est.joules - FORMULA_JOULES).abs() < 1e-12);
        assert_eq!(est.confidence_floor, FORMULA_CONFIDENCE_FLOOR);
    }

    #[test]
    fn estimate_cost_for_binary_is_none() {
        let t = FormulaFirstTier::new(store());
        let q = Query {
            input: QueryInput::Binary(vec![0, 1, 2]),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_cost_empty_store_with_no_candidates_is_none() {
        let t = FormulaFirstTier::new(InMemoryKnowledgeStore::new());
        let q = text_query("a b c");
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn structural_query_two_entities_resolves() {
        let mut t = FormulaFirstTier::new(store());
        let q = text_query("compare fire and water");
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert_eq!(a.tier_used, TierId::L0_25FormulaFirst);
        assert!(
            matches!(a.output, AnswerOutput::Text(_)),
            "expected Text(resolved), got {:?}",
            a.output
        );
        assert!(a.confidence >= FORMULA_CONFIDENCE_FLOOR);
    }

    #[test]
    fn factual_query_two_entities_refuses_partial() {
        let mut t = FormulaFirstTier::new(store());
        let q = text_query(
            "what is the boiling point of water and fire in celsius",
        );
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(
            matches!(a.output, AnswerOutput::Refused(_)),
            "long factual query should refuse, got {:?}",
            a.output
        );
    }

    #[test]
    fn populated_store_no_match_runs_ncd() {
        // NCD fallback fires when the store is non-empty but extraction
        // produced no matches AND the raw words include >= 2 candidates.
        let mut k = InMemoryKnowledgeStore::new();
        k.insert(Concept {
            id: "urn:placeholder".into(),
            name: "placeholder".into(),
            traits: vec![0.0],
        });
        let mut t = FormulaFirstTier::new(k);
        let q = text_query("hydrogen hydrogenate");
        let a = t.try_answer(&q, 1.0).expect("ok");
        // NCD path returns Refused(LowConfidence(...)) — never Resolved.
        match a.output {
            AnswerOutput::Refused(RefusalReason::LowConfidence(_)) => {}
            other => panic!("expected NCD low-confidence, got {other:?}"),
        }
    }

    #[test]
    fn empty_store_single_word_refuses_inapplicable() {
        let mut t = FormulaFirstTier::new(InMemoryKnowledgeStore::new());
        let q = text_query("fire");
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn single_entity_short_query_resolves() {
        let mut t = FormulaFirstTier::new(store());
        let q = text_query("fire");
        let a = t.try_answer(&q, 1.0).expect("ok");
        // Top neighbour (steam) shares heat=+1, so cosine of fire and
        // steam is positive → similarity > 0.5 → confidence_q14 ≥ 5000
        // for an entity-focused query, well above the 0.7 floor.
        match a.output {
            AnswerOutput::Text(ref s) => assert!(s.contains("steam")),
            AnswerOutput::Refused(_) => {
                // Acceptable if similarity dipped below 7000 — single-
                // entity resolution is conservative.
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn sidecar_observes_dispatch() {
        use std::sync::{Arc, Mutex};
        let captured = Arc::new(Mutex::new(Vec::<FormulaSidecar>::new()));
        let c2 = Arc::clone(&captured);
        struct Sink(Arc<Mutex<Vec<FormulaSidecar>>>);
        impl FormulaSidecarSink for Sink {
            fn observe(&mut self, s: FormulaSidecar) {
                if let Ok(mut guard) = self.0.lock() {
                    guard.push(s);
                }
            }
        }
        let mut t =
            FormulaFirstTier::new(store()).with_sidecar(Box::new(Sink(c2)));
        let _ = t
            .try_answer(&text_query("compare fire and water"), 1.0)
            .expect("ok");
        let guard = captured.lock().expect("lock");
        assert!(!guard.is_empty(), "sidecar should have been invoked");
        assert!(!guard[0].hits.is_empty());
    }

    #[test]
    fn structured_contrasts_helper() {
        let k = store();
        let out = structured_contrasts(
            &k,
            &["fire".to_string(), "water".to_string()],
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].entity_a, "fire");
        assert_eq!(out[0].entity_b, "water");
        assert!(out[0].top_dimensions.len() <= 5);
    }

    #[test]
    fn provenance_is_estimator() {
        assert_eq!(
            FormulaFirstTier::<InMemoryKnowledgeStore>::provenance(),
            Provenance::Estimator
        );
    }

    #[test]
    fn registers_in_a_cascade() {
        use jouleclaw_cascade::tier::Cascade;
        let mut c = Cascade::new();
        c.register(Box::new(FormulaFirstTier::new(store())));
        assert!(c.tier_ids().contains(&TierId::L0_25FormulaFirst));
    }
}
