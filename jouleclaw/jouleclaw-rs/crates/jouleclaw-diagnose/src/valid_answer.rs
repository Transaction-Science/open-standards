//! The ValidAnswerModel checkers (spec §6.1).
//!
//! Five named families of violations. Each free function returns a
//! `Vec<Violation>` keyed by the specific failure observed; the
//! conflict-directed search composes them in cost order (cheap
//! checks first, then focused entailment).
//!
//! Severity convention:
//!   - `Critical` — must be resolved before composition (RE_ROUTE
//!     or REFUSE territory).
//!   - `Major` — resolvable via DEGRADE_GRACEFULLY with caveats.
//!   - `Minor` — informational; degrade silently.
//!
//! Retrievability convention (in `Violation.detail`): a
//! `"retrievable": true` field marks a violation that more
//! retrieval could fix. The verdict logic in §6.6 reads this to
//! decide RE_ROUTE vs REFUSE.

use chrono::Utc;
use serde_json::json;

use jouleclaw_schema::{
    AtomicClaim, AuthorityRecord, EntailmentLabel, EntailmentResult, QueryPlan, RetrievedItem,
    Violation, ViolationSeverity,
};

/// **Coverage** — every sub-query has at least one supporting
/// retrieved item (spec §6.1).
pub fn coverage_violations(plan: &QueryPlan, items: &[RetrievedItem]) -> Vec<Violation> {
    let mut out = Vec::new();
    for sq in &plan.decomposition {
        let covered = items
            .iter()
            .any(|it| it.retrieval_context.sub_id == sq.sub_id);
        if !covered {
            out.push(Violation {
                violation_id: format!("coverage:{}", sq.sub_id),
                message: format!(
                    "sub-query {:?} has no retrieved items covering it",
                    sq.sub_id
                ),
                severity: ViolationSeverity::Critical,
                claim_id: None,
                item_id: None,
                detail: json!({
                    "kind": "coverage",
                    "sub_id": sq.sub_id,
                    "retrievable": true,
                }),
            });
        }
    }
    out
}

/// **Grounding** — every asserted claim is entailed by at least one
/// retrieved item (spec §6.1, §6.4).
///
/// Lookup is by `(claim_id, item_id)` pairs in the supplied
/// `EntailmentResult`s. Stakes drive severity: a `Critical` claim
/// with no entailment is itself `Critical`; a `Low` claim is at
/// most `Minor`.
pub fn grounding_violations(
    claims: &[AtomicClaim],
    entailments: &[EntailmentResult],
) -> Vec<Violation> {
    let mut out = Vec::new();
    for claim in claims {
        let entailing: Vec<&EntailmentResult> = entailments
            .iter()
            .filter(|e| e.claim_id == claim.claim_id)
            .filter(|e| matches!(e.label, EntailmentLabel::Entails))
            .collect();
        if entailing.is_empty() {
            let severity = severity_from_stakes(claim.stakes);
            out.push(Violation {
                violation_id: format!("grounding:{}", claim.claim_id),
                message: format!(
                    "claim {:?} (segment {:?}) has no entailing source",
                    claim.text, claim.segment_id
                ),
                severity,
                claim_id: Some(claim.claim_id),
                item_id: None,
                detail: json!({
                    "kind": "grounding",
                    "claim_text": claim.text,
                    "segment_id": claim.segment_id,
                    "stakes": format!("{:?}", claim.stakes).to_lowercase(),
                    "retrievable": true,
                }),
            });
        }
    }
    out
}

/// **Authority** — cited sources meet the query's minimum tier
/// requirement (spec §6.1).
///
/// We check that at least one retrieved item per sub-query meets
/// `plan.constraints.minimum_authority_tier`. The check inspects
/// `KnowledgeAxes.source_authority_tier` on the item and, when an
/// explicit `AuthorityRecord` is supplied, prefers that record's
/// tier.
pub fn authority_violations(
    plan: &QueryPlan,
    items: &[RetrievedItem],
    authority: &[AuthorityRecord],
) -> Vec<Violation> {
    let min_tier = plan.constraints.minimum_authority_tier;
    let mut out = Vec::new();
    for sq in &plan.decomposition {
        let sq_items: Vec<&RetrievedItem> = items
            .iter()
            .filter(|it| it.retrieval_context.sub_id == sq.sub_id)
            .collect();
        if sq_items.is_empty() {
            // Coverage will already have flagged this; nothing more
            // to say here.
            continue;
        }
        let any_meets = sq_items
            .iter()
            .any(|it| effective_tier(it, authority) <= min_tier);
        if !any_meets {
            let highest_seen = sq_items
                .iter()
                .map(|it| effective_tier(it, authority))
                .min()
                .unwrap_or(u8::MAX);
            out.push(Violation {
                violation_id: format!("authority:{}", sq.sub_id),
                message: format!(
                    "sub-query {:?} required tier ≤{} but best available is tier {}",
                    sq.sub_id, min_tier, highest_seen
                ),
                severity: ViolationSeverity::Critical,
                claim_id: None,
                item_id: None,
                detail: json!({
                    "kind": "authority",
                    "sub_id": sq.sub_id,
                    "required_tier": min_tier,
                    "found_tier": highest_seen,
                    "retrievable": true,
                }),
            });
        }
    }
    out
}

/// **Freshness** — if the plan demands fresh information, at least
/// one retrieved item per sub-query must satisfy its
/// [`KnowledgeAxes::freshness_budget_seconds`] window (spec §6.1).
pub fn freshness_violations(plan: &QueryPlan, items: &[RetrievedItem]) -> Vec<Violation> {
    if !plan.constraints.freshness_required {
        return Vec::new();
    }
    let now = Utc::now();
    let mut out = Vec::new();
    for sq in &plan.decomposition {
        let sq_items: Vec<&RetrievedItem> = items
            .iter()
            .filter(|it| it.retrieval_context.sub_id == sq.sub_id)
            .collect();
        if sq_items.is_empty() {
            continue;
        }
        let any_fresh = sq_items.iter().any(|it| {
            let Some(budget_secs) = it.knowledge_axes.freshness_budget_seconds() else {
                return true; // INVARIANT class — never stale
            };
            let staleness = (now - it.temporal.retrieval_timestamp).num_seconds().max(0)
                as u64;
            staleness <= budget_secs
        });
        if !any_fresh {
            out.push(Violation {
                violation_id: format!("freshness:{}", sq.sub_id),
                message: format!(
                    "sub-query {:?} required fresh sources but all retrieved items \
                     exceed their temporal-stability freshness budget",
                    sq.sub_id
                ),
                severity: ViolationSeverity::Critical,
                claim_id: None,
                item_id: None,
                detail: json!({
                    "kind": "freshness",
                    "sub_id": sq.sub_id,
                    "retrievable": true,
                }),
            });
        }
    }
    out
}

/// **Consistency** — contradictions are surfaced, not hidden (spec
/// §6.1, invariant I6). When two retrieved items entail mutually
/// contradictory claims, the violation is flagged Major (always
/// retrievable=false — we don't paper over conflicts).
pub fn consistency_violations(
    claims: &[AtomicClaim],
    entailments: &[EntailmentResult],
) -> Vec<Violation> {
    let mut out = Vec::new();
    for claim in claims {
        let entailing = entailments
            .iter()
            .filter(|e| e.claim_id == claim.claim_id)
            .filter(|e| matches!(e.label, EntailmentLabel::Entails))
            .count();
        let contradicting = entailments
            .iter()
            .filter(|e| e.claim_id == claim.claim_id)
            .filter(|e| matches!(e.label, EntailmentLabel::Contradicts))
            .count();
        if entailing > 0 && contradicting > 0 {
            out.push(Violation {
                violation_id: format!("consistency:{}", claim.claim_id),
                message: format!(
                    "claim {:?} has both entailing ({}) and contradicting ({}) sources",
                    claim.text, entailing, contradicting
                ),
                severity: ViolationSeverity::Major,
                claim_id: Some(claim.claim_id),
                item_id: None,
                detail: json!({
                    "kind": "consistency",
                    "claim_text": claim.text,
                    "entailing_count": entailing,
                    "contradicting_count": contradicting,
                    "retrievable": false,
                }),
            });
        }
    }
    out
}

/// Convert a claim's [`ClaimStakes`] to the matching violation
/// severity for grounding failures.
fn severity_from_stakes(stakes: jouleclaw_schema::ClaimStakes) -> ViolationSeverity {
    use jouleclaw_schema::ClaimStakes;
    match stakes {
        ClaimStakes::Critical | ClaimStakes::High => ViolationSeverity::Critical,
        ClaimStakes::Medium => ViolationSeverity::Major,
        ClaimStakes::Low => ViolationSeverity::Minor,
    }
}

/// Effective authority tier for an item. Prefers the explicit
/// `AuthorityRecord` when one matches the item's `source_id`; falls
/// back to `KnowledgeAxes.source_authority_tier`.
fn effective_tier(item: &RetrievedItem, authority: &[AuthorityRecord]) -> u8 {
    let from_record = authority
        .iter()
        .find(|r| r.source_id == item.source_id)
        .map(|r| r.tier.as_u8());
    from_record.unwrap_or(item.knowledge_axes.source_authority_tier)
}

/// Helper exposed for tests/diagnostics: whether a violation flags
/// itself retrievable (consumable by [`crate::verdict::determine_verdict`]).
pub fn is_retrievable(v: &Violation) -> bool {
    v.detail
        .get("retrievable")
        .and_then(|x| x.as_bool())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use jouleclaw_schema::*;
    use uuid::Uuid;

    fn axes_with_tier(tier: u8) -> KnowledgeAxes {
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
            certainty: 0.95,
            certainty_basis: "test".into(),
            source_uri: None,
            source_authority_tier: tier,
            extraction_method: None,
            citation_chain: vec![],
            metadata: Default::default(),
        }
    }

    fn item_for(sub_id: &str, source_id: &str, tier: u8) -> RetrievedItem {
        RetrievedItem {
            schema_version: "2.0".into(),
            item_id: Uuid::new_v4(),
            source_id: source_id.into(),
            source_url: None,
            source_type: SourceType::StructuredKb,
            content: Content {
                modality: Modality::Text,
                text: Some("text".into()),
                media_ref: None,
                structured: None,
                excerpt_span: None,
            },
            retrieval_context: RetrievalContext {
                retriever_id: "wikidata".into(),
                matched_against: "test".into(),
                sub_id: sub_id.into(),
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
            knowledge_axes: axes_with_tier(tier),
            metadata: Default::default(),
        }
    }

    fn plan_with(sub_ids: &[&str], min_tier: u8) -> QueryPlan {
        let decomposition: Vec<SubQuery> = sub_ids
            .iter()
            .map(|s| SubQuery {
                sub_id: (*s).into(),
                text: format!("question {s}"),
                depends_on: vec![],
                required_modalities: vec![Modality::Text],
                target_stores: vec!["wikidata".into()],
                priority: 1.0,
                rap_id: "rap".into(),
            })
            .collect();
        let mut constraints = Constraints::default();
        constraints.minimum_authority_tier = min_tier;
        QueryPlan {
            schema_version: "2.0".into(),
            plan_id: Uuid::new_v4(),
            original_query: OriginalQuery {
                text: Some("q".into()),
                image_ref: None,
                audio_ref: None,
                video_ref: None,
                language_detected: "en".into(),
                timestamp: Utc::now(),
            },
            intent: Intent::Lookup,
            modalities_in: vec![Modality::Text],
            modalities_out: vec![Modality::Text],
            decomposition,
            constraints,
            budget: Budget::default(),
            invariants_satisfied: PlanInvariants {
                all_subqueries_have_at_least_one_store: true,
                dependency_graph_is_acyclic: true,
                total_estimated_latency_within_budget: true,
                estimated_cost_within_budget: true,
                required_modalities_covered: true,
            },
            metadata: Default::default(),
        }
    }

    fn claim_with(text: &str, stakes: ClaimStakes) -> AtomicClaim {
        AtomicClaim {
            schema_version: "2.0".into(),
            claim_id: Uuid::new_v4(),
            text: text.into(),
            segment_id: "s0".into(),
            stakes,
            knowledge_axes: axes_with_tier(1),
            atomization_notes: None,
            metadata: Default::default(),
        }
    }

    fn entail(claim_id: Uuid, item_id: Uuid, label: EntailmentLabel) -> EntailmentResult {
        EntailmentResult {
            schema_version: "2.0".into(),
            result_id: Uuid::new_v4(),
            claim_id,
            premise_item_ids: vec![item_id],
            label,
            label_probabilities: EntailmentProbabilities {
                entails: 0.7,
                neutral: 0.2,
                contradicts: 0.1,
            },
            running_e_value: None,
            model_id: "fixture".into(),
            joules_spent: 1.0,
            metadata: Default::default(),
        }
    }

    #[test]
    fn coverage_violation_when_sub_query_unanswered() {
        let plan = plan_with(&["q0", "q1"], 3);
        let items = vec![item_for("q0", "wikidata:Q90", 1)];
        let v = coverage_violations(&plan, &items);
        assert_eq!(v.len(), 1);
        assert!(v[0].violation_id.ends_with("q1"));
        assert!(is_retrievable(&v[0]));
    }

    #[test]
    fn authority_violation_when_tier_unmet() {
        let plan = plan_with(&["q0"], 2); // minimum tier 2
        let items = vec![item_for("q0", "blog:post", 4)];
        let v = authority_violations(&plan, &items, &[]);
        assert_eq!(v.len(), 1);
        assert_eq!(
            v[0].detail.get("found_tier").and_then(|x| x.as_u64()).unwrap(),
            4
        );
    }

    #[test]
    fn authority_record_overrides_axes_tier() {
        let plan = plan_with(&["q0"], 2);
        // Axes claim tier 1 but the AuthorityRecord downgrades to 4.
        let items = vec![item_for("q0", "wikidata:Q90", 1)];
        let rec = AuthorityRecord {
            schema_version: "2.0".into(),
            record_id: Uuid::new_v4(),
            source_id: "wikidata:Q90".into(),
            tier: AuthorityTier::Community,
            dimensions: Default::default(),
            assessed_at: Utc::now(),
            rationale: None,
            metadata: Default::default(),
        };
        let v = authority_violations(&plan, &items, &[rec]);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn freshness_skipped_when_not_required() {
        let mut plan = plan_with(&["q0"], 3);
        plan.constraints.freshness_required = false;
        let items = vec![item_for("q0", "x", 1)];
        let v = freshness_violations(&plan, &items);
        assert!(v.is_empty());
    }

    #[test]
    fn freshness_violation_when_stale() {
        let mut plan = plan_with(&["q0"], 3);
        plan.constraints.freshness_required = true;
        let mut item = item_for("q0", "x", 1);
        // VeryFast → 1 hour budget. Backdate the retrieval by 2 hours.
        item.knowledge_axes.temporal_stability = TemporalStabilityClass::VeryFast;
        item.temporal.retrieval_timestamp = Utc::now() - Duration::hours(2);
        let v = freshness_violations(&plan, &[item]);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn grounding_critical_when_high_stakes_claim_unsupported() {
        let claim = claim_with("Take 5mg of X", ClaimStakes::Critical);
        let v = grounding_violations(&[claim], &[]);
        assert_eq!(v.len(), 1);
        assert!(matches!(v[0].severity, ViolationSeverity::Critical));
    }

    #[test]
    fn grounding_minor_when_low_stakes_claim_unsupported() {
        let claim = claim_with("It might be sunny", ClaimStakes::Low);
        let v = grounding_violations(&[claim], &[]);
        assert_eq!(v.len(), 1);
        assert!(matches!(v[0].severity, ViolationSeverity::Minor));
    }

    #[test]
    fn grounding_clears_when_entailed() {
        let claim = claim_with("Paris is the capital of France", ClaimStakes::Medium);
        let item = item_for("q0", "x", 1);
        let e = entail(claim.claim_id, item.item_id, EntailmentLabel::Entails);
        let v = grounding_violations(&[claim], &[e]);
        assert!(v.is_empty());
    }

    #[test]
    fn consistency_violation_when_both_entail_and_contradict() {
        let claim = claim_with("X is true", ClaimStakes::Medium);
        let it1 = item_for("q0", "x", 1);
        let it2 = item_for("q0", "y", 1);
        let e1 = entail(claim.claim_id, it1.item_id, EntailmentLabel::Entails);
        let e2 = entail(claim.claim_id, it2.item_id, EntailmentLabel::Contradicts);
        let v = consistency_violations(&[claim], &[e1, e2]);
        assert_eq!(v.len(), 1);
        assert!(matches!(v[0].severity, ViolationSeverity::Major));
        assert!(!is_retrievable(&v[0]));
    }
}
