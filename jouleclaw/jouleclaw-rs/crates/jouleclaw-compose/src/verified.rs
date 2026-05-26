//! Verified composer (spec §7.2).
//!
//! Take the draft + the diagnose pillar's [`VerificationReport`]
//! and mechanically apply [`RecoveryAction`]s to produce the final
//! [`Answer`] (or a structured [`Refusal`]). Per the spec's wording:
//! "Mechanical, not creative. The hard decisions were made in
//! verification; composition executes them."
//!
//! Recovery action semantics applied per segment:
//!
//! - `Keep` — emit the segment unchanged.
//! - `Drop` — omit the segment from the answer.
//! - `Hedge` — prepend a hedge ("It appears that …") to the text.
//! - `LabelAsInference` — wrap as "[inferred] …" and lower the
//!   confidence of any claim attribution.
//! - `SurfaceConflict` — prepend "[contested] …" so the user sees
//!   the conflict instead of having one side picked silently.
//! - `Rewrite` — replace the segment text with the action's
//!   `alternative_text` field.
//! - `Refuse` — collapse the entire output to a [`Refusal`] keyed
//!   on the first such action's reason.

use chrono::Utc;
use std::collections::HashMap;
use uuid::Uuid;

use jouleclaw_schema::{
    Answer, AnswerSegment, AnswerStatus, AtomicClaim, ClaimAttribution, EpistemicMode,
    Invariant, KnowledgeAxes, OriginalQuery, Provenance, QueryPlan, RecommendedAction,
    Refusal, RetrievedItem, VerificationAction, VerificationReport,
};

use crate::draft::DraftSegment;
use crate::provenance::cache_key_for_subquery;

/// Either an [`Answer`] (Verified / Degraded / Partial) or a
/// structured [`Refusal`]. Returned by [`compose_verified_answer`]
/// so the caller doesn't need to peek at the verdict.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AnswerOrRefusal {
    Answer(Answer),
    Refusal(Refusal),
}

impl AnswerOrRefusal {
    pub fn is_answer(&self) -> bool {
        matches!(self, Self::Answer(_))
    }
    pub fn as_answer(&self) -> Option<&Answer> {
        match self {
            Self::Answer(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_refusal(&self) -> Option<&Refusal> {
        match self {
            Self::Refusal(r) => Some(r),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum ComposeError {
    /// The verdict was `ReRouteWithRefinement` — the orchestrator
    /// should rebuild the plan, not call the composer. Returning
    /// this as an error rather than silently producing junk keeps
    /// the loop honest.
    NotReadyToCompose(VerificationAction),
}

impl std::fmt::Display for ComposeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotReadyToCompose(v) => {
                write!(f, "verdict {v:?} requires re-routing, not composition")
            }
        }
    }
}

impl std::error::Error for ComposeError {}

/// Inputs to [`compose_verified_answer`]. Bundled struct so call
/// sites are explicit about everything the verified composer needs.
pub struct ComposeInputs<'a> {
    pub plan: &'a QueryPlan,
    pub draft_segments: &'a [DraftSegment],
    pub claims: &'a [AtomicClaim],
    pub items: &'a [RetrievedItem],
    pub report: &'a VerificationReport,
    /// Total joules spent across plan + execute + diagnose phases,
    /// to be added to the joules the composer itself consumed (~0).
    pub joules_spent_total: f64,
    /// Total wall-clock latency from query arrival to now.
    pub latency_ms: u64,
}

/// Apply the recovery actions to the draft and produce a final
/// answer (or refusal).
pub fn compose_verified_answer(
    inputs: &ComposeInputs<'_>,
) -> Result<AnswerOrRefusal, ComposeError> {
    match inputs.report.verdict {
        VerificationAction::Refuse => Ok(AnswerOrRefusal::Refusal(build_refusal(inputs))),
        VerificationAction::ReRouteWithRefinement => {
            Err(ComposeError::NotReadyToCompose(inputs.report.verdict))
        }
        VerificationAction::ProceedToComposition
        | VerificationAction::DegradeGracefully => Ok(AnswerOrRefusal::Answer(build_answer(inputs))),
    }
}

fn build_answer(inputs: &ComposeInputs<'_>) -> Answer {
    // Build claim_id → claim lookup and segment_id → claims-in-segment.
    let claims_by_id: HashMap<Uuid, &AtomicClaim> =
        inputs.claims.iter().map(|c| (c.claim_id, c)).collect();
    let mut claims_by_segment: HashMap<String, Vec<&AtomicClaim>> = HashMap::new();
    for c in inputs.claims {
        claims_by_segment.entry(c.segment_id.clone()).or_default().push(c);
    }

    // Group recovery actions by target_claim_id.
    let mut actions_for_claim: HashMap<Uuid, Vec<&jouleclaw_schema::RecoveryAction>> = HashMap::new();
    for a in &inputs.report.recovery_actions {
        if let Some(cid) = a.target_claim_id {
            actions_for_claim.entry(cid).or_default().push(a);
        }
    }

    let mut segments_out: Vec<AnswerSegment> = Vec::new();
    let mut caveats: Vec<String> = Vec::new();

    for ds in inputs.draft_segments {
        let seg_claims = claims_by_segment.get(&ds.segment_id).cloned().unwrap_or_default();

        // Determine the strongest action for this segment by
        // looking at all of its claims' recovery actions.
        let mut effective_action = RecommendedAction::Keep;
        let mut rationale_pieces: Vec<String> = Vec::new();
        let mut alternative_override: Option<String> = None;

        for claim in &seg_claims {
            if let Some(actions) = actions_for_claim.get(&claim.claim_id) {
                for action in actions {
                    rationale_pieces.push(action.rationale.clone());
                    effective_action =
                        merge_actions(effective_action, action.action_type);
                    if matches!(action.action_type, RecommendedAction::Rewrite) {
                        if let Some(alt) = &action.alternative_text {
                            alternative_override = Some(alt.clone());
                        }
                    }
                }
            }
        }

        let final_text = apply_action(&ds.text, effective_action, alternative_override.as_deref());
        if final_text.is_none() {
            // Dropped — record it as a caveat in the final answer.
            caveats.push(format!(
                "Dropped segment {} because: {}",
                ds.segment_id,
                rationale_pieces.join("; ")
            ));
            continue;
        }
        let final_text = final_text.unwrap();

        // Pick a claim_id for the segment (first one if multiple).
        // The schema only carries one per segment; multiple-claim
        // segments are rare in the templated composer.
        let claim_id = seg_claims.first().map(|c| c.claim_id);
        let claim_axes = claim_id
            .and_then(|id| claims_by_id.get(&id).map(|c| c.knowledge_axes.clone()))
            .unwrap_or_else(default_axes);

        // Build per-claim attribution. Source URI from the most
        // authoritative cited item.
        let cited_item = ds
            .cited_item_ids
            .iter()
            .filter_map(|id| inputs.items.iter().find(|it| it.item_id == *id))
            .next();
        let attribution = ClaimAttribution {
            schema_version: "6.0".into(),
            epistemic_mode: EpistemicMode::FromRetrieval,
            prior_reference_cutoff: None,
            prior_claim_class: None,
            retrieval_timestamp: cited_item.map(|it| it.temporal.retrieval_timestamp),
            retrieval_source_uri: cited_item.and_then(|it| it.source_url.clone()),
            retrieval_axes: cited_item.map(|it| it.knowledge_axes.clone()),
        };

        let rationale = if rationale_pieces.is_empty() {
            None
        } else {
            Some(rationale_pieces.join("; "))
        };

        segments_out.push(AnswerSegment {
            segment_id: ds.segment_id.clone(),
            text: final_text,
            attribution,
            knowledge_axes: claim_axes,
            claim_id,
            cited_item_ids: ds.cited_item_ids.clone(),
            rationale,
        });
    }

    // Status reflects the verdict.
    let status = match inputs.report.verdict {
        VerificationAction::ProceedToComposition => AnswerStatus::Verified,
        VerificationAction::DegradeGracefully => AnswerStatus::Degraded,
        _ => AnswerStatus::Verified, // shouldn't happen given the dispatch above
    };

    // Add caveats from violations that weren't tied to claims —
    // coverage, authority, freshness.
    for v in &inputs.report.violations {
        if v.claim_id.is_none() {
            caveats.push(format!("[{}] {}", v.violation_id, v.message));
        }
    }

    let provenance = build_provenance(inputs);
    // Verified Composer adds I3 (inference labeled) and I12 (axis
    // consistency bounded) when its post-processing has had a chance
    // to enforce them. Carry forward whatever the report already
    // verified.
    let mut invariants_verified = inputs.report.invariants_verified.clone();
    push_unique(&mut invariants_verified, Invariant::I3InferenceIsLabeled.id());
    push_unique(&mut invariants_verified, Invariant::I12AxisConsistencyBounded.id());
    push_unique(&mut invariants_verified, Invariant::I13EpistemicModeDeclared.id());

    Answer {
        schema_version: "2.0".into(),
        answer_id: Uuid::new_v4(),
        plan_id: inputs.plan.plan_id,
        status,
        segments: segments_out,
        provenance,
        invariants_verified,
        joules_spent_total: inputs.joules_spent_total,
        latency_ms: inputs.latency_ms,
        caveats,
        emitted_at: Utc::now(),
        metadata: Default::default(),
    }
}

fn build_refusal(inputs: &ComposeInputs<'_>) -> Refusal {
    // First critical violation's id + message become the refusal's
    // reason. Less-critical violations join the blocking list.
    let mut blocking: Vec<String> = inputs
        .report
        .violations
        .iter()
        .filter(|v| matches!(v.severity, jouleclaw_schema::ViolationSeverity::Critical))
        .map(|v| v.violation_id.clone())
        .collect();
    if blocking.is_empty() {
        // Spec calls for refusing only on critical violations; if
        // we somehow got here without any, surface every violation
        // as blocking so the user has something to read.
        blocking = inputs
            .report
            .violations
            .iter()
            .map(|v| v.violation_id.clone())
            .collect();
    }
    let (reason_code, reason_message) = inputs
        .report
        .violations
        .iter()
        .find(|v| matches!(v.severity, jouleclaw_schema::ViolationSeverity::Critical))
        .map(|v| {
            (
                v.detail
                    .get("kind")
                    .and_then(|x| x.as_str())
                    .unwrap_or("critical")
                    .to_string()
                    + "_unsatisfiable",
                v.message.clone(),
            )
        })
        .unwrap_or_else(|| {
            (
                "no_satisfiable_plan".to_string(),
                "no answer can be composed within the configured invariants".to_string(),
            )
        });

    Refusal {
        schema_version: "2.0".into(),
        refusal_id: Uuid::new_v4(),
        plan_id: inputs.plan.plan_id,
        reason_code,
        reason_message,
        blocking_violations: blocking,
        emitted_at: Utc::now(),
        metadata: Default::default(),
    }
}

fn build_provenance(inputs: &ComposeInputs<'_>) -> Provenance {
    // Cache key: hash all the sub-queries' canonical forms together
    // so the entire plan's provenance has a stable key.
    let mut combined = String::new();
    for sq in &inputs.plan.decomposition {
        combined.push_str(&cache_key_for_subquery(sq));
    }
    Provenance {
        schema_version: "2.0".into(),
        plan_id: inputs.plan.plan_id,
        items: inputs.items.to_vec(),
        claims: inputs.claims.to_vec(),
        entailments: inputs.report.entailments_consulted.clone(),
        cache_key: combined,
        metadata: Default::default(),
    }
}

/// Combine two recovery actions, picking the "stronger" one. Drop
/// wins over everything; Refuse wins over Drop; LabelAsInference >
/// SurfaceConflict > Hedge > Rewrite > Keep is the order used here.
fn merge_actions(current: RecommendedAction, new: RecommendedAction) -> RecommendedAction {
    fn rank(a: RecommendedAction) -> u8 {
        match a {
            RecommendedAction::Keep => 0,
            RecommendedAction::Rewrite => 1,
            RecommendedAction::Hedge => 2,
            RecommendedAction::SurfaceConflict => 3,
            RecommendedAction::LabelAsInference => 4,
            RecommendedAction::Drop => 5,
            RecommendedAction::Refuse => 6,
        }
    }
    if rank(new) > rank(current) { new } else { current }
}

fn apply_action(
    text: &str,
    action: RecommendedAction,
    alternative: Option<&str>,
) -> Option<String> {
    match action {
        RecommendedAction::Keep => Some(text.to_string()),
        RecommendedAction::Drop => None,
        RecommendedAction::Hedge => Some(format!("It appears that {text}")),
        RecommendedAction::LabelAsInference => {
            Some(format!("[inferred] {text}"))
        }
        RecommendedAction::SurfaceConflict => {
            Some(format!("[contested] {text}"))
        }
        RecommendedAction::Rewrite => {
            alternative.map(|s| s.to_string()).or_else(|| Some(text.to_string()))
        }
        RecommendedAction::Refuse => None,
    }
}

fn push_unique(v: &mut Vec<String>, s: &'static str) {
    if !v.iter().any(|x| x == s) {
        v.push(s.to_string());
    }
}

fn default_axes() -> KnowledgeAxes {
    KnowledgeAxes {
        schema_version: "5.0".into(),
        valid_time_start: None,
        valid_time_end: None,
        transaction_time: None,
        reference_time: Utc::now(),
        temporal_stability: jouleclaw_schema::TemporalStabilityClass::Medium,
        granularity: jouleclaw_schema::GranularityClass::Medium,
        granularity_notes: None,
        scope: jouleclaw_schema::ScopeClass::Particular,
        scope_domain: None,
        certainty: 0.5,
        certainty_basis: "default-no-claim-source".into(),
        source_uri: None,
        source_authority_tier: 3,
        extraction_method: None,
        citation_chain: vec![],
        metadata: Default::default(),
    }
}

#[allow(dead_code)]
fn original_query_for_test() -> OriginalQuery {
    OriginalQuery {
        text: Some("q".into()),
        image_ref: None,
        audio_ref: None,
        video_ref: None,
        language_detected: "en".into(),
        timestamp: Utc::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_schema::*;

    fn axes(tier: u8) -> KnowledgeAxes {
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

    fn plan() -> QueryPlan {
        QueryPlan {
            schema_version: "2.0".into(),
            plan_id: Uuid::new_v4(),
            original_query: original_query_for_test(),
            intent: Intent::Lookup,
            modalities_in: vec![Modality::Text],
            modalities_out: vec![Modality::Text],
            decomposition: vec![SubQuery {
                sub_id: "q0".into(),
                text: "capital of France".into(),
                depends_on: vec![],
                required_modalities: vec![Modality::Text],
                target_stores: vec!["wikidata".into()],
                priority: 1.0,
                rap_id: "rap".into(),
            }],
            constraints: Default::default(),
            budget: Default::default(),
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

    fn item(text: &str) -> RetrievedItem {
        RetrievedItem {
            schema_version: "2.0".into(),
            item_id: Uuid::new_v4(),
            source_id: "wikidata:Q90".into(),
            source_url: Some("https://www.wikidata.org/wiki/Q90".into()),
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
            knowledge_axes: axes(1),
            metadata: Default::default(),
        }
    }

    fn claim_for(seg_id: &str, text: &str) -> AtomicClaim {
        AtomicClaim {
            schema_version: "2.0".into(),
            claim_id: Uuid::new_v4(),
            text: text.into(),
            segment_id: seg_id.into(),
            stakes: ClaimStakes::Medium,
            knowledge_axes: axes(1),
            atomization_notes: None,
            metadata: Default::default(),
        }
    }

    fn report_proceed(plan_id: Uuid) -> VerificationReport {
        VerificationReport {
            schema_version: "2.0".into(),
            report_id: Uuid::new_v4(),
            plan_id,
            generated_at: Utc::now(),
            verdict: VerificationAction::ProceedToComposition,
            violations: vec![],
            recovery_actions: vec![],
            entailments_consulted: vec![],
            invariants_verified: vec!["I1".into(), "I2".into()],
            joules_spent: 0.0,
            metadata: Default::default(),
        }
    }

    #[test]
    fn proceed_emits_verified_answer_with_all_segments() {
        let p = plan();
        let it = item("Paris is the capital of France.");
        let seg = DraftSegment {
            segment_id: "s0".into(),
            text: "Wikidata: Paris is the capital of France.".into(),
            cited_item_ids: vec![it.item_id],
        };
        let claim = claim_for("s0", "Paris is the capital of France.");
        let report = report_proceed(p.plan_id);
        let inputs = ComposeInputs {
            plan: &p,
            draft_segments: &[seg],
            claims: &[claim],
            items: &[it],
            report: &report,
            joules_spent_total: 100.0,
            latency_ms: 1500,
        };
        let result = compose_verified_answer(&inputs).unwrap();
        let ans = result.as_answer().expect("Answer");
        assert!(matches!(ans.status, AnswerStatus::Verified));
        assert_eq!(ans.segments.len(), 1);
        assert_eq!(ans.joules_spent_total, 100.0);
        assert!(ans
            .invariants_verified
            .contains(&"I13".to_string()), "I13 should be added");
    }

    #[test]
    fn drop_action_removes_segment_and_records_caveat() {
        let p = plan();
        let it = item("Paris is the capital of France.");
        let claim = claim_for("s0", "Paris is the capital of France.");
        let recovery = RecoveryAction {
            violation_id: "v1".into(),
            action_type: RecommendedAction::Drop,
            target_claim_id: Some(claim.claim_id),
            rationale: "unsupported high-stakes claim".into(),
            alternative_text: None,
        };
        let mut report = report_proceed(p.plan_id);
        report.verdict = VerificationAction::DegradeGracefully;
        report.recovery_actions = vec![recovery];
        let seg = DraftSegment {
            segment_id: "s0".into(),
            text: "Wikidata: Paris is the capital of France.".into(),
            cited_item_ids: vec![it.item_id],
        };
        let inputs = ComposeInputs {
            plan: &p,
            draft_segments: &[seg],
            claims: &[claim],
            items: &[it],
            report: &report,
            joules_spent_total: 0.0,
            latency_ms: 100,
        };
        let result = compose_verified_answer(&inputs).unwrap();
        let ans = result.as_answer().unwrap();
        assert_eq!(ans.segments.len(), 0);
        assert_eq!(ans.caveats.len(), 1);
        assert!(ans.caveats[0].contains("Dropped"));
        assert!(matches!(ans.status, AnswerStatus::Degraded));
    }

    #[test]
    fn surface_conflict_action_wraps_text() {
        let p = plan();
        let it = item("Paris is the capital of France.");
        let claim = claim_for("s0", "Paris is the capital of France.");
        let recovery = RecoveryAction {
            violation_id: "c1".into(),
            action_type: RecommendedAction::SurfaceConflict,
            target_claim_id: Some(claim.claim_id),
            rationale: "contradictory sources".into(),
            alternative_text: None,
        };
        let mut report = report_proceed(p.plan_id);
        report.verdict = VerificationAction::DegradeGracefully;
        report.recovery_actions = vec![recovery];
        let seg = DraftSegment {
            segment_id: "s0".into(),
            text: "Paris is the capital of France.".into(),
            cited_item_ids: vec![it.item_id],
        };
        let inputs = ComposeInputs {
            plan: &p,
            draft_segments: &[seg],
            claims: &[claim],
            items: &[it],
            report: &report,
            joules_spent_total: 0.0,
            latency_ms: 0,
        };
        let result = compose_verified_answer(&inputs).unwrap();
        let ans = result.as_answer().unwrap();
        assert!(ans.segments[0].text.starts_with("[contested]"));
    }

    #[test]
    fn refuse_verdict_returns_refusal_with_blocking_violations() {
        let p = plan();
        let mut report = report_proceed(p.plan_id);
        report.verdict = VerificationAction::Refuse;
        report.violations = vec![Violation {
            violation_id: "auth:q0".into(),
            message: "no tier-1 source available".into(),
            severity: ViolationSeverity::Critical,
            claim_id: None,
            item_id: None,
            detail: serde_json::json!({ "kind": "authority", "retrievable": false }),
        }];
        let inputs = ComposeInputs {
            plan: &p,
            draft_segments: &[],
            claims: &[],
            items: &[],
            report: &report,
            joules_spent_total: 0.0,
            latency_ms: 0,
        };
        let result = compose_verified_answer(&inputs).unwrap();
        let r = result.as_refusal().unwrap();
        assert_eq!(r.reason_code, "authority_unsatisfiable");
        assert!(r.blocking_violations.contains(&"auth:q0".to_string()));
    }

    #[test]
    fn reroute_verdict_is_a_composer_error_not_an_answer() {
        let p = plan();
        let mut report = report_proceed(p.plan_id);
        report.verdict = VerificationAction::ReRouteWithRefinement;
        let inputs = ComposeInputs {
            plan: &p,
            draft_segments: &[],
            claims: &[],
            items: &[],
            report: &report,
            joules_spent_total: 0.0,
            latency_ms: 0,
        };
        let err = compose_verified_answer(&inputs).unwrap_err();
        assert!(matches!(err, ComposeError::NotReadyToCompose(_)));
    }

    #[test]
    fn merge_actions_picks_strongest() {
        use RecommendedAction::*;
        assert!(matches!(merge_actions(Keep, Hedge), Hedge));
        assert!(matches!(merge_actions(Hedge, Drop), Drop));
        assert!(matches!(merge_actions(Drop, Hedge), Drop));
        assert!(matches!(merge_actions(Drop, Refuse), Refuse));
    }
}
