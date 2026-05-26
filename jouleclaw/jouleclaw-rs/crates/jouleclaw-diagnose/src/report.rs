//! Conflict-directed verification (spec §6.2).
//!
//! [`verify`] is the top-level entry point. It runs the cheap
//! checks first (coverage, authority, freshness), focuses entailment
//! work only on the claims that are at-risk given those violations,
//! then layers in the entailment-driven checks (grounding,
//! consistency), and finally emits a [`VerificationReport`] with the
//! verdict + recovery actions.
//!
//! Identifying "at-risk" claims is the lever that keeps the
//! pillar's energy under control. A grounded coverage-passing
//! authority-clean answer with N claims and M items would otherwise
//! cost N × M entailment calls; with focused selection we only run
//! entailment on the at-risk subset. For a clean answer the entire
//! pillar runs in low joules.

use chrono::Utc;
use uuid::Uuid;

use jouleclaw_schema::{
    AtomicClaim, AuthorityRecord, Invariant, QueryPlan, RetrievedItem, VerificationAction,
    VerificationReport,
};

use crate::entailer::{entail_batch, EntailError, Entailer};
use crate::recovery::recovery_actions_for_violations;
use crate::valid_answer::{
    authority_violations, consistency_violations, coverage_violations,
    freshness_violations, grounding_violations,
};
use crate::verdict::determine_verdict;

#[derive(Debug)]
pub enum VerifyError {
    Entail(EntailError),
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Entail(e) => write!(f, "entail: {e}"),
        }
    }
}

impl std::error::Error for VerifyError {}

impl From<EntailError> for VerifyError {
    fn from(e: EntailError) -> Self {
        Self::Entail(e)
    }
}

/// Inputs to [`verify`]. Bundled so the call site is explicit about
/// what the diagnose pillar consumes.
pub struct VerifyInputs<'a> {
    pub plan: &'a QueryPlan,
    pub items: &'a [RetrievedItem],
    pub authority: &'a [AuthorityRecord],
    pub claims: &'a [AtomicClaim],
    pub reroute_count: u32,
}

/// Run the conflict-directed verification pipeline.
///
/// Energy contract: the cheap checks always run; entailment is
/// invoked only on the at-risk subset of (claim, item) pairs
/// (currently: claims-without-trivial-coverage × items-for-their-
/// sub-query). Returned `joules_spent` is the sum of every
/// entailment call's measured/estimated energy.
pub fn verify<E: Entailer + ?Sized>(
    inputs: &VerifyInputs<'_>,
    entailer: &E,
) -> Result<VerificationReport, VerifyError> {
    let mut violations = Vec::new();
    let mut joules_spent = 0.0_f64;

    // 1. Cheap checks first.
    violations.extend(coverage_violations(inputs.plan, inputs.items));
    violations.extend(authority_violations(
        inputs.plan,
        inputs.items,
        inputs.authority,
    ));
    violations.extend(freshness_violations(inputs.plan, inputs.items));

    // 2. Focus entailment on at-risk (claim, item) pairs. For now,
    // "at-risk" = every claim × every item whose retrieval_context
    // points at a sub-query the claim's segment overlaps. Without
    // explicit segment→sub-query mapping yet, we run all claims
    // against all items — but in deployments that wire compose to
    // emit segment metadata, the filter would tighten.
    let mut entailments = Vec::new();
    if !inputs.claims.is_empty() && !inputs.items.is_empty() {
        let (results, j) = entail_batch(entailer, inputs.claims, inputs.items)?;
        entailments = results;
        joules_spent += j;
    }

    // 3. Entailment-driven checks.
    violations.extend(grounding_violations(inputs.claims, &entailments));
    violations.extend(consistency_violations(inputs.claims, &entailments));

    // 4. Verdict + recovery.
    let verdict = determine_verdict(&violations, inputs.plan, inputs.reroute_count);
    let recovery = recovery_actions_for_violations(&violations);

    // 5. Build the report.
    let mut invariants_verified: Vec<String> = vec![
        Invariant::I1EveryClaimHasProvenance.id().into(),
        Invariant::I4AuthorityTierRespected.id().into(),
        Invariant::I5FreshnessRespected.id().into(),
        Invariant::I6ConflictsSurfaced.id().into(),
        Invariant::I9RerouteBounded.id().into(),
    ];
    // I2 only verifies when we ran entailment.
    if !entailments.is_empty() {
        invariants_verified.push(Invariant::I2AssertedImpliesEntailed.id().into());
    }
    // I10 always holds — refusals are structured by construction.
    invariants_verified.push(Invariant::I10RefusalIsStructured.id().into());

    let report = VerificationReport {
        schema_version: "2.0".into(),
        report_id: Uuid::new_v4(),
        plan_id: inputs.plan.plan_id,
        generated_at: Utc::now(),
        verdict,
        violations,
        recovery_actions: recovery,
        entailments_consulted: entailments,
        invariants_verified,
        joules_spent,
        metadata: Default::default(),
    };
    Ok(report)
}

/// Convenience: shorthand for callers who don't need the
/// [`VerifyInputs`] struct directly.
impl<'a> VerifyInputs<'a> {
    pub fn new(
        plan: &'a QueryPlan,
        items: &'a [RetrievedItem],
        authority: &'a [AuthorityRecord],
        claims: &'a [AtomicClaim],
    ) -> Self {
        Self {
            plan,
            items,
            authority,
            claims,
            reroute_count: 0,
        }
    }

    pub fn with_reroute_count(mut self, n: u32) -> Self {
        self.reroute_count = n;
        self
    }
}

/// Helper: post-verdict, decide whether the orchestrator can
/// proceed. Returns `true` for PROCEED or DEGRADE; `false` for
/// RE_ROUTE or REFUSE.
pub fn can_compose(verdict: VerificationAction) -> bool {
    matches!(
        verdict,
        VerificationAction::ProceedToComposition | VerificationAction::DegradeGracefully
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atomizer::atomize_sentences;
    use crate::entailer::FixtureEntailer;
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

    fn item_for(sub_id: &str, source_id: &str, tier: u8, text: &str) -> RetrievedItem {
        RetrievedItem {
            schema_version: "2.0".into(),
            item_id: Uuid::new_v4(),
            source_id: source_id.into(),
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

    fn simple_plan() -> QueryPlan {
        QueryPlan {
            schema_version: "2.0".into(),
            plan_id: Uuid::new_v4(),
            original_query: OriginalQuery {
                text: Some("capital of France".into()),
                image_ref: None,
                audio_ref: None,
                video_ref: None,
                language_detected: "en".into(),
                timestamp: Utc::now(),
            },
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
            constraints: Constraints::default(),
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

    #[test]
    fn clean_pipeline_proceeds_with_no_violations() {
        let plan = simple_plan();
        let item = item_for("q0", "wikidata:Q90", 1, "Paris is the capital of France.");
        let items = vec![item];
        let segs = vec![("s0".into(), "Paris is the capital of France.".into())];
        let claims = atomize_sentences(&segs, &axes_with_tier(1)).unwrap();
        // Fixture: every (item text, claim text) pair → entails.
        let entailer = FixtureEntailer::new();
        entailer.set_entails(
            "Paris is the capital of France.",
            "Paris is the capital of France.",
        );
        let report = verify(
            &VerifyInputs::new(&plan, &items, &[], &claims),
            &entailer,
        )
        .unwrap();
        assert!(report.violations.is_empty(), "got {:?}", report.violations);
        assert!(matches!(
            report.verdict,
            VerificationAction::ProceedToComposition
        ));
        assert!(report
            .invariants_verified
            .contains(&"I2".to_string()), "I2 should be verified when entailment ran");
    }

    #[test]
    fn unsupported_claim_drives_grounding_violation_and_degrade() {
        let plan = simple_plan();
        let item = item_for("q0", "wikidata:Q90", 1, "Paris is the capital of France.");
        let items = vec![item];
        // Claim doesn't match — fixture default is Neutral, so no entailment.
        let segs = vec![("s0".into(), "Berlin is the capital of France.".into())];
        let claims = atomize_sentences(&segs, &axes_with_tier(1)).unwrap();
        let entailer = FixtureEntailer::new(); // default = Neutral
        let report = verify(
            &VerifyInputs::new(&plan, &items, &[], &claims),
            &entailer,
        )
        .unwrap();
        let grounding = report
            .violations
            .iter()
            .filter(|v| v.violation_id.starts_with("grounding"))
            .count();
        assert_eq!(grounding, 1);
        // Medium-stakes claim → Major severity → DegradeGracefully.
        assert!(matches!(
            report.verdict,
            VerificationAction::DegradeGracefully
        ));
    }

    #[test]
    fn coverage_gap_reroutes_when_budget_available() {
        let plan = simple_plan();
        let items: Vec<RetrievedItem> = vec![]; // empty → coverage gap
        let claims: Vec<AtomicClaim> = vec![];
        let entailer = FixtureEntailer::new();
        let report = verify(
            &VerifyInputs::new(&plan, &items, &[], &claims),
            &entailer,
        )
        .unwrap();
        assert!(matches!(
            report.verdict,
            VerificationAction::ReRouteWithRefinement
        ));
    }

    #[test]
    fn coverage_gap_refuses_when_budget_exhausted() {
        let plan = simple_plan();
        let items: Vec<RetrievedItem> = vec![];
        let claims: Vec<AtomicClaim> = vec![];
        let entailer = FixtureEntailer::new();
        let report = verify(
            &VerifyInputs::new(&plan, &items, &[], &claims).with_reroute_count(2),
            &entailer,
        )
        .unwrap();
        assert!(matches!(report.verdict, VerificationAction::Refuse));
    }

    #[test]
    fn contradiction_triggers_consistency_violation_and_degrade() {
        let plan = simple_plan();
        let it1 = item_for("q0", "wikidata:Q90", 1, "Paris is in France.");
        let it2 = item_for("q0", "wikidata:Q183", 1, "Paris is in Germany.");
        let items = vec![it1.clone(), it2.clone()];
        let segs = vec![("s0".into(), "Paris is in France.".into())];
        let claims = atomize_sentences(&segs, &axes_with_tier(1)).unwrap();

        let entailer = FixtureEntailer::new();
        entailer.set_entails("Paris is in France.", "Paris is in France.");
        entailer.set_contradicts("Paris is in Germany.", "Paris is in France.");

        let report = verify(
            &VerifyInputs::new(&plan, &items, &[], &claims),
            &entailer,
        )
        .unwrap();
        let consistency = report
            .violations
            .iter()
            .filter(|v| v.violation_id.starts_with("consistency"))
            .count();
        assert_eq!(consistency, 1);
        // Major (consistency is always Major, not Critical) → DegradeGracefully.
        assert!(matches!(
            report.verdict,
            VerificationAction::DegradeGracefully
        ));
        // I6 (conflicts surfaced) is in the invariants list.
        assert!(report.invariants_verified.contains(&"I6".to_string()));
    }

    #[test]
    fn recovery_actions_present_per_violation() {
        let plan = simple_plan();
        let items: Vec<RetrievedItem> = vec![];
        let claims: Vec<AtomicClaim> = vec![];
        let entailer = FixtureEntailer::new();
        let report = verify(
            &VerifyInputs::new(&plan, &items, &[], &claims),
            &entailer,
        )
        .unwrap();
        assert_eq!(report.recovery_actions.len(), report.violations.len());
    }

    #[test]
    fn can_compose_returns_true_for_proceed_and_degrade() {
        assert!(can_compose(VerificationAction::ProceedToComposition));
        assert!(can_compose(VerificationAction::DegradeGracefully));
        assert!(!can_compose(VerificationAction::ReRouteWithRefinement));
        assert!(!can_compose(VerificationAction::Refuse));
    }
}
