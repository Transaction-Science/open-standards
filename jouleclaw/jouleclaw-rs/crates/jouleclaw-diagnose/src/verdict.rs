//! Verdict determination (spec §6.6).
//!
//! Four possible verdicts:
//!
//! - `ProceedToComposition` — no violations.
//! - `ReRouteWithRefinement` — at least one critical violation but
//!   all critical violations are retrievable AND the re-route
//!   budget hasn't been exhausted.
//! - `Refuse` — at least one critical violation isn't retrievable
//!   (e.g., the corpus genuinely lacks an authoritative source).
//! - `DegradeGracefully` — only non-critical violations remain;
//!   compose with caveats.

use jouleclaw_schema::{QueryPlan, VerificationAction, Violation, ViolationSeverity};

use crate::valid_answer::is_retrievable;

pub fn determine_verdict(
    violations: &[Violation],
    plan: &QueryPlan,
    reroute_count: u32,
) -> VerificationAction {
    if violations.is_empty() {
        return VerificationAction::ProceedToComposition;
    }

    let critical: Vec<&Violation> = violations
        .iter()
        .filter(|v| matches!(v.severity, ViolationSeverity::Critical))
        .collect();

    if !critical.is_empty() {
        let any_not_retrievable = critical.iter().any(|v| !is_retrievable(v));
        if any_not_retrievable {
            return VerificationAction::Refuse;
        }
        if reroute_count < plan.budget.max_reroute_iterations {
            return VerificationAction::ReRouteWithRefinement;
        }
        // Critical violations remain but the re-route budget is
        // gone — fall through to degrade or refuse based on whether
        // they're retrievable in principle. Spec §6.6 doesn't define
        // this corner; choosing REFUSE matches the conservative
        // reading of "we couldn't fix this within budget".
        return VerificationAction::Refuse;
    }

    VerificationAction::DegradeGracefully
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use uuid::Uuid;

    fn plan() -> QueryPlan {
        QueryPlan {
            schema_version: "2.0".into(),
            plan_id: Uuid::new_v4(),
            original_query: jouleclaw_schema::OriginalQuery {
                text: Some("q".into()),
                image_ref: None,
                audio_ref: None,
                video_ref: None,
                language_detected: "en".into(),
                timestamp: chrono::Utc::now(),
            },
            intent: jouleclaw_schema::Intent::Lookup,
            modalities_in: vec![jouleclaw_schema::Modality::Text],
            modalities_out: vec![jouleclaw_schema::Modality::Text],
            decomposition: vec![],
            constraints: Default::default(),
            budget: Default::default(),
            invariants_satisfied: jouleclaw_schema::PlanInvariants {
                all_subqueries_have_at_least_one_store: true,
                dependency_graph_is_acyclic: true,
                total_estimated_latency_within_budget: true,
                estimated_cost_within_budget: true,
                required_modalities_covered: true,
            },
            metadata: Default::default(),
        }
    }

    fn v(severity: ViolationSeverity, retrievable: bool) -> Violation {
        Violation {
            violation_id: "v".into(),
            message: "m".into(),
            severity,
            claim_id: None,
            item_id: None,
            detail: json!({ "retrievable": retrievable }),
        }
    }

    #[test]
    fn no_violations_proceeds() {
        let p = plan();
        assert!(matches!(
            determine_verdict(&[], &p, 0),
            VerificationAction::ProceedToComposition
        ));
    }

    #[test]
    fn critical_retrievable_under_budget_reroutes() {
        let p = plan();
        let vs = vec![v(ViolationSeverity::Critical, true)];
        assert!(matches!(
            determine_verdict(&vs, &p, 0),
            VerificationAction::ReRouteWithRefinement
        ));
    }

    #[test]
    fn critical_not_retrievable_refuses() {
        let p = plan();
        let vs = vec![v(ViolationSeverity::Critical, false)];
        assert!(matches!(
            determine_verdict(&vs, &p, 0),
            VerificationAction::Refuse
        ));
    }

    #[test]
    fn critical_retrievable_but_budget_exhausted_refuses() {
        let p = plan();
        let vs = vec![v(ViolationSeverity::Critical, true)];
        // p.budget.max_reroute_iterations defaults to 2; pass 2 → exhausted.
        assert!(matches!(
            determine_verdict(&vs, &p, 2),
            VerificationAction::Refuse
        ));
    }

    #[test]
    fn only_minor_or_major_degrades() {
        let p = plan();
        let vs = vec![
            v(ViolationSeverity::Major, true),
            v(ViolationSeverity::Minor, true),
        ];
        assert!(matches!(
            determine_verdict(&vs, &p, 0),
            VerificationAction::DegradeGracefully
        ));
    }
}
