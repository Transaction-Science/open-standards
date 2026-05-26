//! Recovery action synthesis (spec §6.5).
//!
//! For each violation, name the operation the verified composer
//! should apply to its draft. The mapping is mechanical and
//! deterministic:
//!
//! - **grounding** (unsupported claim) → `Hedge` if Minor/Major,
//!   `Drop` if Critical and stakes are High,
//!   `LabelAsInference` if Major.
//! - **consistency** (contradicting sources) → `SurfaceConflict`.
//! - **coverage** (sub-query unanswered) → `Keep`; the composer
//!   doesn't have a draft segment to act on, the orchestrator
//!   already re-routes.
//! - **authority** / **freshness** — `Hedge` (compose with explicit
//!   caveat); the verdict layer is responsible for the more
//!   aggressive RE_ROUTE / REFUSE decisions.

use jouleclaw_schema::{RecommendedAction, RecoveryAction, Violation, ViolationSeverity};

pub fn recovery_actions_for_violations(violations: &[Violation]) -> Vec<RecoveryAction> {
    violations.iter().map(action_for).collect()
}

fn action_for(v: &Violation) -> RecoveryAction {
    let kind = v
        .detail
        .get("kind")
        .and_then(|x| x.as_str())
        .unwrap_or("");
    let (action_type, rationale) = match kind {
        "grounding" => grounding_action(v),
        "consistency" => (
            RecommendedAction::SurfaceConflict,
            "contradictory sources — surface the disagreement instead of picking one"
                .to_string(),
        ),
        "coverage" => (
            RecommendedAction::Keep,
            "no draft segment is affected directly; orchestrator handles re-routing"
                .to_string(),
        ),
        "authority" => (
            RecommendedAction::Hedge,
            "best available source is below the required authority tier".to_string(),
        ),
        "freshness" => (
            RecommendedAction::Hedge,
            "best available source exceeds its freshness budget".to_string(),
        ),
        _ => (
            RecommendedAction::Keep,
            "unknown violation kind; defaulting to keep".to_string(),
        ),
    };
    RecoveryAction {
        violation_id: v.violation_id.clone(),
        action_type,
        target_claim_id: v.claim_id,
        rationale,
        alternative_text: None,
    }
}

fn grounding_action(v: &Violation) -> (RecommendedAction, String) {
    let stakes = v.detail.get("stakes").and_then(|x| x.as_str()).unwrap_or("");
    match (v.severity, stakes) {
        (ViolationSeverity::Critical, "high") => (
            RecommendedAction::Drop,
            "critical unsupported claim with high stakes — drop from the answer".into(),
        ),
        (ViolationSeverity::Critical, "critical") => (
            RecommendedAction::Drop,
            "safety/compliance-critical claim has no entailing source — drop".into(),
        ),
        (ViolationSeverity::Critical, _) => (
            RecommendedAction::Drop,
            "critical unsupported claim — drop from the answer".into(),
        ),
        (ViolationSeverity::Major, _) => (
            RecommendedAction::LabelAsInference,
            "no direct entailment; label this claim as inference".into(),
        ),
        (ViolationSeverity::Minor, _) => (
            RecommendedAction::Hedge,
            "weakly-supported claim — hedge the phrasing".into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use uuid::Uuid;

    fn v(violation_id: &str, severity: ViolationSeverity, kind: &str, stakes: &str) -> Violation {
        Violation {
            violation_id: violation_id.into(),
            message: "m".into(),
            severity,
            claim_id: Some(Uuid::new_v4()),
            item_id: None,
            detail: json!({ "kind": kind, "stakes": stakes }),
        }
    }

    #[test]
    fn grounding_critical_high_stakes_drops() {
        let action = action_for(&v("g1", ViolationSeverity::Critical, "grounding", "high"));
        assert!(matches!(action.action_type, RecommendedAction::Drop));
    }

    #[test]
    fn grounding_major_labels_as_inference() {
        let action = action_for(&v("g1", ViolationSeverity::Major, "grounding", "medium"));
        assert!(matches!(action.action_type, RecommendedAction::LabelAsInference));
    }

    #[test]
    fn grounding_minor_hedges() {
        let action = action_for(&v("g1", ViolationSeverity::Minor, "grounding", "low"));
        assert!(matches!(action.action_type, RecommendedAction::Hedge));
    }

    #[test]
    fn consistency_surfaces_conflict() {
        let action = action_for(&v("c1", ViolationSeverity::Major, "consistency", ""));
        assert!(matches!(action.action_type, RecommendedAction::SurfaceConflict));
    }

    #[test]
    fn authority_hedges() {
        let action = action_for(&v("a1", ViolationSeverity::Critical, "authority", ""));
        assert!(matches!(action.action_type, RecommendedAction::Hedge));
    }

    #[test]
    fn coverage_keeps_for_orchestrator() {
        let action = action_for(&v("cov1", ViolationSeverity::Critical, "coverage", ""));
        assert!(matches!(action.action_type, RecommendedAction::Keep));
    }
}
