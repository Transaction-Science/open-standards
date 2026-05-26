//! Verification report (spec §3.4 deferred to v1, §6.1–§6.6).
//!
//! The Diagnose pillar produces a [`VerificationReport`] that names
//! every violation found against the ValidAnswerModel (§6.1),
//! attaches a [`RecoveryAction`] per violation (§6.5), and concludes
//! with one of four verdicts (§6.6). The orchestrator branches on the
//! verdict.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::common::Metadata;
use crate::entailment::EntailmentResult;

/// Per-violation severity. Drives the verdict logic in §6.6:
/// any `Critical` violation triggers RE_ROUTE or REFUSE depending
/// on whether more retrieval could fix it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ViolationSeverity {
    Critical,
    Major,
    Minor,
}

/// One named failure against the ValidAnswerModel. Examples from
/// §6.1: `sub_query_3_uncovered`, `claim_7_unsupported`,
/// `authority_tier_2_required_only_tier_3_found`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Violation {
    pub violation_id: String,
    /// Human-readable explanation; user-facing on REFUSE / DEGRADE.
    pub message: String,
    pub severity: ViolationSeverity,
    /// If the violation is tied to a specific atomic claim.
    #[serde(default)]
    pub claim_id: Option<Uuid>,
    /// If the violation is tied to a specific retrieved item.
    #[serde(default)]
    pub item_id: Option<Uuid>,
    /// Optional structured detail (e.g. `{"required_tier": 2,
    /// "found_tier": 3}`).
    #[serde(default)]
    pub detail: serde_json::Value,
}

/// The atomic operations the verified composer applies to a draft.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendedAction {
    /// Keep the claim as-is.
    Keep,
    /// Add hedging language ("appears to", "is widely understood to").
    Hedge,
    /// Drop the claim from the answer.
    Drop,
    /// Replace with the supplied `alternative_text` from
    /// [`RecoveryAction`].
    Rewrite,
    /// Surface as an inference labeled as such, with explicit
    /// premises.
    LabelAsInference,
    /// Surface the conflict to the user rather than silently picking.
    SurfaceConflict,
    /// Don't compose; refuse the whole answer.
    Refuse,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveryAction {
    pub violation_id: String,
    pub action_type: RecommendedAction,
    #[serde(default)]
    pub target_claim_id: Option<Uuid>,
    pub rationale: String,
    /// Used when `action_type == Rewrite`.
    #[serde(default)]
    pub alternative_text: Option<String>,
}

/// The four verdicts from §6.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationAction {
    ProceedToComposition,
    ReRouteWithRefinement,
    DegradeGracefully,
    Refuse,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationReport {
    pub schema_version: String,
    pub report_id: Uuid,
    /// The plan this report verifies an answer to.
    pub plan_id: Uuid,
    pub generated_at: DateTime<Utc>,
    pub verdict: VerificationAction,
    pub violations: Vec<Violation>,
    pub recovery_actions: Vec<RecoveryAction>,
    /// The entailment results the verifier consulted (focused subset,
    /// not exhaustive — see §6.2).
    pub entailments_consulted: Vec<EntailmentResult>,
    /// Invariant ids (e.g. `"I1"`, `"I13"`) that were checked and
    /// passed. The orchestrator refuses to emit any Answer whose
    /// required invariants aren't in this list.
    pub invariants_verified: Vec<String>,
    /// Joules spent during verification, contributing to §8.5.
    pub joules_spent: f64,
    #[serde(default)]
    pub metadata: Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proceed_with_no_violations_is_well_formed() {
        let r = VerificationReport {
            schema_version: "2.0".into(),
            report_id: Uuid::new_v4(),
            plan_id: Uuid::new_v4(),
            generated_at: Utc::now(),
            verdict: VerificationAction::ProceedToComposition,
            violations: vec![],
            recovery_actions: vec![],
            entailments_consulted: vec![],
            invariants_verified: vec!["I1".into(), "I2".into(), "I13".into()],
            joules_spent: 0.0,
            metadata: Default::default(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: VerificationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn violation_severity_serialize_lowercase() {
        let v = Violation {
            violation_id: "v1".into(),
            message: "missing".into(),
            severity: ViolationSeverity::Critical,
            claim_id: None,
            item_id: None,
            detail: serde_json::Value::Null,
        };
        let json = serde_json::to_string(&v).unwrap();
        assert!(json.contains("\"severity\":\"critical\""));
    }
}
