//! The output of the Plan Pillar (spec §3.2).
//!
//! A [`QueryPlan`] is constraint-validated by construction. Its
//! `invariants_satisfied` field must be all-true before the plan can
//! be executed; the constraint solver guarantees this in §4.2.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::common::Metadata;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Intent {
    Lookup,
    Aggregation,
    Comparison,
    Recommendation,
    Generation,
    Action,
    Clarification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    Structured,
    Chart,
    Table,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OriginalQuery {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_ref: Option<String>,
    #[serde(default)]
    pub audio_ref: Option<String>,
    #[serde(default)]
    pub video_ref: Option<String>,
    pub language_detected: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubQuery {
    pub sub_id: String,
    pub text: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    pub required_modalities: Vec<Modality>,
    /// Constraint solver fills this — which retrievers are allowed to
    /// answer this sub-query.
    pub target_stores: Vec<String>,
    /// 0.0..=1.0.
    pub priority: f64,
    /// Which RAP id applies (`ReactiveActionPackage::rap_id`).
    pub rap_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalScope {
    #[serde(default)]
    pub start: Option<DateTime<Utc>>,
    #[serde(default)]
    pub end: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Constraints {
    #[serde(default)]
    pub freshness_required: bool,
    #[serde(default)]
    pub freshness_max_age_days: Option<u32>,
    /// 1..=4. Default 3.
    pub minimum_authority_tier: u8,
    #[serde(default)]
    pub geographic_scope: Option<Vec<String>>,
    #[serde(default)]
    pub temporal_scope: Option<TemporalScope>,
    #[serde(default)]
    pub language_constraint: Option<Vec<String>>,
}

impl Default for Constraints {
    fn default() -> Self {
        Self {
            freshness_required: false,
            freshness_max_age_days: None,
            minimum_authority_tier: 3,
            geographic_scope: None,
            temporal_scope: None,
            language_constraint: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    pub latency_target_ms: u32,
    /// Hard cutoff — once breached the orchestrator stops adding work.
    pub latency_hard_ceiling_ms: u32,
    pub cost_ceiling_usd: f64,
    /// Thermodynamic budget added in v3 — joules-per-query first-class.
    pub energy_ceiling_joules: f64,
    pub max_stores_to_query: u32,
    /// Bounded re-route iterations (§8.3); spec default 2.
    pub max_reroute_iterations: u32,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            latency_target_ms: 3000,
            latency_hard_ceiling_ms: 5000,
            cost_ceiling_usd: 0.10,
            energy_ceiling_joules: 100.0,
            max_stores_to_query: 6,
            max_reroute_iterations: 2,
        }
    }
}

/// All five must be `true` before a [`QueryPlan`] is valid for
/// execution. The constraint solver enforces this; plans where any
/// invariant is false must be rejected at plan time, not execution
/// time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanInvariants {
    pub all_subqueries_have_at_least_one_store: bool,
    pub dependency_graph_is_acyclic: bool,
    pub total_estimated_latency_within_budget: bool,
    pub estimated_cost_within_budget: bool,
    pub required_modalities_covered: bool,
}

impl PlanInvariants {
    pub fn all_satisfied(&self) -> bool {
        self.all_subqueries_have_at_least_one_store
            && self.dependency_graph_is_acyclic
            && self.total_estimated_latency_within_budget
            && self.estimated_cost_within_budget
            && self.required_modalities_covered
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryPlan {
    pub schema_version: String,
    pub plan_id: Uuid,
    pub original_query: OriginalQuery,
    pub intent: Intent,
    pub modalities_in: Vec<Modality>,
    pub modalities_out: Vec<Modality>,
    pub decomposition: Vec<SubQuery>,
    pub constraints: Constraints,
    pub budget: Budget,
    /// Must be all true for the plan to be valid (spec §3.2).
    pub invariants_satisfied: PlanInvariants,
    #[serde(default)]
    pub metadata: Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_plan() -> QueryPlan {
        QueryPlan {
            schema_version: "2.0".into(),
            plan_id: Uuid::new_v4(),
            original_query: OriginalQuery {
                text: Some("the capital of France".into()),
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
                rap_id: "wikidata_sparql_v1".into(),
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
    fn all_satisfied_helper_matches_per_field_truth() {
        let p = minimal_plan();
        assert!(p.invariants_satisfied.all_satisfied());
        let mut bad = p.invariants_satisfied.clone();
        bad.required_modalities_covered = false;
        assert!(!bad.all_satisfied());
    }

    #[test]
    fn roundtrips_through_json() {
        let p = minimal_plan();
        let json = serde_json::to_string(&p).unwrap();
        let back: QueryPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn default_budget_matches_spec() {
        let b = Budget::default();
        assert_eq!(b.latency_target_ms, 3000);
        assert_eq!(b.latency_hard_ceiling_ms, 5000);
        assert_eq!(b.energy_ceiling_joules, 100.0);
        assert_eq!(b.max_reroute_iterations, 2);
    }
}
