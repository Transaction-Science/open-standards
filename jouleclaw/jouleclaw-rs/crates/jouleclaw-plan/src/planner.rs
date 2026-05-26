//! Constraint Planner (spec §4.2).
//!
//! Takes a [`QueryAnalysis`] and a [`SystemCapabilities`] snapshot
//! and produces a constraint-validated [`QueryPlan`]. The output's
//! `invariants_satisfied` is checked once more after the CSP returns
//! a solution and asserted before the plan is handed downstream.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Duration;

use chrono::Utc;
use uuid::Uuid;

use crate::analysis::{QueryAnalysis, RawSubQuery};
use crate::csp::{Assignment, CspError, CspSolver};
use jouleclaw_schema::{
    Budget, CapabilityStatus, Constraints, Modality, PlanInvariants, QueryPlan, ReactiveActionPackage,
    RetrievalMethod, RetrieverCapability, SubQuery, SystemCapabilities,
};

/// Per-store cost/method/RAP profile the planner consults during CSP
/// search. Built once from a retriever registry; references match
/// `RetrieverCapability.retriever_id`.
#[derive(Debug, Clone)]
pub struct RetrieverProfile {
    pub retriever_id: String,
    pub default_rap_id: String,
    pub retrieval_method: RetrievalMethod,
    /// Static latency estimate per call (ms).
    pub estimated_latency_ms: u32,
    /// Static cost estimate per call (USD).
    pub estimated_cost_usd: f64,
    /// Static energy estimate per call (joules).
    pub estimated_joules: f64,
}

/// Registry of available retriever profiles + RAPs. Built by the
/// orchestrator at start-up; passed read-only to the planner.
#[derive(Debug, Clone, Default)]
pub struct StoreCatalog {
    pub profiles: HashMap<String, RetrieverProfile>,
    pub raps: HashMap<String, ReactiveActionPackage>,
}

impl StoreCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, profile: RetrieverProfile) {
        self.profiles
            .insert(profile.retriever_id.clone(), profile);
    }

    pub fn add_rap(&mut self, rap: ReactiveActionPackage) {
        self.raps.insert(rap.rap_id.clone(), rap);
    }
}

#[derive(Debug)]
pub enum PlanError {
    Csp(CspError),
    NoCandidateStores {
        sub_id: String,
        wanted_modalities: Vec<Modality>,
    },
    /// A constraint that wasn't expressible to the CSP failed
    /// post-solve. Indicates a bug in the planner's invariant
    /// checking or in the profile data; treat as an internal error.
    InvariantBroken(String),
    CycleInDependencies,
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Csp(e) => write!(f, "csp: {e}"),
            Self::NoCandidateStores {
                sub_id,
                wanted_modalities,
            } => write!(
                f,
                "sub-query {sub_id}: no candidate stores cover modalities {wanted_modalities:?}"
            ),
            Self::InvariantBroken(s) => write!(f, "invariant broken: {s}"),
            Self::CycleInDependencies => write!(f, "dependency graph has a cycle"),
        }
    }
}

impl std::error::Error for PlanError {}

impl From<CspError> for PlanError {
    fn from(e: CspError) -> Self {
        Self::Csp(e)
    }
}

/// Produce a constraint-validated [`QueryPlan`].
///
/// The CSP variable per sub-query is its assigned store id; values
/// are the store ids that satisfy the cheap filters (modality
/// coverage + capability status ≠ Unavailable). The full constraint
/// then verifies dependency acyclicity, budget bounds, authority
/// tier requirement, and freshness requirement.
pub fn plan(
    analysis: &QueryAnalysis,
    capabilities: &SystemCapabilities,
    catalog: &StoreCatalog,
    constraints: Constraints,
    budget: Budget,
) -> Result<QueryPlan, PlanError> {
    if has_cycle(&analysis.raw_decomposition) {
        return Err(PlanError::CycleInDependencies);
    }

    // Compute candidate-stores per sub-query: those that cover the
    // required modalities AND aren't UNAVAILABLE in capabilities.
    let mut candidates: HashMap<String, Vec<String>> = HashMap::new();
    let healthy_retrievers: HashMap<String, &RetrieverCapability> = capabilities
        .retrievers
        .iter()
        .filter(|r| !matches!(r.status, CapabilityStatus::Unavailable))
        .map(|r| (r.retriever_id.clone(), r))
        .collect();

    for sq in &analysis.raw_decomposition {
        let mut stores: Vec<String> = healthy_retrievers
            .iter()
            .filter(|(_, cap)| {
                sq.required_modalities
                    .iter()
                    .all(|m| cap.modalities_supported.contains(m))
            })
            .filter(|(id, _)| catalog.profiles.contains_key(id.as_str()))
            .map(|(id, _)| id.clone())
            .collect();
        // Honor `preferred_store` hints — if the user told us
        // which retriever to use and it's still in the candidate
        // pool, prune to just that one. Lets the analyzer direct
        // parallel sub-queries to specific retrievers without
        // re-implementing the CSP.
        if let Some(preferred) = &sq.preferred_store {
            if stores.iter().any(|s| s == preferred) {
                stores.retain(|s| s == preferred);
            } else {
                // The hint pointed at a store that's either
                // unhealthy or absent — treat the same as
                // "no candidates" so the caller gets a clear
                // refusal rather than silent fallback.
                return Err(PlanError::NoCandidateStores {
                    sub_id: sq.sub_id.clone(),
                    wanted_modalities: sq.required_modalities.clone(),
                });
            }
        }
        stores.sort();
        if stores.is_empty() {
            return Err(PlanError::NoCandidateStores {
                sub_id: sq.sub_id.clone(),
                wanted_modalities: sq.required_modalities.clone(),
            });
        }
        candidates.insert(sq.sub_id.clone(), stores);
    }

    let constraints_for_check = constraints.clone();
    let budget_for_check = budget.clone();
    let catalog_for_solve = catalog.clone();
    let retrievers_for_solve: HashMap<String, RetrieverCapability> = healthy_retrievers
        .iter()
        .map(|(k, v)| (k.clone(), (*v).clone()))
        .collect();

    let solver_constraint = move |a: &Assignment<String, String>| -> bool {
        // Authority tier: when minimum_authority_tier <= 2, at least
        // one assigned store must have authority_tier <= 2.
        if constraints_for_check.minimum_authority_tier <= 2 {
            let any_tier_ok = a.values().any(|store| {
                retrievers_for_solve
                    .get(store)
                    .map(|r| r.authority_tier <= 2)
                    .unwrap_or(false)
            });
            if !any_tier_ok && a.len() == candidates_len_helper(&a) {
                // Partial assignment can still satisfy this once more
                // variables get assigned; only enforce on completion.
                // We approximate completion by checking that no
                // remaining slot can supply a tier-1-or-2 store: if
                // even the partial assignment has used all sub-ids,
                // fail; otherwise allow.
                // For simplicity here we only enforce at the full
                // assignment level via the post-solve invariant
                // check below.
            }
        }
        // Budget: total estimated latency must be within target.
        let total_latency_ms: u32 = a
            .values()
            .filter_map(|store| {
                catalog_for_solve
                    .profiles
                    .get(store)
                    .map(|p| p.estimated_latency_ms)
            })
            .sum();
        if total_latency_ms > budget_for_check.latency_hard_ceiling_ms {
            return false;
        }
        // Cost ceiling.
        let total_cost: f64 = a
            .values()
            .filter_map(|store| {
                catalog_for_solve
                    .profiles
                    .get(store)
                    .map(|p| p.estimated_cost_usd)
            })
            .sum();
        if total_cost > budget_for_check.cost_ceiling_usd {
            return false;
        }
        // Energy ceiling.
        let total_joules: f64 = a
            .values()
            .filter_map(|store| {
                catalog_for_solve
                    .profiles
                    .get(store)
                    .map(|p| p.estimated_joules)
            })
            .sum();
        if total_joules > budget_for_check.energy_ceiling_joules {
            return false;
        }
        true
    };

    let catalog_for_cost = catalog.clone();
    let unary_cost = move |_v: &String, val: &String| -> f64 {
        catalog_for_cost
            .profiles
            .get(val)
            .map(|p| p.estimated_joules)
            .unwrap_or(f64::INFINITY)
    };

    let mut solver = CspSolver::new(solver_constraint, unary_cost)
        .timeout(Duration::from_millis(200));
    for sq in &analysis.raw_decomposition {
        let domain = candidates
            .get(&sq.sub_id)
            .cloned()
            .unwrap_or_default();
        solver = solver.variable(sq.sub_id.clone(), domain);
    }
    let (assignment, _score) = solver.solve()?;

    // Authority tier full check on the completed assignment.
    if constraints.minimum_authority_tier <= 2 {
        let any_tier_ok = assignment.values().any(|store| {
            healthy_retrievers
                .get(store)
                .map(|r| r.authority_tier <= 2)
                .unwrap_or(false)
        });
        if !any_tier_ok {
            return Err(PlanError::InvariantBroken(
                "minimum_authority_tier <= 2 but no assigned store meets it".into(),
            ));
        }
    }

    // Freshness: if required, at least one assigned store must be
    // LIVE-capable. We approximate this with the retriever having a
    // `live_feed` domain tag — the catalog doesn't currently model
    // freshness explicitly, so we accept any healthy store and let
    // the orchestrator's per-item freshness check catch staleness.
    if constraints.freshness_required {
        let any_live = assignment.values().any(|store| {
            healthy_retrievers
                .get(store)
                .map(|r| r.domains_covered.iter().any(|d| d == "live_feed"))
                .unwrap_or(false)
        });
        if !any_live {
            return Err(PlanError::InvariantBroken(
                "freshness_required but no live retriever assigned".into(),
            ));
        }
    }

    // Build the SubQuery list.
    let decomposition: Vec<SubQuery> = analysis
        .raw_decomposition
        .iter()
        .map(|sq| {
            let store = assignment.get(&sq.sub_id).cloned().unwrap_or_default();
            let rap_id = catalog
                .profiles
                .get(&store)
                .map(|p| p.default_rap_id.clone())
                .unwrap_or_else(|| "unknown".into());
            SubQuery {
                sub_id: sq.sub_id.clone(),
                text: sq.text.clone(),
                depends_on: sq.depends_on.clone(),
                required_modalities: sq.required_modalities.clone(),
                target_stores: vec![store],
                priority: sq.priority,
                rap_id,
            }
        })
        .collect();

    let invariants = check_invariants(&decomposition, &budget, catalog);
    if !invariants.all_satisfied() {
        return Err(PlanError::InvariantBroken(format!(
            "post-solve invariant check failed: {invariants:?}"
        )));
    }

    Ok(QueryPlan {
        schema_version: "2.0".into(),
        plan_id: Uuid::new_v4(),
        original_query: analysis.original_query.clone(),
        intent: analysis.intent,
        modalities_in: analysis.modalities_in.clone(),
        modalities_out: analysis.modalities_out.clone(),
        decomposition,
        constraints,
        budget,
        invariants_satisfied: invariants,
        metadata: {
            let mut m = serde_json::Map::new();
            m.insert(
                "planner".into(),
                serde_json::Value::String("jouleclaw-plan/0.1".into()),
            );
            m.insert(
                "planned_at".into(),
                serde_json::Value::String(Utc::now().to_rfc3339()),
            );
            m
        },
    })
}

fn check_invariants(
    decomposition: &[SubQuery],
    budget: &Budget,
    catalog: &StoreCatalog,
) -> PlanInvariants {
    let all_have_store = decomposition.iter().all(|sq| !sq.target_stores.is_empty());
    let acyclic = !has_cycle_sub(decomposition);
    let total_latency: u32 = decomposition
        .iter()
        .flat_map(|sq| sq.target_stores.iter())
        .filter_map(|s| catalog.profiles.get(s).map(|p| p.estimated_latency_ms))
        .sum();
    let total_cost: f64 = decomposition
        .iter()
        .flat_map(|sq| sq.target_stores.iter())
        .filter_map(|s| catalog.profiles.get(s).map(|p| p.estimated_cost_usd))
        .sum();
    let modalities_covered = decomposition
        .iter()
        .all(|sq| !sq.required_modalities.is_empty());
    PlanInvariants {
        all_subqueries_have_at_least_one_store: all_have_store,
        dependency_graph_is_acyclic: acyclic,
        total_estimated_latency_within_budget: total_latency <= budget.latency_hard_ceiling_ms,
        estimated_cost_within_budget: total_cost <= budget.cost_ceiling_usd,
        required_modalities_covered: modalities_covered,
    }
}

fn has_cycle(decomposition: &[RawSubQuery]) -> bool {
    let mut edges: HashMap<&str, Vec<&str>> = HashMap::new();
    for sq in decomposition {
        edges.insert(sq.sub_id.as_str(), sq.depends_on.iter().map(|s| s.as_str()).collect());
    }
    let mut visited: HashSet<&str> = HashSet::new();
    let mut stack: HashSet<&str> = HashSet::new();
    for sq in decomposition {
        if dfs_cycle(sq.sub_id.as_str(), &edges, &mut visited, &mut stack) {
            return true;
        }
    }
    false
}

fn has_cycle_sub(decomposition: &[SubQuery]) -> bool {
    let mut edges: HashMap<&str, Vec<&str>> = HashMap::new();
    for sq in decomposition {
        edges.insert(sq.sub_id.as_str(), sq.depends_on.iter().map(|s| s.as_str()).collect());
    }
    let mut visited: HashSet<&str> = HashSet::new();
    let mut stack: HashSet<&str> = HashSet::new();
    for sq in decomposition {
        if dfs_cycle(sq.sub_id.as_str(), &edges, &mut visited, &mut stack) {
            return true;
        }
    }
    false
}

fn dfs_cycle<'a>(
    node: &'a str,
    edges: &HashMap<&'a str, Vec<&'a str>>,
    visited: &mut HashSet<&'a str>,
    stack: &mut HashSet<&'a str>,
) -> bool {
    if stack.contains(node) {
        return true;
    }
    if visited.contains(node) {
        return false;
    }
    visited.insert(node);
    stack.insert(node);
    if let Some(neighbors) = edges.get(node) {
        for n in neighbors {
            if dfs_cycle(n, edges, visited, stack) {
                return true;
            }
        }
    }
    stack.remove(node);
    false
}

/// Helper for partial-assignment constraint check: how many sub-ids
/// the assignment currently covers. Not exposed as public API.
fn candidates_len_helper(a: &Assignment<String, String>) -> usize {
    let keys: BTreeSet<_> = a.keys().collect();
    keys.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{RawSubQuery, StakesSignal};
    use chrono::Utc;
    use jouleclaw_schema::{
        Intent, Modality, OriginalQuery, RapStep, RapStepCondition, ReactiveActionPackage,
        RetrievalMethod,
    };

    fn fresh_capabilities_with(retrievers: Vec<RetrieverCapability>) -> SystemCapabilities {
        SystemCapabilities {
            schema_version: "5.0".into(),
            snapshot_timestamp: Utc::now(),
            retrievers,
            reasoners: vec![],
            overall_status: CapabilityStatus::Healthy,
            degradation_notes: vec![],
            metadata: Default::default(),
        }
    }

    fn wikidata_retriever() -> RetrieverCapability {
        RetrieverCapability {
            retriever_id: "wikidata".into(),
            status: CapabilityStatus::Healthy,
            domains_covered: vec!["geography".into()],
            modalities_supported: vec![Modality::Text, Modality::Structured],
            typical_latency_ms: 800,
            p99_latency_ms: 4000,
            success_rate_recent: 0.97,
            last_failure_at: None,
            known_limitations: vec![],
            populates_valid_time: true,
            populates_transaction_time: true,
            populates_granularity: false,
            populates_scope: true,
            populates_certainty: false,
            populates_provenance: true,
            authority_tier: 1,
        }
    }

    fn wikidata_catalog() -> StoreCatalog {
        let mut c = StoreCatalog::new();
        c.add(RetrieverProfile {
            retriever_id: "wikidata".into(),
            default_rap_id: "wikidata_sparql_v1".into(),
            retrieval_method: RetrievalMethod::Sparql,
            estimated_latency_ms: 1000,
            estimated_cost_usd: 0.0,
            estimated_joules: 5.0,
        });
        c.add_rap(ReactiveActionPackage {
            schema_version: "2.0".into(),
            rap_id: "wikidata_sparql_v1".into(),
            description: "Wikidata SPARQL".into(),
            applies_to_methods: vec![RetrievalMethod::Sparql],
            steps: vec![RapStep {
                step_id: "primary".into(),
                description: "Exact label match".into(),
                method: "sparql_label_match".into(),
                condition: RapStepCondition::Always,
                parameters: Default::default(),
                timeout_ms: 4000,
                cost_estimate_usd: 0.0,
            }],
            max_total_attempts: 3,
            metadata: Default::default(),
        });
        c
    }

    fn simple_analysis() -> QueryAnalysis {
        QueryAnalysis {
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
            entities_extracted: vec![],
            relations_extracted: vec![],
            temporal_anchors: vec![],
            geographic_anchors: vec!["France".into()],
            domain_tags: vec!["geography".into()],
            freshness_signal: false,
            stakes_signal: StakesSignal::Low,
            raw_decomposition: vec![RawSubQuery {
                sub_id: "q0".into(),
                text: "capital of France".into(),
                required_modalities: vec![Modality::Text],
                depends_on: vec![],
                priority: 1.0,
                preferred_store: None,
            }],
            confidence: 0.95,
        }
    }

    #[test]
    fn produces_valid_plan_for_simple_lookup() {
        let analysis = simple_analysis();
        let caps = fresh_capabilities_with(vec![wikidata_retriever()]);
        let catalog = wikidata_catalog();
        let plan = plan(&analysis, &caps, &catalog, Constraints::default(), Budget::default())
            .unwrap();
        assert_eq!(plan.decomposition.len(), 1);
        assert_eq!(plan.decomposition[0].target_stores, vec!["wikidata"]);
        assert_eq!(plan.decomposition[0].rap_id, "wikidata_sparql_v1");
        assert!(plan.invariants_satisfied.all_satisfied());
    }

    #[test]
    fn refuses_when_no_store_covers_modality() {
        let mut analysis = simple_analysis();
        analysis.raw_decomposition[0].required_modalities = vec![Modality::Audio];
        let caps = fresh_capabilities_with(vec![wikidata_retriever()]);
        let catalog = wikidata_catalog();
        let err = plan(&analysis, &caps, &catalog, Constraints::default(), Budget::default())
            .unwrap_err();
        assert!(matches!(err, PlanError::NoCandidateStores { .. }));
    }

    #[test]
    fn refuses_when_unavailable() {
        let analysis = simple_analysis();
        let mut r = wikidata_retriever();
        r.status = CapabilityStatus::Unavailable;
        let caps = fresh_capabilities_with(vec![r]);
        let catalog = wikidata_catalog();
        let err = plan(&analysis, &caps, &catalog, Constraints::default(), Budget::default())
            .unwrap_err();
        assert!(matches!(err, PlanError::NoCandidateStores { .. }));
    }

    #[test]
    fn rejects_cyclic_dependencies() {
        let mut analysis = simple_analysis();
        analysis.raw_decomposition = vec![
            RawSubQuery {
                sub_id: "a".into(),
                text: "a".into(),
                required_modalities: vec![Modality::Text],
                depends_on: vec!["b".into()],
                priority: 1.0,
                preferred_store: None,
            },
            RawSubQuery {
                sub_id: "b".into(),
                text: "b".into(),
                required_modalities: vec![Modality::Text],
                depends_on: vec!["a".into()],
                priority: 1.0,
                preferred_store: None,
            },
        ];
        let caps = fresh_capabilities_with(vec![wikidata_retriever()]);
        let catalog = wikidata_catalog();
        let err = plan(&analysis, &caps, &catalog, Constraints::default(), Budget::default())
            .unwrap_err();
        assert!(matches!(err, PlanError::CycleInDependencies));
    }

    #[test]
    fn refuses_when_authority_tier_unsatisfiable() {
        let analysis = simple_analysis();
        let mut r = wikidata_retriever();
        r.authority_tier = 3; // below tier-2 minimum
        let caps = fresh_capabilities_with(vec![r]);
        let catalog = wikidata_catalog();
        let mut constraints = Constraints::default();
        constraints.minimum_authority_tier = 2;
        let err = plan(&analysis, &caps, &catalog, constraints, Budget::default()).unwrap_err();
        assert!(matches!(err, PlanError::InvariantBroken(_)));
    }

    #[test]
    fn refuses_when_budget_is_too_tight() {
        let analysis = simple_analysis();
        let caps = fresh_capabilities_with(vec![wikidata_retriever()]);
        let catalog = wikidata_catalog();
        let mut budget = Budget::default();
        budget.latency_hard_ceiling_ms = 100; // wikidata profile is 1000
        let err = plan(&analysis, &caps, &catalog, Constraints::default(), budget).unwrap_err();
        assert!(matches!(err, PlanError::Csp(CspError::Unsatisfiable)));
    }
}
