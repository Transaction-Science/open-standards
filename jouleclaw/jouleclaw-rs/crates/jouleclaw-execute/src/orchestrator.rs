//! Orchestrator (spec §5.1).
//!
//! Topologically sorts the QueryPlan's sub-queries, dispatches each
//! level concurrently, runs each sub-query through its RAP, and
//! accumulates [`jouleclaw_schema::RetrievedItem`]s. Budgets are enforced
//! at the level boundary; the inner RAP executor takes per-step
//! timeouts and respects the deadline computed from
//! `latency_hard_ceiling_ms`.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jouleclaw_plan::SelfModel;
use jouleclaw_schema::{QueryPlan, ReactiveActionPackage, RetrievedItem, SubQuery};

use crate::authority::score_authority;
use crate::rap::{rap_execute, RapExecError};
use crate::retriever::RetrieverRegistry;

#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Minimum results before the primary RAP step is considered
    /// "good enough" and the RAP stops.
    pub min_results_per_subquery: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            min_results_per_subquery: 1,
        }
    }
}

#[derive(Debug)]
pub enum ExecuteError {
    /// The plan's dependency graph couldn't be topologically sorted.
    InvalidPlan(String),
    /// The plan references a `rap_id` not in the registry.
    UnknownRap(String),
    /// The plan references a `target_store` with no registered
    /// retriever.
    UnknownRetriever(String),
}

impl std::fmt::Display for ExecuteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPlan(s) => write!(f, "invalid plan: {s}"),
            Self::UnknownRap(s) => write!(f, "unknown rap: {s}"),
            Self::UnknownRetriever(s) => write!(f, "unknown retriever: {s}"),
        }
    }
}

impl std::error::Error for ExecuteError {}

#[derive(Debug)]
pub struct ExecutionResult {
    pub items: Vec<RetrievedItem>,
    pub elapsed: Duration,
    pub estimated_cost_usd: f64,
    /// `(sub_id, error)` for sub-queries that failed. Coverage gaps
    /// here are surfaced by the Diagnose pillar (§6.1), not as
    /// orchestration errors.
    pub subquery_errors: Vec<(String, String)>,
}

/// Top-level orchestrator entry point.
pub async fn execute(
    plan: &QueryPlan,
    raps: &BTreeMap<String, ReactiveActionPackage>,
    retrievers: &RetrieverRegistry,
    self_model: &SelfModel,
    config: &OrchestratorConfig,
) -> Result<ExecutionResult, ExecuteError> {
    let start = Instant::now();
    let deadline = start + Duration::from_millis(plan.budget.latency_hard_ceiling_ms as u64);
    let levels = topo_sort_levels(&plan.decomposition)?;

    let mut all_items: Vec<RetrievedItem> = Vec::new();
    let mut total_cost: f64 = 0.0;
    let mut errors: Vec<(String, String)> = Vec::new();

    for level in levels {
        let mut handles: Vec<(
            String,
            tokio::task::JoinHandle<(Result<crate::rap::RapOutcome, RapExecError>, Duration)>,
        )> = Vec::new();
        for sq in &level {
            // Resolve dependencies — pick first target_store; spec §3.2
            // allows multiple but the planner currently emits one.
            let store = sq
                .target_stores
                .first()
                .ok_or_else(|| ExecuteError::InvalidPlan(format!("sub {} has no store", sq.sub_id)))?
                .clone();
            let retriever = retrievers
                .get(&store)
                .ok_or_else(|| ExecuteError::UnknownRetriever(store.clone()))?
                .clone();
            let rap = raps
                .get(&sq.rap_id)
                .ok_or_else(|| ExecuteError::UnknownRap(sq.rap_id.clone()))?
                .clone();
            let sq_cloned = sq.clone();
            let min_results = config.min_results_per_subquery;
            let sub_id = sq.sub_id.clone();
            let handle = tokio::spawn(async move {
                let t0 = Instant::now();
                let outcome = rap_execute(&rap, &*retriever, &sq_cloned, deadline, min_results).await;
                (outcome, t0.elapsed())
            });
            handles.push((sub_id, handle));
        }

        // Cost ceiling check before awaiting: if we've already burned
        // budget on a prior level, drop the rest.
        if total_cost > plan.budget.cost_ceiling_usd {
            for (sub_id, h) in handles {
                h.abort();
                errors.push((sub_id, "cost ceiling exhausted".into()));
            }
            break;
        }

        for (sub_id, handle) in handles {
            match handle.await {
                Ok((Ok(outcome), elapsed)) => {
                    total_cost += outcome.cost_estimate_usd;
                    self_model.observe(
                        &store_for(&plan.decomposition, &sub_id),
                        jouleclaw_plan::Observation {
                            success: true,
                            latency_ms: elapsed.as_millis() as u32,
                        },
                    );
                    all_items.extend(outcome.items);
                }
                Ok((Err(e), elapsed)) => {
                    self_model.observe(
                        &store_for(&plan.decomposition, &sub_id),
                        jouleclaw_plan::Observation {
                            success: false,
                            latency_ms: elapsed.as_millis() as u32,
                        },
                    );
                    errors.push((sub_id, e.to_string()));
                }
                Err(join_err) => {
                    errors.push((sub_id, format!("task panic: {join_err}")));
                }
            }

            if Instant::now() >= deadline {
                break;
            }
        }
    }

    // Score authority for every item — produces side-channel records
    // currently attached via metadata; explicit AuthorityRecord
    // emission lives one layer up in the Diagnose pillar.
    let _ = all_items
        .iter()
        .map(score_authority)
        .collect::<Vec<_>>();

    let _ = Arc::new(()); // keep clippy happy when no Arc fields are used in tests
    Ok(ExecutionResult {
        items: all_items,
        elapsed: start.elapsed(),
        estimated_cost_usd: total_cost,
        subquery_errors: errors,
    })
}

fn store_for(decomposition: &[SubQuery], sub_id: &str) -> String {
    decomposition
        .iter()
        .find(|s| s.sub_id == sub_id)
        .and_then(|s| s.target_stores.first().cloned())
        .unwrap_or_default()
}

/// Group sub-queries into dependency levels using Kahn's algorithm.
/// Each level is a set of sub-queries that can execute in parallel.
fn topo_sort_levels(decomposition: &[SubQuery]) -> Result<Vec<Vec<SubQuery>>, ExecuteError> {
    use std::collections::HashMap;
    let mut in_deg: HashMap<&str, usize> = HashMap::new();
    let mut out_edges: HashMap<&str, Vec<&str>> = HashMap::new();
    for sq in decomposition {
        in_deg.entry(sq.sub_id.as_str()).or_insert(0);
        for dep in &sq.depends_on {
            *in_deg.entry(sq.sub_id.as_str()).or_insert(0) += 1;
            out_edges.entry(dep.as_str()).or_default().push(sq.sub_id.as_str());
        }
    }
    // Validate every named dep exists.
    let known: HashSet<&str> = decomposition.iter().map(|s| s.sub_id.as_str()).collect();
    for sq in decomposition {
        for dep in &sq.depends_on {
            if !known.contains(dep.as_str()) {
                return Err(ExecuteError::InvalidPlan(format!(
                    "sub {} depends on unknown {dep}",
                    sq.sub_id
                )));
            }
        }
    }

    let mut levels: Vec<Vec<SubQuery>> = Vec::new();
    let lookup: HashMap<&str, &SubQuery> =
        decomposition.iter().map(|s| (s.sub_id.as_str(), s)).collect();
    let mut visited: BTreeSet<String> = BTreeSet::new();

    loop {
        let ready: Vec<&str> = in_deg
            .iter()
            .filter(|(k, v)| **v == 0 && !visited.contains(**k))
            .map(|(k, _)| *k)
            .collect();
        if ready.is_empty() {
            if visited.len() == decomposition.len() {
                break;
            }
            return Err(ExecuteError::InvalidPlan("cycle in dependencies".into()));
        }
        let mut this_level: Vec<SubQuery> = ready
            .iter()
            .filter_map(|id| lookup.get(*id).map(|sq| (*sq).clone()))
            .collect();
        this_level.sort_by(|a, b| a.sub_id.cmp(&b.sub_id));
        for sq in &this_level {
            visited.insert(sq.sub_id.clone());
        }
        for id in &ready {
            if let Some(children) = out_edges.get(id) {
                for c in children {
                    if let Some(v) = in_deg.get_mut(*c) {
                        *v = v.saturating_sub(1);
                    }
                }
            }
            in_deg.remove(*id);
        }
        levels.push(this_level);
    }
    Ok(levels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrievers::fixture::FixtureRetriever;
    use chrono::Utc;
    use jouleclaw_plan::{Observation, SelfModel};
    use jouleclaw_schema::{
        Attribution, Content, FreshnessClass, GranularityClass, Intent, KnowledgeAxes, Modality,
        OriginalQuery, PlanInvariants, RapStep, RapStepCondition, ReactiveActionPackage,
        RetrievalContext, RetrievalMethod, RetrievedItem, ScopeClass, ScoreType, SourceType,
        SubQuery, Temporal, TemporalStabilityClass,
    };
    use uuid::Uuid;

    fn axes() -> KnowledgeAxes {
        KnowledgeAxes {
            schema_version: "5.0".into(),
            valid_time_start: None,
            valid_time_end: None,
            transaction_time: None,
            reference_time: Utc::now(),
            temporal_stability: TemporalStabilityClass::Invariant,
            granularity: GranularityClass::Coarse,
            granularity_notes: None,
            scope: ScopeClass::Universal,
            scope_domain: None,
            certainty: 0.99,
            certainty_basis: "fixture".into(),
            source_uri: None,
            source_authority_tier: 1,
            extraction_method: None,
            citation_chain: vec![],
            metadata: Default::default(),
        }
    }

    fn paris_item() -> RetrievedItem {
        RetrievedItem {
            schema_version: "2.0".into(),
            item_id: Uuid::new_v4(),
            source_id: "wikidata:Q90".into(),
            source_url: Some("https://www.wikidata.org/wiki/Q90".into()),
            source_type: SourceType::StructuredKb,
            content: Content {
                modality: Modality::Text,
                text: Some("Paris".into()),
                media_ref: None,
                structured: None,
                excerpt_span: None,
            },
            retrieval_context: RetrievalContext {
                retriever_id: "fixture".into(),
                matched_against: "capital of France".into(),
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
            knowledge_axes: axes(),
            metadata: Default::default(),
        }
    }

    fn minimal_plan() -> QueryPlan {
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
                target_stores: vec!["fixture".into()],
                priority: 1.0,
                rap_id: "rap0".into(),
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

    fn one_step_rap() -> ReactiveActionPackage {
        ReactiveActionPackage {
            schema_version: "2.0".into(),
            rap_id: "rap0".into(),
            description: "primary only".into(),
            applies_to_methods: vec![RetrievalMethod::Sparql],
            steps: vec![RapStep {
                step_id: "primary".into(),
                description: "primary".into(),
                method: "primary".into(),
                condition: RapStepCondition::Always,
                parameters: Default::default(),
                timeout_ms: 1000,
                cost_estimate_usd: 0.0,
            }],
            max_total_attempts: 1,
            metadata: Default::default(),
        }
    }

    fn build_registry(items_for_primary: Vec<RetrievedItem>) -> RetrieverRegistry {
        let f = FixtureRetriever::new("fixture");
        f.set_method("primary", Ok(items_for_primary));
        let mut reg = RetrieverRegistry::new();
        reg.insert(Arc::new(f));
        reg
    }

    fn build_raps() -> BTreeMap<String, ReactiveActionPackage> {
        let mut m = BTreeMap::new();
        m.insert("rap0".into(), one_step_rap());
        m
    }

    #[tokio::test]
    async fn executes_minimal_plan_and_returns_items() {
        let plan = minimal_plan();
        let reg = build_registry(vec![paris_item()]);
        let raps = build_raps();
        let sm = SelfModel::new();
        let res = execute(&plan, &raps, &reg, &sm, &Default::default()).await.unwrap();
        assert_eq!(res.items.len(), 1);
        assert!(res.subquery_errors.is_empty());
    }

    #[tokio::test]
    async fn surfaces_subquery_errors_without_aborting_plan() {
        let plan = minimal_plan();
        let reg = build_registry(vec![]); // empty primary → RAP exhausts
        let raps = build_raps();
        let sm = SelfModel::new();
        let res = execute(&plan, &raps, &reg, &sm, &Default::default()).await.unwrap();
        assert!(res.items.is_empty());
        assert_eq!(res.subquery_errors.len(), 1);
        assert_eq!(res.subquery_errors[0].0, "q0");
    }

    #[test]
    fn topo_sort_levels_groups_parallelizable_subqueries() {
        let decomp = vec![
            SubQuery {
                sub_id: "a".into(),
                text: "".into(),
                depends_on: vec![],
                required_modalities: vec![Modality::Text],
                target_stores: vec!["x".into()],
                priority: 1.0,
                rap_id: "r".into(),
            },
            SubQuery {
                sub_id: "b".into(),
                text: "".into(),
                depends_on: vec![],
                required_modalities: vec![Modality::Text],
                target_stores: vec!["x".into()],
                priority: 1.0,
                rap_id: "r".into(),
            },
            SubQuery {
                sub_id: "c".into(),
                text: "".into(),
                depends_on: vec!["a".into(), "b".into()],
                required_modalities: vec![Modality::Text],
                target_stores: vec!["x".into()],
                priority: 1.0,
                rap_id: "r".into(),
            },
        ];
        let levels = topo_sort_levels(&decomp).unwrap();
        assert_eq!(levels.len(), 2);
        let ids_l0: Vec<&str> = levels[0].iter().map(|s| s.sub_id.as_str()).collect();
        let ids_l1: Vec<&str> = levels[1].iter().map(|s| s.sub_id.as_str()).collect();
        assert_eq!(ids_l0, vec!["a", "b"]);
        assert_eq!(ids_l1, vec!["c"]);
    }

    #[test]
    fn topo_sort_levels_rejects_dangling_dependency() {
        let decomp = vec![SubQuery {
            sub_id: "a".into(),
            text: "".into(),
            depends_on: vec!["nonexistent".into()],
            required_modalities: vec![Modality::Text],
            target_stores: vec!["x".into()],
            priority: 1.0,
            rap_id: "r".into(),
        }];
        let err = topo_sort_levels(&decomp).unwrap_err();
        assert!(matches!(err, ExecuteError::InvalidPlan(_)));
    }

    #[tokio::test]
    async fn unknown_retriever_is_an_error() {
        let mut plan = minimal_plan();
        plan.decomposition[0].target_stores = vec!["unknown_store".into()];
        let reg = build_registry(vec![paris_item()]);
        let raps = build_raps();
        let sm = SelfModel::new();
        let err = execute(&plan, &raps, &reg, &sm, &Default::default()).await.unwrap_err();
        assert!(matches!(err, ExecuteError::UnknownRetriever(_)));
    }

    #[test]
    fn observation_struct_is_constructible() {
        // Sanity that the jouleclaw-plan re-export wires up.
        let _ = Observation { success: true, latency_ms: 0 };
    }
}
