//! Pipeline runner.
//!
//! Wires Plan → Execute → Diagnose → Compose into a single async
//! function. The result + timing breakdown is what
//! [`crate::render`] turns into stdout.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;

use jouleclaw_compose::{
    cache_key_for_plan, compose_verified_answer, AnswerOrRefusal, CacheStore, CachedAnswer,
    ComposeInputs, DraftComposer, DraftSegment, TemplateComposer,
};
use jouleclaw_deberta::NliEngine;
use jouleclaw_diagnose::{
    atomize_sentences, verify, DebertaEntailer, Entailer, FixtureEntailer, VerifyInputs,
};
use jouleclaw_execute::orchestrator::{execute as orch_execute, OrchestratorConfig};
use jouleclaw_execute::retriever::RetrieverRegistry;
use jouleclaw_execute::retrievers::wikidata::WikidataRetriever;
use jouleclaw_execute::retrievers::wikipedia::WikipediaRetriever;
use jouleclaw_plan::{plan, RetrieverProfile, SelfModel, StoreCatalog};
use jouleclaw_schema::*;

use crate::cli::Options;
use crate::understanding::analyze_query;

#[derive(Debug)]
pub enum PipelineError {
    Understanding(String),
    Plan(String),
    Execute(String),
    Draft(String),
    Atomize(String),
    Verify(String),
    Compose(String),
    LoadModel(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Understanding(s) => write!(f, "understanding: {s}"),
            Self::Plan(s) => write!(f, "plan: {s}"),
            Self::Execute(s) => write!(f, "execute: {s}"),
            Self::Draft(s) => write!(f, "draft: {s}"),
            Self::Atomize(s) => write!(f, "atomize: {s}"),
            Self::Verify(s) => write!(f, "verify: {s}"),
            Self::Compose(s) => write!(f, "compose: {s}"),
            Self::LoadModel(s) => write!(f, "load model: {s}"),
        }
    }
}

impl std::error::Error for PipelineError {}

/// Top-level run output. Holds the verdict, the answer/refusal,
/// the draft + items + verification report (for `--json` and
/// `--verbose`), and per-stage timing.
#[derive(Debug)]
pub struct PipelineOutput {
    pub query: String,
    pub plan: QueryPlan,
    pub items: Vec<RetrievedItem>,
    pub draft: Vec<DraftSegment>,
    pub claims: Vec<AtomicClaim>,
    pub report: VerificationReport,
    pub result: AnswerOrRefusal,
    pub stages: StageTimings,
    pub entailer_kind: EntailerKind,
    /// How many re-route passes the loop took before settling on a
    /// final verdict. `0` = first attempt succeeded; `>0` = one or
    /// more refinements were applied per spec §8.3.
    pub reroute_passes: u32,
    /// `true` when the result came from the on-disk provenance
    /// cache rather than running the full pipeline. Used by
    /// renderers to indicate cache hits to the user.
    pub cache_hit: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum EntailerKind {
    Deberta,
    NoVerify,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StageTimings {
    pub understanding_ms: u64,
    pub plan_ms: u64,
    pub execute_ms: u64,
    pub draft_ms: u64,
    pub atomize_ms: u64,
    pub verify_ms: u64,
    pub compose_ms: u64,
    pub total_ms: u64,
}

/// Run the full pipeline for `opts.query`. Builds the entailer
/// fresh per call — fine for CLI but adds ~1.3 s of DeBERTa load
/// to every invocation. For long-running processes that handle
/// many queries, use [`run_with_entailer`] which accepts a
/// pre-built engine.
pub async fn run(opts: &Options, log: &(dyn Fn(&str) + Send + Sync)) -> Result<PipelineOutput, PipelineError> {
    run_inner(opts, None, log).await
}

/// Variant of [`run`] that uses an externally-managed entailer
/// (e.g. one loaded once by a server and shared across requests).
/// Cache lookup still happens first; on miss the supplied entailer
/// is used for verification instead of building a fresh one.
pub async fn run_with_entailer(
    opts: &Options,
    entailer: &(dyn Entailer + Sync),
    entailer_kind: EntailerKind,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<PipelineOutput, PipelineError> {
    run_inner(opts, Some((entailer, entailer_kind)), log).await
}

async fn run_inner<'e>(
    opts: &Options,
    supplied_entailer: Option<(&'e (dyn Entailer + Sync), EntailerKind)>,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<PipelineOutput, PipelineError> {
    let started = Instant::now();
    let mut t = StageTimings::default();

    // 1. Understanding (once — the loop refines the analysis below).
    let stage = Instant::now();
    log(&format!("[1] understanding: {:?}", opts.query));
    let mut analysis = analyze_query(&opts.query)
        .map_err(|e| PipelineError::Understanding(e.to_string()))?;
    t.understanding_ms = stage.elapsed().as_millis() as u64;
    log(&format!(
        "    → decomposition: {:?}",
        analysis.raw_decomposition.first().map(|s| &s.text)
    ));

    // Cache lookup — content-addressable on the analyzed sub-query
    // set. A hit returns the prior pipeline's full output in
    // microseconds, skipping retrieval + entailment + composition.
    let cache = if opts.no_cache {
        None
    } else {
        match CacheStore::open(&opts.cache_dir) {
            Ok(s) => Some(s),
            Err(e) => {
                log(&format!("    ↳ cache disabled: {e}"));
                None
            }
        }
    };
    let cache_key = if cache.is_some() {
        let preview_sub_queries: Vec<SubQuery> = analysis
            .raw_decomposition
            .iter()
            .map(|sq| SubQuery {
                sub_id: sq.sub_id.clone(),
                text: sq.text.clone(),
                depends_on: sq.depends_on.clone(),
                required_modalities: sq.required_modalities.clone(),
                target_stores: sq
                    .preferred_store
                    .as_ref()
                    .map(|s| vec![s.clone()])
                    .unwrap_or_default(),
                priority: sq.priority,
                rap_id: "tbd".into(),
            })
            .collect();
        Some(cache_key_for_plan(
            &preview_sub_queries,
            CACHE_VERSION_TAG,
            opts.wikidata_endpoint.as_deref(),
            opts.wikipedia_endpoint.as_deref(),
        ))
    } else {
        None
    };
    if let (Some(cache), Some(key)) = (cache.as_ref(), cache_key.as_ref()) {
        match cache.get(key) {
            Ok(Some(env)) => {
                log(&format!("    ↳ cache hit: {} (cached {})", key, env.cached_at));
                t.total_ms = started.elapsed().as_millis() as u64;
                return Ok(rebuild_output_from_cache(&opts.query, env, t));
            }
            Ok(None) => log("    ↳ cache miss"),
            Err(e) => log(&format!("    ↳ cache read error: {e}")),
        }
    }

    // Shared infrastructure for the loop. If the caller supplied
    // an entailer (server path), reuse it — saves the DeBERTa
    // load. Otherwise build one for this run.
    let owned_entailer = match supplied_entailer {
        Some(_) => None,
        None => Some(build_entailer(opts, log)?),
    };
    let (entailer_ref, entailer_kind): (&(dyn Entailer + Sync), EntailerKind) =
        match (&owned_entailer, supplied_entailer) {
            (Some((boxed, kind)), _) => (boxed.as_ref(), *kind),
            (None, Some((e, kind))) => (e, kind),
            (None, None) => unreachable!("exactly one entailer source is populated"),
        };
    let wikidata = Arc::new(match &opts.wikidata_endpoint {
        Some(url) => WikidataRetriever::with_endpoint(url.clone())
            .map_err(|e| PipelineError::Execute(format!("wikidata client: {e}")))?,
        None => WikidataRetriever::new()
            .map_err(|e| PipelineError::Execute(format!("wikidata client: {e}")))?,
    });
    let wikipedia = Arc::new(match &opts.wikipedia_endpoint {
        Some(url) => WikipediaRetriever::with_endpoint(url.clone())
            .map_err(|e| PipelineError::Execute(format!("wikipedia client: {e}")))?,
        None => WikipediaRetriever::new()
            .map_err(|e| PipelineError::Execute(format!("wikipedia client: {e}")))?,
    });
    let mut registry = RetrieverRegistry::new();
    registry.insert(wikidata);
    registry.insert(wikipedia);
    let (catalog, raps) = build_catalog_and_raps();
    let budget = Budget {
        latency_target_ms: 5_000,
        latency_hard_ceiling_ms: 30_000,
        ..Budget::default()
    };
    let max_reroutes = budget.max_reroute_iterations;

    let mut reroute_passes: u32 = 0;
    let mut pass_outcome: Option<PassOutcome>;

    loop {
        let pass_label = if reroute_passes == 0 {
            "first pass".to_string()
        } else {
            format!("reroute pass {reroute_passes}")
        };
        log(&format!("[loop] {pass_label}"));

        // Plan.
        let stage = Instant::now();
        let self_model = SelfModel::new();
        self_model.register_retriever(wikidata_retriever_capability());
        self_model.register_retriever(wikipedia_retriever_capability());
        let capabilities = self_model.snapshot();
        let plan = plan(
            &analysis,
            &capabilities,
            &catalog,
            Constraints::default(),
            budget.clone(),
        )
        .map_err(|e| PipelineError::Plan(e.to_string()))?;
        t.plan_ms += stage.elapsed().as_millis() as u64;

        // Execute.
        let stage = Instant::now();
        let exec = orch_execute(
            &plan,
            &raps,
            &registry,
            &self_model,
            &OrchestratorConfig::default(),
        )
        .await
        .map_err(|e| PipelineError::Execute(e.to_string()))?;
        t.execute_ms += stage.elapsed().as_millis() as u64;
        log(&format!(
            "    → execute: {} item(s), {} sub-query error(s)",
            exec.items.len(),
            exec.subquery_errors.len()
        ));

        // Draft + atomize (skip gracefully when no items).
        let stage = Instant::now();
        let draft = if exec.items.is_empty() {
            Vec::new()
        } else {
            TemplateComposer::new()
                .draft(&plan, &exec.items, &[])
                .map_err(|e| PipelineError::Draft(e.to_string()))?
        };
        t.draft_ms += stage.elapsed().as_millis() as u64;

        let stage = Instant::now();
        let claims = if draft.is_empty() {
            Vec::new()
        } else {
            let segs: Vec<(String, String)> = draft
                .iter()
                .map(|s| (s.segment_id.clone(), s.text.clone()))
                .collect();
            atomize_sentences(&segs, &axes_for_template_claim())
                .map_err(|e| PipelineError::Atomize(e.to_string()))?
        };
        t.atomize_ms += stage.elapsed().as_millis() as u64;

        // Verify (with current reroute_passes count so the verdict
        // logic knows when the budget is exhausted).
        let stage = Instant::now();
        let report = verify(
            &VerifyInputs::new(&plan, &exec.items, &[], &claims)
                .with_reroute_count(reroute_passes),
            entailer_ref,
        )
        .map_err(|e| PipelineError::Verify(e.to_string()))?;
        t.verify_ms += stage.elapsed().as_millis() as u64;
        log(&format!(
            "    → verdict {:?}, {} violation(s), {} entailment(s)",
            report.verdict,
            report.violations.len(),
            report.entailments_consulted.len()
        ));

        let needs_reroute = matches!(
            report.verdict,
            VerificationAction::ReRouteWithRefinement
        );

        // Stash this pass's products before deciding whether to loop.
        pass_outcome = Some(PassOutcome {
            plan,
            items: exec.items,
            draft,
            claims,
            report,
        });

        if !needs_reroute {
            break;
        }
        if reroute_passes >= max_reroutes {
            // verify() already returned ReRoute despite reroute_count
            // hitting the cap — this can happen on the boundary.
            // Treat as exhausted; downstream compose will pick the
            // refusal path from a stricter verdict.
            log("    → reroute budget exhausted, settling on current pass");
            break;
        }

        // Build refinement hints from the violations.
        let violations: Vec<_> = pass_outcome
            .as_ref()
            .map(|o| o.report.violations.clone())
            .unwrap_or_default();
        let refined = apply_refinement(analysis.clone(), &violations);
        if refined.raw_decomposition == analysis.raw_decomposition {
            // No refinement is possible — looping again would be
            // identical. Settle on the current verdict.
            log("    → refinement produced no change, settling");
            break;
        }
        log(&format!(
            "    ↳ refining: {:?} → {:?}",
            analysis.raw_decomposition.first().map(|s| &s.text),
            refined.raw_decomposition.first().map(|s| &s.text),
        ));
        analysis = refined;
        reroute_passes += 1;
    }

    let outcome = pass_outcome.expect("loop ran at least once");

    // Re-verify once more if the final verdict is ReRoute but our
    // budget is exhausted — feed reroute_count = max_reroutes so
    // determine_verdict() escalates to Refuse instead of staying on
    // ReRoute (compose can't proceed from ReRoute).
    let final_report = if matches!(
        outcome.report.verdict,
        VerificationAction::ReRouteWithRefinement
    ) {
        let stage = Instant::now();
        let r = verify(
            &VerifyInputs::new(&outcome.plan, &outcome.items, &[], &outcome.claims)
                .with_reroute_count(max_reroutes),
            entailer_ref,
        )
        .map_err(|e| PipelineError::Verify(e.to_string()))?;
        t.verify_ms += stage.elapsed().as_millis() as u64;
        r
    } else {
        outcome.report
    };

    // Compose.
    let stage = Instant::now();
    let inputs = ComposeInputs {
        plan: &outcome.plan,
        draft_segments: &outcome.draft,
        claims: &outcome.claims,
        items: &outcome.items,
        report: &final_report,
        joules_spent_total: final_report.joules_spent,
        latency_ms: started.elapsed().as_millis() as u64,
    };
    let result =
        compose_verified_answer(&inputs).map_err(|e| PipelineError::Compose(e.to_string()))?;
    t.compose_ms = stage.elapsed().as_millis() as u64;
    t.total_ms = started.elapsed().as_millis() as u64;

    let output = PipelineOutput {
        query: opts.query.clone(),
        plan: outcome.plan,
        items: outcome.items,
        draft: outcome.draft,
        claims: outcome.claims,
        report: final_report,
        result: result.clone(),
        stages: t,
        entailer_kind,
        reroute_passes,
        cache_hit: false,
    };

    // Store on cache miss. Only cache successful runs where the
    // verdict isn't ReRoute (which shouldn't get this far anyway).
    // Refusals are cached too — repeating a query whose plan can't
    // be satisfied shouldn't pay the full Wikidata + DeBERTa cost
    // every time.
    if let (Some(cache), Some(key)) = (cache.as_ref(), cache_key.as_ref()) {
        let env = CachedAnswer {
            schema_version: "1.0".into(),
            cache_key: key.clone(),
            cached_at: Utc::now(),
            query_text: opts.query.clone(),
            result,
        };
        if let Err(e) = cache.put(&env) {
            log(&format!("    ↳ cache write failed: {e}"));
        } else {
            log(&format!("    ↳ cached under {key}"));
        }
    }

    Ok(output)
}

/// Bump when changing prompts, atomizer, entailer model, or
/// anything else that meaningfully alters the cached answer.
/// Cached entries from an older tag are invisible after a bump —
/// no manual cleanup required.
const CACHE_VERSION_TAG: &str = "jouleclaw-edge-cli/0.1";

/// Reconstruct a `PipelineOutput` from a cached answer so the
/// renderer can show the same shape (with `cache_hit = true`).
/// Most fields are derived from the AnswerOrRefusal itself; the
/// pipeline-specific bookkeeping (plan, draft, claims, items,
/// stages timings) we reconstruct cheaply from the cached Answer.
fn rebuild_output_from_cache(
    query: &str,
    env: CachedAnswer,
    stages: StageTimings,
) -> PipelineOutput {
    use jouleclaw_schema::*;
    let (items, claims, report) = match &env.result {
        AnswerOrRefusal::Answer(a) => (
            a.provenance.items.clone(),
            a.provenance.claims.clone(),
            VerificationReport {
                schema_version: "2.0".into(),
                report_id: uuid::Uuid::new_v4(),
                plan_id: a.plan_id,
                generated_at: env.cached_at,
                // Reconstruct the verdict from the cached
                // AnswerStatus so the renderer's status / verdict
                // pair is consistent.
                verdict: match a.status {
                    AnswerStatus::Verified => VerificationAction::ProceedToComposition,
                    AnswerStatus::Degraded => VerificationAction::DegradeGracefully,
                    AnswerStatus::Partial => VerificationAction::ProceedToComposition,
                },
                violations: vec![],
                recovery_actions: vec![],
                entailments_consulted: a.provenance.entailments.clone(),
                invariants_verified: a.invariants_verified.clone(),
                joules_spent: 0.0,
                metadata: Default::default(),
            },
        ),
        AnswerOrRefusal::Refusal(r) => (
            vec![],
            vec![],
            VerificationReport {
                schema_version: "2.0".into(),
                report_id: uuid::Uuid::new_v4(),
                plan_id: r.plan_id,
                generated_at: env.cached_at,
                verdict: VerificationAction::Refuse,
                violations: vec![],
                recovery_actions: vec![],
                entailments_consulted: vec![],
                invariants_verified: vec![],
                joules_spent: 0.0,
                metadata: Default::default(),
            },
        ),
    };
    let plan_id = match &env.result {
        AnswerOrRefusal::Answer(a) => a.plan_id,
        AnswerOrRefusal::Refusal(r) => r.plan_id,
    };
    PipelineOutput {
        query: query.to_string(),
        plan: QueryPlan {
            schema_version: "2.0".into(),
            plan_id,
            original_query: OriginalQuery {
                text: Some(query.to_string()),
                image_ref: None,
                audio_ref: None,
                video_ref: None,
                language_detected: "en".into(),
                timestamp: env.cached_at,
            },
            intent: Intent::Lookup,
            modalities_in: vec![Modality::Text],
            modalities_out: vec![Modality::Text],
            decomposition: vec![],
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
        },
        items,
        draft: vec![],
        claims,
        report,
        result: env.result,
        stages,
        entailer_kind: EntailerKind::NoVerify, // We didn't run DeBERTa
        reroute_passes: 0,
        cache_hit: true,
    }
}

/// Bundle returned by a single pass of the loop.
struct PassOutcome {
    plan: QueryPlan,
    items: Vec<RetrievedItem>,
    draft: Vec<DraftSegment>,
    claims: Vec<AtomicClaim>,
    report: VerificationReport,
}

/// Apply refinement hints to the analysis based on the previous
/// pass's violations. The simplest concrete refinement: for
/// coverage gaps, simplify the sub-query decomposition by dropping
/// a relational prefix ("capital of France" → "France") so the
/// next pass at least retrieves the bare entity.
///
/// Returns the analysis unchanged if no actionable refinement
/// applies — the caller treats that as a signal to break.
fn apply_refinement(
    mut analysis: jouleclaw_plan::QueryAnalysis,
    violations: &[Violation],
) -> jouleclaw_plan::QueryAnalysis {
    use serde_json::Value;

    let coverage_sub_ids: std::collections::BTreeSet<String> = violations
        .iter()
        .filter_map(|v| {
            let kind = v.detail.get("kind")?.as_str()?;
            if kind != "coverage" {
                return None;
            }
            v.detail
                .get("sub_id")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
        })
        .collect();

    if coverage_sub_ids.is_empty() {
        // Other violation kinds (authority, grounding, freshness)
        // don't currently produce refinement hints. The loop will
        // settle on the original analysis.
        let _ = Value::Null;
        return analysis;
    }

    for sq in analysis.raw_decomposition.iter_mut() {
        if !coverage_sub_ids.contains(&sq.sub_id) {
            continue;
        }
        let simplified = simplify_relational_query(&sq.text);
        if simplified != sq.text {
            sq.text = simplified;
        }
    }
    analysis
}

/// Drop a leading relational prefix ("capital of ", "currency of ",
/// "official language of ", …) so the next pass searches for just
/// the subject entity. If no prefix matches, return the input
/// unchanged.
fn simplify_relational_query(s: &str) -> String {
    let lower = s.to_lowercase();
    const PREFIXES: &[&str] = &[
        "capital of ",
        "the capital of ",
        "currency of ",
        "official language of ",
        "language of ",
        "head of state of ",
        "head of government of ",
        "continent of ",
        "country of ",
        "population of ",
        "area of ",
        "borders of ",
    ];
    for p in PREFIXES {
        if let Some(rest) = lower.strip_prefix(p) {
            // Slice the original by byte offset so we preserve
            // capitalization on the subject.
            let byte_off = p.len();
            if byte_off <= s.len() {
                return s[byte_off..].trim().to_string();
            }
            return rest.trim().to_string();
        }
    }
    s.to_string()
}

fn build_entailer(
    opts: &Options,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<(Box<dyn Entailer>, EntailerKind), PipelineError> {
    if opts.no_verify {
        let fx = FixtureEntailer::new();
        // Default to permissive "Entails" for the dev path.
        fx.set_default(jouleclaw_diagnose::entailer::RawEntailment {
            label: EntailmentLabel::Entails,
            probabilities: EntailmentProbabilities {
                entails: 1.0,
                neutral: 0.0,
                contradicts: 0.0,
            },
            joules_spent: 0.0,
        });
        return Ok((Box::new(fx), EntailerKind::NoVerify));
    }

    log(&format!(
        "      ↳ loading DeBERTa from {} (this allocates ~1.7 GB)",
        opts.model_dir.display()
    ));
    let engine = NliEngine::from_dir(&opts.model_dir)
        .map_err(|e| PipelineError::LoadModel(e.to_string()))?;
    let entailer = DebertaEntailer::new(engine);
    Ok((Box::new(entailer), EntailerKind::Deberta))
}

fn wikidata_retriever_capability() -> RetrieverCapability {
    RetrieverCapability {
        retriever_id: "wikidata".into(),
        status: CapabilityStatus::Healthy,
        domains_covered: vec!["geography".into(), "people".into(), "structured".into()],
        modalities_supported: vec![Modality::Text, Modality::Structured],
        typical_latency_ms: 1000,
        p99_latency_ms: 8000,
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

fn build_catalog_and_raps() -> (StoreCatalog, BTreeMap<String, ReactiveActionPackage>) {
    let mut catalog = StoreCatalog::new();
    catalog.add(RetrieverProfile {
        retriever_id: "wikidata".into(),
        default_rap_id: "wikidata_sparql_v1".into(),
        retrieval_method: RetrievalMethod::Sparql,
        estimated_latency_ms: 2000,
        estimated_cost_usd: 0.0,
        estimated_joules: 5.0,
    });
    catalog.add(RetrieverProfile {
        retriever_id: "wikipedia".into(),
        default_rap_id: "wikipedia_summary_v1".into(),
        retrieval_method: RetrievalMethod::LiveSearch,
        estimated_latency_ms: 800,
        estimated_cost_usd: 0.0,
        estimated_joules: 3.0,
    });

    let wikidata_rap = ReactiveActionPackage {
        schema_version: "2.0".into(),
        rap_id: "wikidata_sparql_v1".into(),
        description: "Wikidata SPARQL: EntitySearch primary + property-path fallback".into(),
        applies_to_methods: vec![RetrievalMethod::Sparql],
        steps: vec![
            RapStep {
                step_id: "primary".into(),
                description: "Entity-search SPARQL, filtered for meta-entities".into(),
                method: "wikidata_primary".into(),
                condition: RapStepCondition::Always,
                parameters: Default::default(),
                timeout_ms: 8000,
                cost_estimate_usd: 0.0,
            },
            RapStep {
                step_id: "property_path".into(),
                description: "Property-path SPARQL for relational queries".into(),
                method: "wikidata_property_path".into(),
                condition: RapStepCondition::OnEmpty,
                parameters: Default::default(),
                timeout_ms: 8000,
                cost_estimate_usd: 0.0,
            },
        ],
        max_total_attempts: 2,
        metadata: Default::default(),
    };
    let wikipedia_rap = ReactiveActionPackage {
        schema_version: "2.0".into(),
        rap_id: "wikipedia_summary_v1".into(),
        description: "Wikipedia REST summary, single step".into(),
        applies_to_methods: vec![RetrievalMethod::LiveSearch],
        steps: vec![RapStep {
            step_id: "primary".into(),
            description: "REST summary of the candidate title".into(),
            method: "wikipedia_summary".into(),
            condition: RapStepCondition::Always,
            parameters: Default::default(),
            timeout_ms: 5000,
            cost_estimate_usd: 0.0,
        }],
        max_total_attempts: 1,
        metadata: Default::default(),
    };
    catalog.add_rap(wikidata_rap.clone());
    catalog.add_rap(wikipedia_rap.clone());
    let mut raps = BTreeMap::new();
    raps.insert("wikidata_sparql_v1".into(), wikidata_rap);
    raps.insert("wikipedia_summary_v1".into(), wikipedia_rap);
    (catalog, raps)
}

fn wikipedia_retriever_capability() -> RetrieverCapability {
    RetrieverCapability {
        retriever_id: "wikipedia".into(),
        status: CapabilityStatus::Healthy,
        domains_covered: vec!["encyclopedia".into(), "prose".into()],
        modalities_supported: vec![Modality::Text],
        typical_latency_ms: 500,
        p99_latency_ms: 3000,
        success_rate_recent: 0.99,
        last_failure_at: None,
        known_limitations: vec!["English Wikipedia only".into()],
        populates_valid_time: false,
        populates_transaction_time: true,
        populates_granularity: false,
        populates_scope: true,
        populates_certainty: false,
        populates_provenance: true,
        authority_tier: 3, // tertiary aggregator
    }
}

fn axes_for_template_claim() -> KnowledgeAxes {
    KnowledgeAxes {
        schema_version: "5.0".into(),
        valid_time_start: None,
        valid_time_end: None,
        transaction_time: None,
        reference_time: Utc::now(),
        temporal_stability: TemporalStabilityClass::Slow,
        granularity: GranularityClass::Medium,
        granularity_notes: None,
        scope: ScopeClass::Particular,
        scope_domain: Some("Wikidata".into()),
        certainty: 0.95,
        certainty_basis: "wikidata_structured_kb".into(),
        source_uri: None,
        source_authority_tier: 1,
        extraction_method: Some("sparql_entity_search".into()),
        citation_chain: vec![],
        metadata: Default::default(),
    }
}
