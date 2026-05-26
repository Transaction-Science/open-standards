//! Integration test: the diagnose pillar running against the real
//! DeBERTa-v3-large-mnli NLI engine.
//!
//! Verifies the whole chain — sentence atomization, focused
//! entailment via DeBERTa, ValidAnswerModel violations, verdict,
//! recovery — produces sensible outputs on the canonical NLI pair
//! and on a deliberately contradictory pair.
//!
//! Slow (~minute per case at 22s/inference × claims × items in
//! single-threaded fp32). Skipped if the model isn't on disk.

use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use uuid::Uuid;

use jouleclaw_deberta::NliEngine;
use jouleclaw_diagnose::{atomize_sentences, verify, DebertaEntailer, VerifyInputs};
use jouleclaw_schema::*;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .expect("workspace root")
}

fn model_dir() -> Option<PathBuf> {
    let p = workspace_root().join("models/deberta-v3-large-mnli");
    if p.join("model.safetensors").exists() {
        Some(p)
    } else {
        None
    }
}

fn axes() -> KnowledgeAxes {
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
        scope_domain: Some("France".into()),
        certainty: 0.95,
        certainty_basis: "wikidata".into(),
        source_uri: Some("wikidata:Q90".into()),
        source_authority_tier: 1,
        extraction_method: Some("structured_api".into()),
        citation_chain: vec![],
        metadata: Default::default(),
    }
}

fn make_item(sub_id: &str, source_id: &str, text: &str) -> RetrievedItem {
    RetrievedItem {
        schema_version: "2.0".into(),
        item_id: Uuid::new_v4(),
        source_id: source_id.into(),
        source_url: Some(format!("https://www.wikidata.org/wiki/{source_id}")),
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
            matched_against: sub_id.into(),
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
        attribution: Attribution {
            publisher: Some("Wikidata".into()),
            license: Some("CC0".into()),
            canonical_url: Some(format!("https://www.wikidata.org/wiki/{source_id}")),
            ..Default::default()
        },
        knowledge_axes: axes(),
        metadata: Default::default(),
    }
}

fn simple_plan() -> QueryPlan {
    QueryPlan {
        schema_version: "2.0".into(),
        plan_id: Uuid::new_v4(),
        original_query: OriginalQuery {
            text: Some("What is the capital of France?".into()),
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
fn diagnose_proceeds_when_deberta_entails_the_claim() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: DeBERTa model not downloaded");
        return;
    };
    let engine = NliEngine::from_dir(&dir).expect("load engine");
    let entailer = DebertaEntailer::new(engine);

    let plan = simple_plan();
    let items = vec![make_item(
        "q0",
        "wikidata:Q90",
        "Paris is the capital of France.",
    )];
    let segs = vec![("s0".into(), "Paris is the capital of France.".into())];
    let claims = atomize_sentences(&segs, &axes()).expect("atomize");

    let report = verify(
        &VerifyInputs::new(&plan, &items, &[], &claims),
        &entailer,
    )
    .expect("verify");

    eprintln!(
        "[entails case] verdict={:?} violations={} entailments={} joules_spent={:.1}",
        report.verdict,
        report.violations.len(),
        report.entailments_consulted.len(),
        report.joules_spent
    );
    assert!(
        matches!(report.verdict, VerificationAction::ProceedToComposition),
        "expected ProceedToComposition, got {:?} with violations {:?}",
        report.verdict, report.violations
    );
    assert_eq!(
        report.entailments_consulted.len(),
        1,
        "one claim × one item = one entailment call"
    );
    let e = &report.entailments_consulted[0];
    assert!(matches!(e.label, EntailmentLabel::Entails));
    assert!(
        e.label_probabilities.entails > 0.9,
        "DeBERTa should be very confident about Paris=capital-of-France"
    );
}

#[test]
fn diagnose_surfaces_contradiction_via_deberta() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: DeBERTa model not downloaded");
        return;
    };
    let engine = NliEngine::from_dir(&dir).expect("load engine");
    let entailer = DebertaEntailer::new(engine);

    let plan = simple_plan();
    // Two contradictory items for the same sub-query.
    let it_paris = make_item("q0", "wikidata:Q90", "Paris is the capital of France.");
    let it_lyon = make_item("q0", "blog:misinfo", "Lyon is the capital of France.");
    let items = vec![it_paris, it_lyon];
    let segs = vec![("s0".into(), "Paris is the capital of France.".into())];
    let claims = atomize_sentences(&segs, &axes()).expect("atomize");

    let report = verify(
        &VerifyInputs::new(&plan, &items, &[], &claims),
        &entailer,
    )
    .expect("verify");

    eprintln!(
        "[contradiction case] verdict={:?} violations={} entailments={}",
        report.verdict,
        report.violations.len(),
        report.entailments_consulted.len()
    );
    for v in &report.violations {
        eprintln!("  - {} ({:?}): {}", v.violation_id, v.severity, v.message);
    }

    // Expect a consistency violation surfaced (because one source
    // entails Paris and the other contradicts it).
    let consistency = report
        .violations
        .iter()
        .filter(|v| v.violation_id.starts_with("consistency"))
        .count();
    assert!(
        consistency >= 1,
        "expected at least one consistency violation, got: {:?}",
        report.violations
    );
    // Consistency is Major (not Critical), so the verdict should be DEGRADE.
    assert!(matches!(report.verdict, VerificationAction::DegradeGracefully));
    // I6 — conflicts surfaced — must be in the verified-invariants list.
    assert!(report.invariants_verified.iter().any(|s| s == "I6"));
}

#[test]
fn diagnose_with_shared_engine_runs_two_cases_sequentially() {
    // Validates that we can run the engine through repeated verify
    // calls; matters for caching and the future cascade integration.
    let Some(dir) = model_dir() else {
        eprintln!("skip: DeBERTa model not downloaded");
        return;
    };
    let engine = NliEngine::from_dir(&dir).expect("load engine");
    let entailer = Arc::new(DebertaEntailer::new(engine));

    for (premise_text, hyp_text) in &[
        ("Paris is the capital of France.", "Paris is the capital of France."),
        ("Marie Curie won the Nobel Prize in 1903.", "Marie Curie won a Nobel."),
    ] {
        let plan = simple_plan();
        let items = vec![make_item("q0", "wikidata:test", premise_text)];
        let segs = vec![("s0".into(), (*hyp_text).into())];
        let claims = atomize_sentences(&segs, &axes()).expect("atomize");
        let report = verify(
            &VerifyInputs::new(&plan, &items, &[], &claims),
            entailer.as_ref(),
        )
        .expect("verify");
        eprintln!(
            "[{:.40}]: verdict={:?} entail_prob={:?}",
            hyp_text,
            report.verdict,
            report
                .entailments_consulted
                .first()
                .map(|e| (e.label, e.label_probabilities.entails))
        );
        assert!(
            matches!(report.verdict, VerificationAction::ProceedToComposition),
            "premise={premise_text} hyp={hyp_text} verdict={:?}",
            report.verdict
        );
    }
}
