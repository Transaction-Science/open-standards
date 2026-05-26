//! RAP executor (spec §5.2).
//!
//! Walks a [`ReactiveActionPackage`]'s steps in order. Each step
//! fires only if its `condition` is consistent with the previous
//! step's outcome (or if it's `Always`). Per-step `timeout_ms` is
//! enforced via `tokio::time::timeout`. Stops when:
//!
//! - A step produces items (returns them annotated with `rap_step`).
//! - `max_total_attempts` is exhausted.
//! - The supplied deadline is exceeded.

use std::time::{Duration, Instant};

use jouleclaw_schema::{
    RapStep, RapStepCondition, ReactiveActionPackage, RetrievedItem, SubQuery,
};

use crate::retriever::Retriever;

/// What happened the last time a step ran. Drives the next step's
/// condition check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LastOutcome {
    Success,
    Empty,
    Error,
    Timeout,
    /// Reserved — no retriever currently emits this. Wired through
    /// the match below so a future scorer-aware retriever can mark a
    /// result low-confidence and trigger an `OnLowConfidence`
    /// fallback without touching the executor.
    #[allow(dead_code)]
    LowConfidence,
}

#[derive(Debug)]
pub enum RapExecError {
    /// The deadline passed before any step could be tried.
    DeadlinePassed,
    /// All steps tried, none produced items.
    Exhausted,
}

impl std::fmt::Display for RapExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeadlinePassed => write!(f, "RAP deadline passed"),
            Self::Exhausted => write!(f, "RAP exhausted all steps"),
        }
    }
}

impl std::error::Error for RapExecError {}

/// One step's effective outcome from the executor's point of view.
#[derive(Debug)]
pub struct RapOutcome {
    pub items: Vec<RetrievedItem>,
    pub steps_attempted: u32,
    pub final_step_id: String,
    pub cost_estimate_usd: f64,
}

/// Run `rap` against `retriever` for `subquery`, returning the items
/// the first successful step produced. The `min_results` parameter
/// controls the "got enough" cutoff used by spec §5.2 — once the
/// primary step has produced at least this many items, the executor
/// stops; for non-primary steps the first item is enough.
pub async fn rap_execute(
    rap: &ReactiveActionPackage,
    retriever: &dyn Retriever,
    subquery: &SubQuery,
    deadline: Instant,
    min_results: usize,
) -> Result<RapOutcome, RapExecError> {
    if Instant::now() >= deadline {
        return Err(RapExecError::DeadlinePassed);
    }

    let mut items: Vec<RetrievedItem> = Vec::new();
    let mut last: Option<LastOutcome> = None;
    let mut attempts: u32 = 0;
    let mut cost: f64 = 0.0;
    let mut final_step_id = String::new();

    for step in &rap.steps {
        if attempts >= rap.max_total_attempts {
            break;
        }
        if Instant::now() >= deadline {
            break;
        }
        if !should_attempt(step, last) {
            continue;
        }

        attempts += 1;
        cost += step.cost_estimate_usd;
        final_step_id = step.step_id.clone();

        let timeout_budget = Duration::from_millis(step.timeout_ms as u64);
        let remaining = deadline.saturating_duration_since(Instant::now());
        let effective_timeout = std::cmp::min(timeout_budget, remaining);
        if effective_timeout.is_zero() {
            break;
        }

        let call = retriever.call(&step.method, subquery, &step.parameters);
        match tokio::time::timeout(effective_timeout, call).await {
            Ok(Ok(step_items)) => {
                if step_items.is_empty() {
                    last = Some(LastOutcome::Empty);
                } else {
                    let annotated: Vec<RetrievedItem> = step_items
                        .into_iter()
                        .map(|mut it| {
                            it.retrieval_context.rap_step = step.step_id.clone();
                            it.retrieval_context.rap_attempts = attempts;
                            it
                        })
                        .collect();
                    items.extend(annotated);
                    last = Some(LastOutcome::Success);
                    let enough = (step.step_id == "primary" && items.len() >= min_results)
                        || (step.step_id != "primary" && !items.is_empty());
                    if enough {
                        return Ok(RapOutcome {
                            items,
                            steps_attempted: attempts,
                            final_step_id,
                            cost_estimate_usd: cost,
                        });
                    }
                }
            }
            Ok(Err(_e)) => {
                last = Some(LastOutcome::Error);
            }
            Err(_elapsed) => {
                last = Some(LastOutcome::Timeout);
            }
        }
    }

    if items.is_empty() {
        Err(RapExecError::Exhausted)
    } else {
        Ok(RapOutcome {
            items,
            steps_attempted: attempts,
            final_step_id,
            cost_estimate_usd: cost,
        })
    }
}

fn should_attempt(step: &RapStep, last: Option<LastOutcome>) -> bool {
    match step.condition {
        RapStepCondition::Always => true,
        RapStepCondition::OnEmpty => matches!(last, Some(LastOutcome::Empty)),
        RapStepCondition::OnError => matches!(last, Some(LastOutcome::Error)),
        RapStepCondition::OnTimeout => matches!(last, Some(LastOutcome::Timeout)),
        RapStepCondition::OnLowConfidence => matches!(last, Some(LastOutcome::LowConfidence)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retriever::{Retriever, RetrieverError};
    use async_trait::async_trait;
    use chrono::Utc;
    use jouleclaw_schema::{
        Attribution, Content, ExcerptSpan, FreshnessClass, GranularityClass, KnowledgeAxes,
        Modality, RapStep, RapStepCondition, ReactiveActionPackage, RetrievalContext,
        RetrievalMethod, RetrievedItem, ScopeClass, ScoreType, SourceType, SubQuery, Temporal,
        TemporalStabilityClass,
    };
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
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

    fn item(source_id: &str, text: &str) -> RetrievedItem {
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
                excerpt_span: Some(ExcerptSpan { start: 0, end: text.len() }),
            },
            retrieval_context: RetrievalContext {
                retriever_id: "fixture".into(),
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
            knowledge_axes: axes(),
            metadata: Default::default(),
        }
    }

    fn sample_sub() -> SubQuery {
        SubQuery {
            sub_id: "q0".into(),
            text: "what".into(),
            depends_on: vec![],
            required_modalities: vec![Modality::Text],
            target_stores: vec!["fixture".into()],
            priority: 1.0,
            rap_id: "rap0".into(),
        }
    }

    /// Fixture retriever programmable per-method.
    struct ProgrammableRetriever {
        calls: AtomicU32,
        results: std::sync::Mutex<std::collections::HashMap<String, MockBehavior>>,
    }

    #[derive(Clone)]
    enum MockBehavior {
        Items(Vec<RetrievedItem>),
        Error(String),
        Sleep(Duration),
    }

    impl ProgrammableRetriever {
        fn new() -> Self {
            Self {
                calls: AtomicU32::new(0),
                results: std::sync::Mutex::new(Default::default()),
            }
        }
        fn set(&self, method: &str, behavior: MockBehavior) {
            self.results.lock().unwrap().insert(method.into(), behavior);
        }
        fn call_count(&self) -> u32 {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Retriever for ProgrammableRetriever {
        fn retriever_id(&self) -> &str {
            "fixture"
        }
        async fn call(
            &self,
            method: &str,
            _subquery: &SubQuery,
            _parameters: &serde_json::Map<String, serde_json::Value>,
        ) -> Result<Vec<RetrievedItem>, RetrieverError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let behavior = self.results.lock().unwrap().get(method).cloned();
            match behavior {
                Some(MockBehavior::Items(v)) => Ok(v),
                Some(MockBehavior::Error(e)) => Err(RetrieverError::Backend(e)),
                Some(MockBehavior::Sleep(d)) => {
                    tokio::time::sleep(d).await;
                    Ok(vec![])
                }
                None => Err(RetrieverError::UnknownMethod(method.into())),
            }
        }
    }

    fn rap_two_steps() -> ReactiveActionPackage {
        ReactiveActionPackage {
            schema_version: "2.0".into(),
            rap_id: "rap0".into(),
            description: "primary + on_empty fallback".into(),
            applies_to_methods: vec![RetrievalMethod::Sparql],
            steps: vec![
                RapStep {
                    step_id: "primary".into(),
                    description: "primary".into(),
                    method: "m_primary".into(),
                    condition: RapStepCondition::Always,
                    parameters: Default::default(),
                    timeout_ms: 1000,
                    cost_estimate_usd: 0.0,
                },
                RapStep {
                    step_id: "fallback".into(),
                    description: "fallback".into(),
                    method: "m_fallback".into(),
                    condition: RapStepCondition::OnEmpty,
                    parameters: Default::default(),
                    timeout_ms: 1000,
                    cost_estimate_usd: 0.0,
                },
            ],
            max_total_attempts: 3,
            metadata: Default::default(),
        }
    }

    #[tokio::test]
    async fn primary_success_stops_after_primary() {
        let r = Arc::new(ProgrammableRetriever::new());
        r.set("m_primary", MockBehavior::Items(vec![item("a", "Paris")]));
        r.set("m_fallback", MockBehavior::Items(vec![item("b", "x")]));
        let rap = rap_two_steps();
        let sub = sample_sub();
        let outcome = rap_execute(
            &rap,
            &*r,
            &sub,
            Instant::now() + Duration::from_secs(5),
            1,
        )
        .await
        .unwrap();
        assert_eq!(outcome.steps_attempted, 1);
        assert_eq!(outcome.final_step_id, "primary");
        assert_eq!(outcome.items.len(), 1);
        assert_eq!(r.call_count(), 1);
    }

    #[tokio::test]
    async fn on_empty_advances_to_fallback() {
        let r = Arc::new(ProgrammableRetriever::new());
        r.set("m_primary", MockBehavior::Items(vec![]));
        r.set("m_fallback", MockBehavior::Items(vec![item("b", "Paris")]));
        let rap = rap_two_steps();
        let outcome =
            rap_execute(&rap, &*r, &sample_sub(), Instant::now() + Duration::from_secs(5), 1)
                .await
                .unwrap();
        assert_eq!(outcome.steps_attempted, 2);
        assert_eq!(outcome.final_step_id, "fallback");
        assert_eq!(outcome.items.len(), 1);
        assert_eq!(outcome.items[0].retrieval_context.rap_step, "fallback");
        assert_eq!(outcome.items[0].retrieval_context.rap_attempts, 2);
    }

    #[tokio::test]
    async fn on_error_does_not_match_on_empty_condition() {
        let r = Arc::new(ProgrammableRetriever::new());
        r.set("m_primary", MockBehavior::Error("backend down".into()));
        r.set("m_fallback", MockBehavior::Items(vec![item("b", "x")]));
        let rap = rap_two_steps();
        let res = rap_execute(
            &rap,
            &*r,
            &sample_sub(),
            Instant::now() + Duration::from_secs(5),
            1,
        )
        .await;
        // Primary errored; fallback is OnEmpty (not OnError), so it
        // shouldn't fire. The RAP exhausts with no items.
        assert!(matches!(res, Err(RapExecError::Exhausted)));
    }

    #[tokio::test]
    async fn timeout_passes_through() {
        let r = Arc::new(ProgrammableRetriever::new());
        // m_primary sleeps longer than the step's 1000 ms budget.
        r.set("m_primary", MockBehavior::Sleep(Duration::from_secs(3)));
        r.set("m_fallback", MockBehavior::Items(vec![item("b", "Paris")]));
        let rap = rap_two_steps();
        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            rap_execute(
                &rap,
                &*r,
                &sample_sub(),
                Instant::now() + Duration::from_secs(5),
                1,
            ),
        )
        .await
        .unwrap();
        // m_primary timed out (Timeout outcome) but fallback condition
        // is OnEmpty, not OnTimeout, so fallback doesn't fire and the
        // RAP exhausts. This is the spec's behavior: each fallback
        // explicitly opts in to which previous outcome triggers it.
        assert!(matches!(outcome, Err(RapExecError::Exhausted)));
    }

    #[tokio::test]
    async fn deadline_in_the_past_short_circuits() {
        let r = Arc::new(ProgrammableRetriever::new());
        r.set("m_primary", MockBehavior::Items(vec![item("a", "x")]));
        let rap = rap_two_steps();
        let res = rap_execute(
            &rap,
            &*r,
            &sample_sub(),
            Instant::now() - Duration::from_secs(1),
            1,
        )
        .await;
        assert!(matches!(res, Err(RapExecError::DeadlinePassed)));
        assert_eq!(r.call_count(), 0);
    }
}
