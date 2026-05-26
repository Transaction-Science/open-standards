//! Reactive Action Packages (spec §3.5).
//!
//! A RAP describes a sequence of attempts for a single action, with
//! explicit conditions for falling back. Every retriever, scorer, and
//! composer ships with a RAP (commitment C3). The RAP executor in
//! `jouleclaw-execute` walks the steps and stops at the first success.

use serde::{Deserialize, Serialize};

use crate::common::Metadata;
use crate::retrieved_item::RetrievalMethod;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RapStepCondition {
    /// Always try this step (typically the primary).
    Always,
    /// Try if the previous step returned no results.
    OnEmpty,
    /// Try if the previous step errored.
    OnError,
    /// Try if the previous step's results were low-confidence.
    OnLowConfidence,
    /// Try if the previous step timed out.
    OnTimeout,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RapStep {
    /// Stable identifier — e.g. `"primary"`, `"fallback_predicate"`,
    /// `"text_search"`. Surfaces in
    /// `RetrievedItem.retrieval_context.rap_step`.
    pub step_id: String,
    pub description: String,
    /// Method name the RAP executor will invoke on the retriever.
    pub method: String,
    pub condition: RapStepCondition,
    /// Free-form per-step parameters, opaque to the executor.
    #[serde(default)]
    pub parameters: serde_json::Map<String, serde_json::Value>,
    pub timeout_ms: u32,
    pub cost_estimate_usd: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReactiveActionPackage {
    pub schema_version: String,
    /// Stable identifier — referenced from
    /// [`crate::query_plan::SubQuery::rap_id`].
    pub rap_id: String,
    pub description: String,
    pub applies_to_methods: Vec<RetrievalMethod>,
    pub steps: Vec<RapStep>,
    pub max_total_attempts: u32,
    #[serde(default)]
    pub metadata: Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rap() -> ReactiveActionPackage {
        ReactiveActionPackage {
            schema_version: "2.0".into(),
            rap_id: "wikidata_sparql_v1".into(),
            description: "Wikidata SPARQL with predicate-search fallback".into(),
            applies_to_methods: vec![RetrievalMethod::Sparql],
            steps: vec![
                RapStep {
                    step_id: "primary".into(),
                    description: "Exact label match".into(),
                    method: "sparql_label_match".into(),
                    condition: RapStepCondition::Always,
                    parameters: Default::default(),
                    timeout_ms: 4000,
                    cost_estimate_usd: 0.0,
                },
                RapStep {
                    step_id: "fallback_predicate".into(),
                    description: "Search by P31 = ...".into(),
                    method: "sparql_predicate_search".into(),
                    condition: RapStepCondition::OnEmpty,
                    parameters: Default::default(),
                    timeout_ms: 6000,
                    cost_estimate_usd: 0.0,
                },
            ],
            max_total_attempts: 3,
            metadata: Default::default(),
        }
    }

    #[test]
    fn roundtrips_through_json() {
        let r = sample_rap();
        let json = serde_json::to_string(&r).unwrap();
        let back: ReactiveActionPackage = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
