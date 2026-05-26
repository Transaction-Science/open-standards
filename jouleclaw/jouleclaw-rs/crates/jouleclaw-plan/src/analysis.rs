//! [`QueryAnalysis`] — the structured output of Query Understanding
//! (spec §4.1). Not a `QueryPlan`; the planner consumes this and
//! produces the plan via CSP.

use serde::{Deserialize, Serialize};

use jouleclaw_schema::{Intent, Modality, OriginalQuery};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StakesSignal {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    /// Free-form tag — e.g. `"PERSON"`, `"COUNTRY"`, `"PRODUCT"`.
    pub kind: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Relation {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawSubQuery {
    /// Stable id assigned by the analyzer (e.g. `"q0"`, `"q1"`).
    pub sub_id: String,
    /// Natural-language sub-question text.
    pub text: String,
    /// Modalities required to answer this sub-question. Drives the
    /// CSP's modality constraint (§4.2).
    pub required_modalities: Vec<Modality>,
    /// Sub-ids this sub-query depends on (must execute first).
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// 0.0..=1.0; planner uses this to break solver ties.
    #[serde(default = "default_priority")]
    pub priority: f64,
    /// Optional retriever-id hint. When `Some(id)`, the planner
    /// constrains this sub-query to that store (subject to it
    /// being healthy + present in the StoreCatalog); other
    /// candidates are dropped. When `None`, the CSP picks any
    /// modality-matching healthy store. Used by the CLI to
    /// dispatch parallel sub-queries to specific retrievers
    /// (Wikidata vs Wikipedia) without re-inventing the planner.
    #[serde(default)]
    pub preferred_store: Option<String>,
}

fn default_priority() -> f64 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryAnalysis {
    pub original_query: OriginalQuery,
    pub intent: Intent,
    pub modalities_in: Vec<Modality>,
    pub modalities_out: Vec<Modality>,
    pub entities_extracted: Vec<Entity>,
    pub relations_extracted: Vec<Relation>,
    pub temporal_anchors: Vec<String>,
    pub geographic_anchors: Vec<String>,
    pub domain_tags: Vec<String>,
    /// True iff the query requires current information (§0.6 Rule 3
    /// triggers).
    pub freshness_signal: bool,
    pub stakes_signal: StakesSignal,
    /// Sub-questions in natural language, with the metadata the
    /// planner needs to assign stores.
    pub raw_decomposition: Vec<RawSubQuery>,
    /// Analyzer's self-reported confidence in the extraction. The
    /// planner can refuse low-confidence analyses.
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn roundtrips_through_json() {
        let a = QueryAnalysis {
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
            entities_extracted: vec![Entity {
                name: "France".into(),
                kind: "COUNTRY".into(),
                confidence: 0.99,
            }],
            relations_extracted: vec![],
            temporal_anchors: vec![],
            geographic_anchors: vec!["France".into()],
            domain_tags: vec!["geography".into()],
            freshness_signal: false,
            stakes_signal: StakesSignal::Low,
            raw_decomposition: vec![RawSubQuery {
                sub_id: "q0".into(),
                text: "What is the capital of France?".into(),
                required_modalities: vec![Modality::Text],
                depends_on: vec![],
                priority: 1.0,
                preferred_store: None,
            }],
            confidence: 0.95,
        };
        let json = serde_json::to_string(&a).unwrap();
        let back: QueryAnalysis = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }
}
