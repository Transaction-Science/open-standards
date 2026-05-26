//! Query Understanding (spec §4.1).
//!
//! Production implementations call out to a reasoner (joule cascade
//! L3, frontier API, …) and parse a structured response. The trait
//! decouples the planner from any single backend so the deployment
//! can swap models freely. A [`FixtureUnderstanding`] is provided for
//! tests and deterministic acceptance runs.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::analysis::QueryAnalysis;
use jouleclaw_schema::OriginalQuery;

#[derive(Debug)]
pub enum UnderstandingError {
    /// Reasoner returned content the parser couldn't interpret.
    ParseFailed(String),
    /// Reasoner was unreachable / errored.
    Backend(String),
    /// The fixture has no canned analysis for this query.
    UnknownFixture(String),
}

impl std::fmt::Display for UnderstandingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseFailed(s) => write!(f, "parse failed: {s}"),
            Self::Backend(s) => write!(f, "backend error: {s}"),
            Self::UnknownFixture(s) => write!(f, "no fixture for query: {s}"),
        }
    }
}

impl std::error::Error for UnderstandingError {}

pub trait QueryUnderstanding: Send + Sync {
    /// Analyze `query` and return a structured [`QueryAnalysis`]. The
    /// returned analysis is the input to the constraint planner.
    fn analyze(&self, query: &OriginalQuery) -> Result<QueryAnalysis, UnderstandingError>;
}

/// Canned-analysis implementation. Keyed by the trimmed lowercase
/// query text. Use for deterministic tests and acceptance runs that
/// must not depend on a running reasoner.
pub struct FixtureUnderstanding {
    fixtures: Mutex<HashMap<String, QueryAnalysis>>,
}

impl FixtureUnderstanding {
    pub fn new() -> Self {
        Self {
            fixtures: Mutex::new(HashMap::new()),
        }
    }

    pub fn insert(&self, query_text: &str, analysis: QueryAnalysis) {
        self.fixtures
            .lock()
            .unwrap()
            .insert(normalize_key(query_text), analysis);
    }
}

impl Default for FixtureUnderstanding {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryUnderstanding for FixtureUnderstanding {
    fn analyze(&self, query: &OriginalQuery) -> Result<QueryAnalysis, UnderstandingError> {
        let key = match query.text.as_deref() {
            Some(t) => normalize_key(t),
            None => {
                return Err(UnderstandingError::UnknownFixture(
                    "(non-text query)".into(),
                ));
            }
        };
        self.fixtures
            .lock()
            .unwrap()
            .get(&key)
            .cloned()
            .ok_or_else(|| UnderstandingError::UnknownFixture(key))
    }
}

fn normalize_key(s: &str) -> String {
    s.trim().to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{RawSubQuery, StakesSignal};
    use chrono::Utc;
    use jouleclaw_schema::{Intent, Modality};

    fn mk_query(text: &str) -> OriginalQuery {
        OriginalQuery {
            text: Some(text.into()),
            image_ref: None,
            audio_ref: None,
            video_ref: None,
            language_detected: "en".into(),
            timestamp: Utc::now(),
        }
    }

    fn mk_analysis(text: &str) -> QueryAnalysis {
        QueryAnalysis {
            original_query: mk_query(text),
            intent: Intent::Lookup,
            modalities_in: vec![Modality::Text],
            modalities_out: vec![Modality::Text],
            entities_extracted: vec![],
            relations_extracted: vec![],
            temporal_anchors: vec![],
            geographic_anchors: vec![],
            domain_tags: vec![],
            freshness_signal: false,
            stakes_signal: StakesSignal::Low,
            raw_decomposition: vec![RawSubQuery {
                sub_id: "q0".into(),
                text: text.into(),
                required_modalities: vec![Modality::Text],
                depends_on: vec![],
                priority: 1.0,
                preferred_store: None,
            }],
            confidence: 1.0,
        }
    }

    #[test]
    fn fixture_returns_canned_analysis() {
        let f = FixtureUnderstanding::new();
        f.insert("capital of france", mk_analysis("capital of france"));
        let q = mk_query("Capital of France");
        let a = f.analyze(&q).unwrap();
        assert_eq!(a.raw_decomposition.len(), 1);
    }

    #[test]
    fn unknown_fixture_errors() {
        let f = FixtureUnderstanding::new();
        let q = mk_query("something else");
        let err = f.analyze(&q).unwrap_err();
        assert!(matches!(err, UnderstandingError::UnknownFixture(_)));
    }
}
