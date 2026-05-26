//! Heuristic query understanding (spec §4.1, MVP shape).
//!
//! Turns a user query string into a [`QueryAnalysis`] without an
//! LLM. The full §4.1 path is LLM-based; this is the floor that
//! lets the CLI run any English query that fits the property-path
//! patterns or stands as a bare entity lookup.
//!
//! Behavior:
//!   - Strip "what is the / what is / what's the / who is / where
//!     is" framings + trailing "?". The remainder becomes the
//!     sub-query decomposition text.
//!   - Detect intent from leading wh-word ("who" → Lookup w/ person
//!     hint via metadata, "where" → geographic anchor, etc.).
//!   - Single text modality; single sub-query (no decomposition);
//!     no entity extraction beyond what the retriever does.
//!
//! An LLM-backed impl can replace this trivially via the
//! [`jouleclaw_plan::QueryUnderstanding`] trait.

use chrono::Utc;

use jouleclaw_plan::{
    QueryAnalysis, QueryUnderstanding, RawSubQuery, StakesSignal, UnderstandingError,
};
use jouleclaw_schema::{Intent, Modality, OriginalQuery};

/// Rule-based [`QueryUnderstanding`] that does prefix-stripping
/// + lightweight intent classification.
pub struct HeuristicUnderstanding;

impl HeuristicUnderstanding {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HeuristicUnderstanding {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryUnderstanding for HeuristicUnderstanding {
    fn analyze(&self, query: &OriginalQuery) -> Result<QueryAnalysis, UnderstandingError> {
        let text = query
            .text
            .as_deref()
            .ok_or_else(|| UnderstandingError::Backend("query has no text".into()))?
            .trim();
        if text.is_empty() {
            return Err(UnderstandingError::Backend("empty query".into()));
        }

        let intent = classify_intent(text);
        let decomposition_text = strip_question_framing(text);
        let geographic_anchors = guess_geographic_anchors(text);
        let stakes = guess_stakes(text);

        // Emit two parallel sub-queries — structured (Wikidata) +
        // prose (Wikipedia) — so the diagnose pillar gets
        // independent sources to cross-check. Both run in the
        // same dependency level so the orchestrator dispatches
        // them concurrently.
        Ok(QueryAnalysis {
            original_query: OriginalQuery {
                text: Some(text.to_string()),
                image_ref: query.image_ref.clone(),
                audio_ref: query.audio_ref.clone(),
                video_ref: query.video_ref.clone(),
                language_detected: query.language_detected.clone(),
                timestamp: query.timestamp,
            },
            intent,
            modalities_in: vec![Modality::Text],
            modalities_out: vec![Modality::Text],
            entities_extracted: vec![],
            relations_extracted: vec![],
            temporal_anchors: vec![],
            geographic_anchors,
            domain_tags: vec![],
            freshness_signal: needs_freshness(text),
            stakes_signal: stakes,
            raw_decomposition: vec![
                RawSubQuery {
                    sub_id: "wd".into(),
                    text: decomposition_text.clone(),
                    required_modalities: vec![Modality::Text],
                    depends_on: vec![],
                    priority: 1.0,
                    preferred_store: Some("wikidata".into()),
                },
                RawSubQuery {
                    sub_id: "wp".into(),
                    text: decomposition_text,
                    required_modalities: vec![Modality::Text],
                    depends_on: vec![],
                    priority: 0.8,
                    preferred_store: Some("wikipedia".into()),
                },
            ],
            confidence: 0.7,
        })
    }
}

/// Convenience helper: directly produce a [`QueryAnalysis`] for a
/// raw query string. Used by the pipeline runner.
pub fn analyze_query(query_text: &str) -> Result<QueryAnalysis, UnderstandingError> {
    let understanding = HeuristicUnderstanding::new();
    understanding.analyze(&OriginalQuery {
        text: Some(query_text.to_string()),
        image_ref: None,
        audio_ref: None,
        video_ref: None,
        language_detected: "en".into(),
        timestamp: Utc::now(),
    })
}

fn strip_question_framing(text: &str) -> String {
    let trimmed = text.trim().trim_end_matches(['?', '.', '!', ' ']);
    let lc = trimmed.to_lowercase();
    const PREFIXES: &[&str] = &[
        "what is the ",
        "what is ",
        "what's the ",
        "what's ",
        "who is the ",
        "who is ",
        "where is the ",
        "where is ",
        "tell me the ",
        "tell me ",
        "give me the ",
        "give me ",
    ];
    for p in PREFIXES {
        if lc.starts_with(p) {
            return trimmed[p.len()..].trim().to_string();
        }
    }
    trimmed.to_string()
}

fn classify_intent(text: &str) -> Intent {
    let lc = text.to_lowercase();
    if lc.starts_with("compare") || lc.contains(" vs ") || lc.contains(" versus ") {
        Intent::Comparison
    } else if lc.starts_with("which should") || lc.starts_with("should i") {
        Intent::Recommendation
    } else if lc.starts_with("how many") || lc.starts_with("count") {
        Intent::Aggregation
    } else if lc.starts_with("write ") || lc.starts_with("generate ") {
        Intent::Generation
    } else if lc.contains('?')
        || lc.starts_with("what")
        || lc.starts_with("who")
        || lc.starts_with("where")
        || lc.starts_with("when")
    {
        Intent::Lookup
    } else {
        Intent::Lookup
    }
}

fn guess_geographic_anchors(text: &str) -> Vec<String> {
    // Crude: pick out capitalized multi-word noun phrases. The
    // retriever does proper entity resolution downstream; this is
    // just an annotation for the planner / authority scorer to
    // optionally weight.
    let mut out = Vec::new();
    let mut buf = String::new();
    for word in text.split_whitespace() {
        let w = word.trim_matches(|c: char| !c.is_alphanumeric());
        if !w.is_empty()
            && w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
        {
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(w);
        } else if !buf.is_empty() {
            out.push(std::mem::take(&mut buf));
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

fn needs_freshness(text: &str) -> bool {
    let lc = text.to_lowercase();
    const FRESHNESS_MARKERS: &[&str] = &[
        "current",
        "currently",
        "now",
        "latest",
        "today",
        "this year",
        "right now",
        "as of",
    ];
    FRESHNESS_MARKERS.iter().any(|m| lc.contains(m))
}

fn guess_stakes(text: &str) -> StakesSignal {
    let lc = text.to_lowercase();
    for needle in ["dosage", "diagnose", "diagnosis", "prescribe", "lethal"] {
        if lc.contains(needle) {
            return StakesSignal::High;
        }
    }
    for needle in ["should i", "i should", "buy", "deploy", "use this"] {
        if lc.contains(needle) {
            return StakesSignal::High;
        }
    }
    StakesSignal::Low
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_what_is_the_prefix() {
        assert_eq!(
            strip_question_framing("What is the capital of France?"),
            "capital of France"
        );
        assert_eq!(
            strip_question_framing("What is the currency of Japan?"),
            "currency of Japan"
        );
    }

    #[test]
    fn strips_whos_and_wheres_too() {
        assert_eq!(
            strip_question_framing("Who is the president of Brazil?"),
            "president of Brazil"
        );
        assert_eq!(
            strip_question_framing("Where is the Eiffel Tower?"),
            "Eiffel Tower"
        );
    }

    #[test]
    fn bare_entity_passes_through() {
        assert_eq!(strip_question_framing("Albert Einstein"), "Albert Einstein");
        assert_eq!(strip_question_framing("Paris"), "Paris");
    }

    #[test]
    fn handles_trailing_punctuation() {
        assert_eq!(strip_question_framing("Paris!"), "Paris");
        assert_eq!(strip_question_framing("the capital. "), "the capital");
    }

    #[test]
    fn intent_classified() {
        assert!(matches!(classify_intent("What is X?"), Intent::Lookup));
        assert!(matches!(
            classify_intent("compare X and Y"),
            Intent::Comparison
        ));
        assert!(matches!(
            classify_intent("X vs Y"),
            Intent::Comparison
        ));
        assert!(matches!(
            classify_intent("how many states are there?"),
            Intent::Aggregation
        ));
        assert!(matches!(
            classify_intent("Should I use this library?"),
            Intent::Recommendation
        ));
    }

    #[test]
    fn detects_freshness_markers() {
        assert!(needs_freshness("what is the current CEO of Google?"));
        assert!(needs_freshness("latest version of Rust"));
        assert!(!needs_freshness("when did Marie Curie win a Nobel?"));
    }

    #[test]
    fn detects_high_stakes_medical() {
        assert!(matches!(
            guess_stakes("what is the dosage of aspirin?"),
            StakesSignal::High
        ));
    }

    #[test]
    fn analyze_query_returns_lookup_with_decomposition() {
        let a = analyze_query("What is the capital of France?").unwrap();
        assert!(matches!(a.intent, Intent::Lookup));
        // HeuristicUnderstanding emits parallel sub-queries: one
        // for Wikidata (structured), one for Wikipedia (prose).
        assert_eq!(a.raw_decomposition.len(), 2);
        for sq in &a.raw_decomposition {
            assert_eq!(sq.text, "capital of France");
        }
        assert!(a.raw_decomposition.iter().any(|s| s.preferred_store.as_deref() == Some("wikidata")));
        assert!(a.raw_decomposition.iter().any(|s| s.preferred_store.as_deref() == Some("wikipedia")));
    }

    #[test]
    fn analyze_query_handles_bare_entity() {
        let a = analyze_query("Paris").unwrap();
        assert_eq!(a.raw_decomposition[0].text, "Paris");
    }

    #[test]
    fn analyze_query_picks_up_geographic_anchor() {
        let a = analyze_query("What is the capital of France?").unwrap();
        assert!(a.geographic_anchors.iter().any(|g| g == "France"));
    }

    #[test]
    fn empty_query_errors() {
        assert!(analyze_query("").is_err());
        assert!(analyze_query("   ").is_err());
    }
}
