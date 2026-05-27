//! Query decomposition.

use jouleclaw_cascade::types::{Query, QueryInput};

/// A single decomposed step. Carries its own text; the agent wraps it
/// back into a full [`Query`] (inheriting the parent's budget/quality)
/// before dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubQuery {
    pub text: String,
}

impl SubQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Turns a query into an ordered list of sub-queries. A planner that
/// sees no decomposition returns a single sub-query equal to the input
/// (the agent then behaves as a pass-through).
pub trait AgentPlanner: Send + Sync {
    fn plan(&self, q: &Query) -> Vec<SubQuery>;
}

/// Splits on coordinating conjunctions and separators: " and ",
/// " then ", ";". Case-insensitive on the conjunctions. Empty fragments
/// are dropped; a query with no separators yields one sub-query.
#[derive(Debug, Default, Clone, Copy)]
pub struct KeywordPlanner;

impl KeywordPlanner {
    /// Lowercased separators, longest first so " and then " splits
    /// cleanly. We split on word-boundary conjunctions and the literal
    /// ";".
    fn split(text: &str) -> Vec<String> {
        // Normalise separators to a single sentinel, then split.
        let lower = text.to_lowercase();
        // Find split points without destroying original casing: we work
        // on byte indices over the lowercased copy, which is 1:1 for the
        // ASCII separators we look for.
        let mut pieces: Vec<String> = Vec::new();
        let mut start = 0usize;
        let bytes = lower.as_bytes();
        let seps = [" and then ", " then ", " and ", "; ", ";"];
        let mut i = 0usize;
        while i < bytes.len() {
            let mut matched = None;
            for sep in &seps {
                if lower[i..].starts_with(sep) {
                    matched = Some(sep.len());
                    break;
                }
            }
            if let Some(seplen) = matched {
                if i > start {
                    pieces.push(text[start..i].trim().to_string());
                }
                i += seplen;
                start = i;
            } else {
                i += 1;
            }
        }
        if start < text.len() {
            pieces.push(text[start..].trim().to_string());
        }
        pieces.retain(|p| !p.is_empty());
        pieces
    }
}

impl AgentPlanner for KeywordPlanner {
    fn plan(&self, q: &Query) -> Vec<SubQuery> {
        let text = match &q.input {
            QueryInput::Text(t) => t.as_str(),
            QueryInput::Multimodal { text, .. } => text.as_str(),
            _ => return Vec::new(),
        };
        let pieces = Self::split(text);
        if pieces.is_empty() {
            return Vec::new();
        }
        if pieces.len() == 1 {
            return vec![SubQuery::new(pieces.into_iter().next().unwrap_or_default())];
        }
        pieces.into_iter().map(SubQuery::new).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{ContextRef, JouleBudget, QualityFloor};

    fn q(text: &str) -> Query {
        Query {
            input: QueryInput::Text(text.to_string()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn single_clause_is_one_subquery() {
        let p = KeywordPlanner;
        let subs = p.plan(&q("what is the capital of france"));
        assert_eq!(subs.len(), 1);
    }

    #[test]
    fn and_splits_two() {
        let p = KeywordPlanner;
        let subs = p.plan(&q("capital of france and population of germany"));
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0], SubQuery::new("capital of france"));
        assert_eq!(subs[1], SubQuery::new("population of germany"));
    }

    #[test]
    fn then_splits() {
        let p = KeywordPlanner;
        let subs = p.plan(&q("compute 2+2 then square the result"));
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn semicolon_splits() {
        let p = KeywordPlanner;
        let subs = p.plan(&q("step one; step two; step three"));
        assert_eq!(subs.len(), 3);
    }

    #[test]
    fn and_then_splits_once() {
        let p = KeywordPlanner;
        let subs = p.plan(&q("do a and then do b"));
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0], SubQuery::new("do a"));
        assert_eq!(subs[1], SubQuery::new("do b"));
    }

    #[test]
    fn non_text_yields_nothing() {
        let p = KeywordPlanner;
        let query = Query {
            input: QueryInput::Binary(vec![1]),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(p.plan(&query).is_empty());
    }

    #[test]
    fn empty_text_yields_nothing() {
        let p = KeywordPlanner;
        assert!(p.plan(&q("   ")).is_empty());
    }
}
