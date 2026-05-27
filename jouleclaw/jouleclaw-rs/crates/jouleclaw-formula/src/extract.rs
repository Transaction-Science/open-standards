//! Entity extraction from raw query text.
//!
//! The donor's algorithm: tokenise on whitespace + light punctuation, then
//! probe the knowledge store with progressively shorter n-grams
//! (3-gram → 2-gram → 1-gram). For 1-grams, skip a fixed stop-word list to
//! cut false positives ("the formula" matching a concept called "the").
//!
//! Kept verbatim from the donor: the n-gram order, the candidate filtering,
//! and the stop-word list. Inlined here so this crate stays self-contained.

use crate::knowledge::{Concept, KnowledgeStore};

/// Result of extracting entities from a query.
#[derive(Debug, Clone)]
pub struct Extraction {
    /// `(candidate_text, concept)` pairs in extraction order.
    pub matches: Vec<(String, Concept)>,
    /// Raw word tokens — used by the NCD zero-shot fallback when no
    /// entities resolved.
    pub words: Vec<String>,
}

/// Run the donor's entity-extraction algorithm against a [`KnowledgeStore`].
///
/// The store's `search_by_name` SHOULD be substring-friendly so partial
/// matches like "boiling" → "boiling point" still resolve. We additionally
/// require either:
///
/// - the candidate is a substring of the concept name, OR
/// - the candidate is a substring of the concept id, OR
/// - the concept name is a substring of the candidate.
///
/// This stops the search from returning spurious matches for very short
/// candidates ("a" matching some concept whose name happens to contain "a").
pub fn extract_entities<K: KnowledgeStore + ?Sized>(
    query: &str,
    knowledge: &K,
) -> Extraction {
    let words: Vec<String> = query
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == '?')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect();

    let mut matched: Vec<(String, Concept)> = Vec::new();

    for window in (1..=3usize).rev() {
        if window > words.len() {
            continue;
        }
        for ngram in words.windows(window) {
            let candidate = ngram
                .iter()
                .map(|w| {
                    w.trim_matches(|c: char| !c.is_alphanumeric())
                        .to_lowercase()
                })
                .collect::<Vec<_>>()
                .join(" ");

            if candidate.len() < 2 {
                continue;
            }
            if window == 1 && is_stop_word(&candidate) {
                continue;
            }

            let results = knowledge.search_by_name(&candidate, 1);
            if let Some(concept) = results.first() {
                let name_lower = concept.name.to_lowercase();
                let id_lower = concept.id.to_lowercase();
                if name_lower.contains(&candidate)
                    || id_lower.contains(&candidate)
                    || candidate.contains(&name_lower)
                {
                    if !matched.iter().any(|(_, c)| c.id == concept.id) {
                        matched.push((candidate, concept.clone()));
                    }
                }
            }
        }
    }

    Extraction { matches: matched, words }
}

/// Donor stop-word list. Kept verbatim — these are the words that, on their
/// own, are most likely to be false positives against a concept-name index.
///
/// Includes both standard English function words AND a small set of generic
/// nouns ("number", "set", "function", …) that match too many concepts to
/// be useful as 1-gram entity candidates.
pub fn is_stop_word(word: &str) -> bool {
    matches!(
        word,
        "the" | "a" | "an" | "is" | "are" | "was" | "were" | "be" | "been"
            | "being" | "have" | "has" | "had" | "do" | "does" | "did"
            | "will" | "would" | "shall" | "should" | "may" | "might"
            | "must" | "can" | "could" | "of" | "in" | "to" | "for"
            | "with" | "on" | "at" | "from" | "by" | "about" | "as"
            | "into" | "through" | "during" | "before" | "after" | "above"
            | "below" | "between" | "out" | "off" | "over" | "under"
            | "again" | "further" | "then" | "once" | "here" | "there"
            | "when" | "where" | "why" | "how" | "all" | "each" | "every"
            | "both" | "few" | "more" | "most" | "other" | "some" | "such"
            | "no" | "not" | "only" | "own" | "same" | "so" | "than"
            | "too" | "very" | "just" | "because" | "but" | "and" | "or"
            | "if" | "while" | "that" | "what" | "which" | "who" | "whom"
            | "this" | "these" | "those" | "it" | "its" | "his" | "her"
            | "he" | "she" | "they" | "them" | "we" | "us" | "me" | "my"
            | "your" | "tell" | "describe" | "explain" | "compare"
            | "versus" | "vs" | "relate" | "related"
            | "relationship" | "similar" | "different" | "difference"
            // Generic words that match concept names but produce false positives
            // in factual queries (e.g., "What is Avogadro's number?")
            | "number" | "set" | "function" | "variable" | "proof"
            | "graph" | "map" | "key" | "bridge" | "table" | "point"
            | "system" | "model" | "process" | "state" | "type"
            | "continent" | "many" | "much"
    )
}

/// Check if a query is primarily about a specific entity (structural /
/// definitional). Donor logic, preserved.
pub fn query_is_about_entity(query: &str, entity_name: &str) -> bool {
    let q = query.to_lowercase();
    let e = entity_name.to_lowercase();

    if !q.contains(&e) {
        return false;
    }

    let word_count = q.split_whitespace().count();
    if word_count <= 3 {
        return true;
    }

    let is_what_is = q.starts_with("what is") || q.starts_with("what are");
    if is_what_is && word_count <= 5 {
        return true;
    }

    if (q.starts_with("tell me about") || q.starts_with("describe"))
        && word_count <= 5
    {
        return true;
    }
    if q.starts_with("explain") && !q.contains("how") && word_count <= 4 {
        return true;
    }

    false
}

/// Whether a query is a short structural comparison the formula can fully
/// answer. Donor logic, preserved.
pub fn is_structural_query(query: &str) -> bool {
    let q = query.to_lowercase();
    let word_count = q.split_whitespace().count();
    if word_count > 6 {
        return false;
    }
    q.contains("compare")
        || q.contains("contrast")
        || q.contains("similar")
        || q.contains("different")
        || q.contains("versus")
        || q.contains(" vs ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{Concept, InMemoryKnowledgeStore};

    fn store() -> InMemoryKnowledgeStore {
        let mut k = InMemoryKnowledgeStore::new();
        k.insert(Concept {
            id: "urn:fire".into(),
            name: "fire".into(),
            traits: vec![1.0],
        });
        k.insert(Concept {
            id: "urn:water".into(),
            name: "water".into(),
            traits: vec![-1.0],
        });
        k
    }

    #[test]
    fn extracts_two_entities_from_compare_query() {
        let e = extract_entities("compare fire and water", &store());
        assert_eq!(e.matches.len(), 2);
    }

    #[test]
    fn skips_pure_stop_word_query() {
        let e = extract_entities("what is the", &store());
        assert!(e.matches.is_empty());
    }

    #[test]
    fn structural_query_detected() {
        assert!(is_structural_query("compare fire and water"));
        assert!(!is_structural_query(
            "What is the relationship between fire and water in chemistry?"
        ));
    }

    #[test]
    fn entity_focused_query_detected() {
        assert!(query_is_about_entity("fire", "fire"));
        assert!(query_is_about_entity("what is fire?", "fire"));
        assert!(!query_is_about_entity(
            "what is the boiling point of water in celsius",
            "water"
        ));
    }
}
