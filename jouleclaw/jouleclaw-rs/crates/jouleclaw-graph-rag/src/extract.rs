//! Deterministic entity extraction from raw text.
//!
//! The donor (`verity-cascade::l125_graph_rag`) uses three pattern classes:
//! capitalized multi-word proper nouns, CamelCase / snake-case / kebab-case
//! technical terms, and quantity-with-unit pairs. The implementation here
//! is a verbatim port — no LLM, no ML, just byte-level heuristics.

/// Coarse entity class inferred by [`extract_entity_candidates`].
///
/// Mirrors the donor's `EntityClass`. Distinct from the [`crate::graph::Entity`]
/// shape so consumers can distinguish "the extractor proposed this" from
/// "the knowledge graph confirmed this".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EntityClass {
    /// Capitalized multi-word phrase (proper noun / named entity).
    ProperNoun,
    /// Technical term — CamelCase, snake_case, or kebab-case identifier.
    Technical,
    /// Numeric quantity with unit (e.g. `344 µJ`, `10 ms`).
    Quantity,
}

impl EntityClass {
    /// Short tag used in prose summaries.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::ProperNoun => "entity",
            Self::Technical => "tech",
            Self::Quantity => "quantity",
        }
    }
}

/// One extraction candidate — the surface form and the class the heuristic
/// assigned. The runtime then resolves the surface form against a
/// [`crate::graph::KnowledgeGraph`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EntityCandidate {
    /// Canonical surface form (whitespace-normalised, punctuation trimmed).
    pub surface: String,
    /// Class inferred by the extractor.
    pub class: EntityClass,
}

/// Extract entity candidates from text using deterministic heuristics.
///
/// No LLM, no ML — pure pattern matching. The output may contain duplicates;
/// callers that care about uniqueness should dedupe by surface form.
pub fn extract_entity_candidates(text: &str) -> Vec<EntityCandidate> {
    let mut out = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    // Pattern 1: Multi-word capitalized sequences.
    let mut i = 0;
    while i < words.len() {
        if is_capitalized(words[i]) && !is_stopword(words[i]) {
            let start = i;
            while i + 1 < words.len() && is_capitalized(words[i + 1]) {
                i += 1;
            }
            if i > start {
                let surface: String = words[start..=i]
                    .iter()
                    .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                    .collect::<Vec<_>>()
                    .join(" ");
                if surface.len() >= 3 {
                    out.push(EntityCandidate {
                        surface,
                        class: EntityClass::ProperNoun,
                    });
                }
            }
        }
        i += 1;
    }

    // Pattern 2: Technical terms — CamelCase, snake_case, kebab-case.
    for word in &words {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');
        if clean.len() >= 4 && has_mixed_case(clean) && !is_all_upper(clean) {
            out.push(EntityCandidate {
                surface: clean.to_string(),
                class: EntityClass::Technical,
            });
        }
        if (clean.contains('-') || clean.contains('_')) && clean.len() >= 5 {
            out.push(EntityCandidate {
                surface: clean.to_string(),
                class: EntityClass::Technical,
            });
        }
    }

    // Pattern 3: Quantities with units.
    const QUANTITY_PATTERNS: &[&str] = &[
        "µJ", "µWh", "kWh", "Wh", "GB", "MB", "KB", "ms", "µs", "GHz", "MHz",
    ];
    for window in words.windows(2) {
        for pattern in QUANTITY_PATTERNS {
            if window[1].contains(pattern) || window[0].ends_with(pattern) {
                let quantity = format!("{} {}", window[0], window[1]);
                out.push(EntityCandidate {
                    surface: quantity,
                    class: EntityClass::Quantity,
                });
                break;
            }
        }
    }

    out
}

/// True iff the first character of `word` is uppercase.
pub fn is_capitalized(word: &str) -> bool {
    word.chars().next().is_some_and(|c| c.is_uppercase())
}

/// True iff every alphabetic character in `word` is uppercase.
pub fn is_all_upper(word: &str) -> bool {
    word.chars().all(|c| !c.is_alphabetic() || c.is_uppercase())
}

/// True iff `word` mixes uppercase and lowercase letters (CamelCase-style).
pub fn has_mixed_case(word: &str) -> bool {
    let has_upper = word.chars().any(|c| c.is_uppercase());
    let has_lower = word.chars().any(|c| c.is_lowercase());
    has_upper && has_lower
}

/// True iff `word` is a closed-class English stop word — filtered out of
/// the proper-noun pattern so phrases like "The Rust Programming Language"
/// don't include the leading article.
pub fn is_stopword(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "the"
            | "a"
            | "an"
            | "in"
            | "on"
            | "at"
            | "to"
            | "for"
            | "of"
            | "and"
            | "or"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "have"
            | "has"
            | "had"
            | "do"
            | "does"
            | "did"
            | "will"
            | "would"
            | "could"
            | "should"
            | "may"
            | "might"
            | "this"
            | "that"
            | "these"
            | "those"
            | "it"
            | "its"
            | "with"
            | "from"
            | "by"
            | "as"
            | "if"
            | "not"
            | "but"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_proper_nouns_basic() {
        let cands = extract_entity_candidates("The Rust Programming Language is fast.");
        let proper: Vec<_> = cands
            .iter()
            .filter(|c| c.class == EntityClass::ProperNoun)
            .map(|c| c.surface.as_str())
            .collect();
        assert!(
            proper.contains(&"Rust Programming Language"),
            "expected multi-word proper noun, got {proper:?}",
        );
    }

    #[test]
    fn extract_technical_terms() {
        let cands = extract_entity_candidates("Use BTreeMap and ssm-supreme for inference.");
        let tech: Vec<_> = cands
            .iter()
            .filter(|c| c.class == EntityClass::Technical)
            .map(|c| c.surface.as_str())
            .collect();
        assert!(tech.contains(&"BTreeMap"), "got {tech:?}");
        assert!(tech.contains(&"ssm-supreme"), "got {tech:?}");
    }

    #[test]
    fn extract_quantities() {
        let cands = extract_entity_candidates("This uses 344 µJ per query.");
        let quants: Vec<_> = cands
            .iter()
            .filter(|c| c.class == EntityClass::Quantity)
            .map(|c| c.surface.as_str())
            .collect();
        assert!(!quants.is_empty(), "expected at least one quantity hit");
    }

    #[test]
    fn empty_text_yields_nothing() {
        assert!(extract_entity_candidates("").is_empty());
    }

    #[test]
    fn leading_stopword_excluded_from_proper_noun() {
        let cands = extract_entity_candidates("The Rust Language is great.");
        // The leading "The" must not anchor the proper-noun span.
        let proper: Vec<_> = cands
            .iter()
            .filter(|c| c.class == EntityClass::ProperNoun)
            .map(|c| c.surface.as_str())
            .collect();
        assert!(
            proper.iter().any(|s| !s.starts_with("The ")),
            "no proper noun should start with 'The ', got {proper:?}",
        );
    }

    #[test]
    fn all_caps_acronym_is_not_technical() {
        // ABC is all upper-case → not a CamelCase technical term.
        let cands = extract_entity_candidates("ABC ABC ABC test");
        let tech: Vec<_> = cands
            .iter()
            .filter(|c| c.class == EntityClass::Technical)
            .collect();
        assert!(tech.is_empty(), "all-caps should not be technical: {tech:?}");
    }

    #[test]
    fn helpers_behave() {
        assert!(is_capitalized("Foo"));
        assert!(!is_capitalized("foo"));
        assert!(is_all_upper("ABC"));
        assert!(!is_all_upper("AbC"));
        assert!(has_mixed_case("BTreeMap"));
        assert!(!has_mixed_case("rust"));
        assert!(is_stopword("the"));
        assert!(is_stopword("THE"));
        assert!(!is_stopword("rust"));
    }

    #[test]
    fn entity_class_tags() {
        assert_eq!(EntityClass::ProperNoun.tag(), "entity");
        assert_eq!(EntityClass::Technical.tag(), "tech");
        assert_eq!(EntityClass::Quantity.tag(), "quantity");
    }
}
