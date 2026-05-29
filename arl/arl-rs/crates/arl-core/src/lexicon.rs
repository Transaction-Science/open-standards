//! Controlled-vocabulary enforcement.
//!
//! The ARL Lexicon governs the words that may appear in an ARL claim.
//! A term can anchor a claim only when it can be measured. Terms with no
//! single operational definition are set aside — not out of any stance on
//! the AI field, but because a measurement framework cannot carry a word
//! it cannot measure. This module flags such terms in a claim's prose.
//! ARL takes no position on whether these terms are meaningful, real, or
//! imminent; it reports only on whether they are measurable.
//!
//! Two severities:
//! - **Unmeasurable** — no single operational definition; cannot anchor a
//!   claim ([`Severity::Unmeasurable`]). Presence is a hard validation error.
//! - **OperationalSense** — has a measurable operational meaning but is
//!   also used in broader, unmeasurable senses ([`Severity::OperationalSense`]).
//!   Allowed, with a note to confirm the measurable sense is intended.

use serde::{Deserialize, Serialize};

/// How the lexicon regards a flagged term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// No single operational definition — cannot anchor an ARL claim.
    Unmeasurable,
    /// Has a measurable operational sense; also used in broader senses — note.
    OperationalSense,
}

/// A term found in a claim's prose, with where it was found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexiconFinding {
    /// The matched term (canonical lexicon spelling).
    pub term: String,
    /// The claim field the term appeared in (e.g. `"task"`).
    pub field: String,
    pub severity: Severity,
}

/// Terms with no single operational definition (LEXICON: "No operational
/// definition" — cannot anchor an ARL claim). Phrases (with a space or
/// hyphen) are matched as substrings; bare words are matched as whole
/// tokens.
const UNMEASURABLE: &[&str] = &[
    "agi",
    "artificial general intelligence",
    "superintelligence",
    "consciousness",
    "conscious",
    "sentient",
    "sentience",
    "sapience",
    "singularity",
    "human-level",
    "human level",
    "self-aware",
    "self aware",
];

/// Terms with a measurable operational sense that are also used in broader
/// senses (LEXICON "Measurable in its operational sense"). Allowed, noted.
const OPERATIONAL_SENSE: &[&str] = &[
    "alignment",
    "capability emergence",
    "frontier model",
    "hallucination",
    "reasoning",
    "reasoning model",
    "safety",
    "world model",
];

/// Whether `term` (lowercase) occurs in `text` (lowercase). Phrase terms
/// (containing a space or hyphen) match as substrings; single-word terms
/// match only as whole alphanumeric tokens, so `"reasoning"` does not
/// fire inside `"unreasoning"`.
fn occurs(text_lower: &str, term: &str) -> bool {
    if term.contains(' ') || term.contains('-') {
        text_lower.contains(term)
    } else {
        text_lower
            .split(|c: char| !c.is_alphanumeric())
            .any(|tok| tok == term)
    }
}

/// Scan one named field of claim prose for lexicon terms.
pub fn scan_field(field: &str, text: &str) -> Vec<LexiconFinding> {
    let lower = text.to_lowercase();
    let mut out = Vec::new();
    for term in UNMEASURABLE {
        if occurs(&lower, term) {
            out.push(LexiconFinding {
                term: (*term).to_string(),
                field: field.to_string(),
                severity: Severity::Unmeasurable,
            });
        }
    }
    for term in OPERATIONAL_SENSE {
        if occurs(&lower, term) {
            out.push(LexiconFinding {
                term: (*term).to_string(),
                field: field.to_string(),
                severity: Severity::OperationalSense,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_unmeasurable_terms() {
        let f = scan_field("task", "Demonstrates AGI-level performance");
        assert!(f.iter().any(|x| x.term == "agi" && x.severity == Severity::Unmeasurable));
    }

    #[test]
    fn flags_operational_sense() {
        let f = scan_field("context", "improves model alignment and safety");
        assert!(f.iter().any(|x| x.term == "alignment" && x.severity == Severity::OperationalSense));
        assert!(f.iter().any(|x| x.term == "safety"));
    }

    #[test]
    fn whole_token_match_avoids_false_positives() {
        // "reasoning" is operational-sense, but must not fire inside "seasoning".
        let f = scan_field("task", "season the seasoning");
        assert!(f.iter().all(|x| x.term != "reasoning"));
        // and does fire on the real word
        let g = scan_field("task", "chain-of-thought reasoning steps");
        assert!(g.iter().any(|x| x.term == "reasoning"));
    }

    #[test]
    fn phrase_terms_match_as_substring() {
        let f = scan_field("task", "claims human-level translation");
        assert!(f.iter().any(|x| x.term == "human-level"));
    }

    #[test]
    fn clean_prose_yields_nothing() {
        let f = scan_field("task", "translate EN→FR sentences from the WMT24 test set");
        assert!(f.is_empty());
    }
}
