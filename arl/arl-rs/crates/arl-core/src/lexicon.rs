//! Controlled-vocabulary enforcement.
//!
//! The ARL Lexicon governs the words that may appear in an ARL claim.
//! Terms that cannot be operationalized are excluded — not out of
//! philosophical objection, but because a measurement framework cannot
//! carry a word that cannot be measured. This module flags such terms in
//! a claim's prose.
//!
//! Two severities:
//! - **Excluded** — no operational definition; cannot appear in a claim
//!   ([`Severity::Excluded`]). Presence is a hard validation error.
//! - **PartiallyHype** — has a narrow operational meaning but is commonly
//!   used in an unmeasurable sense ([`Severity::PartiallyHype`]). Allowed,
//!   but flagged for operational-sense review.

use serde::{Deserialize, Serialize};

/// How the lexicon regards a flagged term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// No operational definition — excluded from ARL claims entirely.
    Excluded,
    /// Operational meaning exists but is routinely overloaded — review.
    PartiallyHype,
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

/// Excluded terms — no operational definition (LEXICON: "Not in this
/// lexicon" / "Hype term … excluded from ARL claims"). Phrases (with a
/// space or hyphen) are matched as substrings; bare words are matched as
/// whole tokens.
const EXCLUDED: &[&str] = &[
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

/// Partially-hype terms — operational only in a narrow sense (LEXICON
/// "Partially hype term"). Allowed but flagged.
const PARTIALLY_HYPE: &[&str] = &[
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
    for term in EXCLUDED {
        if occurs(&lower, term) {
            out.push(LexiconFinding {
                term: (*term).to_string(),
                field: field.to_string(),
                severity: Severity::Excluded,
            });
        }
    }
    for term in PARTIALLY_HYPE {
        if occurs(&lower, term) {
            out.push(LexiconFinding {
                term: (*term).to_string(),
                field: field.to_string(),
                severity: Severity::PartiallyHype,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_excluded_terms() {
        let f = scan_field("task", "Demonstrates AGI-level performance");
        assert!(f.iter().any(|x| x.term == "agi" && x.severity == Severity::Excluded));
    }

    #[test]
    fn flags_partial_hype() {
        let f = scan_field("context", "improves model alignment and safety");
        assert!(f.iter().any(|x| x.term == "alignment" && x.severity == Severity::PartiallyHype));
        assert!(f.iter().any(|x| x.term == "safety"));
    }

    #[test]
    fn whole_token_match_avoids_false_positives() {
        // "reasoning" is partial-hype, but must not fire inside "seasoning".
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
