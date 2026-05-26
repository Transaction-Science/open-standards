//! Atomization (spec §6.3).
//!
//! Decomposes a draft answer into [`AtomicClaim`]s with stakes
//! determination. The spec calls for an LLM-based atomizer; this
//! module ships a sentence-splitting [`SentenceAtomizer`] as the
//! default impl so the diagnose pillar is functional without
//! routing through a reasoner. Replace it with an LLM-backed
//! [`Atomizer`] impl when the deployment can afford the energy.

use uuid::Uuid;

use jouleclaw_schema::{AtomicClaim, ClaimStakes, KnowledgeAxes};

#[derive(Debug)]
pub enum AtomizeError {
    EmptyDraft,
    Backend(String),
}

impl std::fmt::Display for AtomizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDraft => write!(f, "empty draft"),
            Self::Backend(s) => write!(f, "backend error: {s}"),
        }
    }
}

impl std::error::Error for AtomizeError {}

/// `Atomizer` produces a sequence of [`AtomicClaim`]s from a draft.
/// The draft is supplied as `(segment_id, text)` pairs so each claim
/// carries the segment it was extracted from — needed by the
/// verified composer (spec §7.2) to apply recovery actions per
/// segment.
pub trait Atomizer: Send + Sync {
    fn atomize(
        &self,
        segments: &[(String, String)],
        default_axes: &KnowledgeAxes,
    ) -> Result<Vec<AtomicClaim>, AtomizeError>;
}

/// Naive default atomizer: splits each segment into sentence-bounded
/// claims and assigns a heuristic [`ClaimStakes`] from text features
/// (decision-language → High, hedging → Low, etc.). Sufficient for
/// the minimum-viable diagnose pillar; an LLM-backed impl will land
/// when accuracy matters more than dependency footprint.
pub struct SentenceAtomizer;

impl SentenceAtomizer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SentenceAtomizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Atomizer for SentenceAtomizer {
    fn atomize(
        &self,
        segments: &[(String, String)],
        default_axes: &KnowledgeAxes,
    ) -> Result<Vec<AtomicClaim>, AtomizeError> {
        let total_chars: usize = segments.iter().map(|(_, t)| t.len()).sum();
        if total_chars == 0 {
            return Err(AtomizeError::EmptyDraft);
        }
        let mut out = Vec::new();
        for (seg_id, text) in segments {
            for sentence in split_sentences(text) {
                if sentence.trim().is_empty() {
                    continue;
                }
                out.push(AtomicClaim {
                    schema_version: "2.0".into(),
                    claim_id: Uuid::new_v4(),
                    text: sentence.trim().to_string(),
                    segment_id: seg_id.clone(),
                    stakes: heuristic_stakes(&sentence),
                    knowledge_axes: default_axes.clone(),
                    atomization_notes: None,
                    metadata: Default::default(),
                });
            }
        }
        Ok(out)
    }
}

/// Free function form for callers that don't want to manage an
/// `Atomizer` instance — just splits the joined draft on sentence
/// boundaries.
pub fn atomize_sentences(
    segments: &[(String, String)],
    default_axes: &KnowledgeAxes,
) -> Result<Vec<AtomicClaim>, AtomizeError> {
    SentenceAtomizer::new().atomize(segments, default_axes)
}

/// Split a paragraph on sentence-final punctuation. Naive but
/// deterministic — splits on `[.!?]` followed by whitespace and a
/// capital-letter start, OR end-of-input. The capital-letter rule
/// keeps `"U.S. capital"` intact (next non-space is `c`), since the
/// real sentence boundary is the next capitalized word.
fn split_sentences(text: &str) -> Vec<String> {
    let bytes: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut current = String::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        current.push(c);
        if matches!(c, '.' | '!' | '?') {
            // Find the next non-whitespace character (if any).
            let mut j = i + 1;
            let mut crossed_whitespace = false;
            while j < bytes.len() && bytes[j].is_whitespace() {
                crossed_whitespace = true;
                j += 1;
            }
            let split_here = if j == bytes.len() {
                // End of input → split.
                crossed_whitespace || c != '.' || !looks_like_abbrev(&current)
            } else if crossed_whitespace {
                // Whitespace between punctuation and next token.
                // Split only if the next token starts with an
                // uppercase letter (a fresh sentence). For "U.S.
                // capital" the next token starts lowercase, so we
                // keep them in the same sentence.
                bytes[j].is_uppercase() || bytes[j].is_ascii_digit()
            } else {
                false
            };
            if split_here {
                out.push(std::mem::take(&mut current));
            }
        }
        i += 1;
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

/// Cheap check: a "U.S." style abbreviation looks like alternating
/// uppercase letters and periods. Used only to decide whether to
/// emit an end-of-input fragment as its own sentence.
fn looks_like_abbrev(token: &str) -> bool {
    let s = token.trim_start();
    if s.len() < 2 {
        return false;
    }
    // Take the last 6 chars and look for a Period-Letter-Period
    // pattern.
    let tail: Vec<char> = s.chars().rev().take(6).collect();
    // Reversed, so e.g. "S.U." → ['.', 'S', '.', 'U', ' ', '.']…
    // we look for the `.X.` triplet at the start of the reversed
    // tail.
    matches!(tail.first(), Some('.'))
        && matches!(tail.get(1), Some(c) if c.is_alphabetic())
        && matches!(tail.get(2), Some('.'))
}

/// Quick heuristic for assigning [`ClaimStakes`] from sentence text.
/// Critical-marked sentences trigger more aggressive verification
/// (higher entailment confidence floor); Low sentences are allowed
/// to pass with weaker support.
fn heuristic_stakes(sentence: &str) -> ClaimStakes {
    let lc = sentence.to_lowercase();
    // Safety/compliance critical markers.
    for needle in ["dosage", "diagnose", "diagnosis", "prescribe", "lethal", "fatal"] {
        if lc.contains(needle) {
            return ClaimStakes::Critical;
        }
    }
    // Decision-guiding language.
    for needle in [
        "you should",
        "i recommend",
        "do not",
        "must not",
        "best option",
        "buy",
        "deploy",
        "use this",
    ] {
        if lc.contains(needle) {
            return ClaimStakes::High;
        }
    }
    // Hedging markers reduce stakes.
    for needle in ["maybe", "perhaps", "could be", "might", "possibly"] {
        if lc.contains(needle) {
            return ClaimStakes::Low;
        }
    }
    ClaimStakes::Medium
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use jouleclaw_schema::{GranularityClass, ScopeClass, TemporalStabilityClass};

    fn axes() -> KnowledgeAxes {
        KnowledgeAxes {
            schema_version: "5.0".into(),
            valid_time_start: None,
            valid_time_end: None,
            transaction_time: None,
            reference_time: Utc::now(),
            temporal_stability: TemporalStabilityClass::Slow,
            granularity: GranularityClass::Coarse,
            granularity_notes: None,
            scope: ScopeClass::Particular,
            scope_domain: None,
            certainty: 1.0,
            certainty_basis: "test".into(),
            source_uri: None,
            source_authority_tier: 1,
            extraction_method: None,
            citation_chain: vec![],
            metadata: Default::default(),
        }
    }

    #[test]
    fn splits_multi_sentence_paragraph() {
        let segs = vec![(
            "s0".into(),
            "Paris is the capital. France is in Europe. The Eiffel Tower is in Paris.".into(),
        )];
        let claims = atomize_sentences(&segs, &axes()).unwrap();
        assert_eq!(claims.len(), 3);
        assert!(claims[0].text.starts_with("Paris"));
        assert!(claims[1].text.starts_with("France"));
        assert!(claims[2].text.contains("Eiffel"));
    }

    #[test]
    fn preserves_segment_id_per_claim() {
        let segs = vec![
            ("intro".into(), "Background statement.".into()),
            ("body".into(), "Main claim. Supporting detail.".into()),
        ];
        let claims = atomize_sentences(&segs, &axes()).unwrap();
        let intro_count = claims.iter().filter(|c| c.segment_id == "intro").count();
        let body_count = claims.iter().filter(|c| c.segment_id == "body").count();
        assert_eq!(intro_count, 1);
        assert_eq!(body_count, 2);
    }

    #[test]
    fn empty_draft_errors() {
        let segs: Vec<(String, String)> = vec![];
        assert!(matches!(
            atomize_sentences(&segs, &axes()),
            Err(AtomizeError::EmptyDraft)
        ));
        let segs = vec![("s0".into(), "".into())];
        assert!(matches!(
            atomize_sentences(&segs, &axes()),
            Err(AtomizeError::EmptyDraft)
        ));
    }

    #[test]
    fn does_not_split_at_internal_periods() {
        // "U.S." has internal periods that aren't sentence
        // terminators — followed by a non-space character.
        let segs = vec![(
            "s0".into(),
            "The U.S. capital is Washington. London is in the U.K.".into(),
        )];
        let claims = atomize_sentences(&segs, &axes()).unwrap();
        assert_eq!(claims.len(), 2, "got {claims:?}");
    }

    #[test]
    fn stakes_heuristic_flags_decision_language() {
        let segs = vec![("s0".into(), "You should use this library for parsing.".into())];
        let claims = atomize_sentences(&segs, &axes()).unwrap();
        assert!(matches!(claims[0].stakes, ClaimStakes::High));
    }

    #[test]
    fn stakes_heuristic_flags_critical_markers() {
        let segs = vec![("s0".into(), "The recommended dosage is 5mg.".into())];
        let claims = atomize_sentences(&segs, &axes()).unwrap();
        assert!(matches!(claims[0].stakes, ClaimStakes::Critical));
    }

    #[test]
    fn stakes_heuristic_lowers_hedged_claims() {
        let segs = vec![("s0".into(), "Paris might be the capital of France.".into())];
        let claims = atomize_sentences(&segs, &axes()).unwrap();
        assert!(matches!(claims[0].stakes, ClaimStakes::Low));
    }
}
