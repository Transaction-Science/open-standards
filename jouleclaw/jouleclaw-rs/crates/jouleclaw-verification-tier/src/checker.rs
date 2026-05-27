//! Agreement checkers — given the ≥2 candidate answers produced by
//! the cross-model dispatch, decide whether they agree.
//!
//! The whole point of L4 is to refuse on disagreement. The tier holds a
//! `Box<dyn AgreementChecker>`, so the comparison strategy is
//! swappable: `StringMatchChecker` for short factual answers,
//! `JaccardChecker` for longer free-form text, or a custom semantic
//! checker for downstream crates.

use std::collections::HashSet;

/// The verdict an [`AgreementChecker`] returns.
#[derive(Debug, Clone, PartialEq)]
pub enum AgreementVerdict {
    /// All candidates agreed. Carries the canonical consensus string
    /// (usually the first candidate's text after normalisation) and a
    /// confidence in `[0.0, 1.0]` that the tier folds into the
    /// answer's `confidence` field.
    Agree { consensus: String, confidence: f32 },
    /// Candidates disagreed. `reason` is a human-readable summary used
    /// only for diagnostics and tier-specific refusal reasons.
    Disagree { reason: String },
    /// The checker could not decide (empty input, all candidates blank,
    /// etc.). The tier treats this identically to `Disagree` but emits
    /// a distinct refusal reason so audits can distinguish them.
    Inconclusive,
}

/// A pluggable agreement strategy.
///
/// Implementations decide what "the models agreed" means: byte
/// equality, normalised equality, token-Jaccard ≥ threshold, an
/// embedding similarity, etc. The tier hands the strategy *all*
/// candidate strings at once (not a sequence of pairs) so checkers
/// that look at the distribution as a whole — majority vote,
/// clustering — are first-class.
pub trait AgreementChecker: Send + Sync {
    fn check(&self, candidates: &[String]) -> AgreementVerdict;
}

/// Normalise a string for comparison: lowercase, collapse all
/// whitespace runs to a single space, strip leading/trailing
/// whitespace. Pure-function; shared by both built-in checkers.
pub fn normalise(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

// ============================================================
// StringMatchChecker — exact normalised equality
// ============================================================

/// All candidates must normalise to the same string. The strictest
/// built-in checker; appropriate for fact-shaped answers ("3", "Paris",
/// "rust") where any deviation is meaningful.
#[derive(Debug, Clone, Default)]
pub struct StringMatchChecker;

impl StringMatchChecker {
    pub fn new() -> Self {
        Self
    }
}

impl AgreementChecker for StringMatchChecker {
    fn check(&self, candidates: &[String]) -> AgreementVerdict {
        if candidates.len() < 2 {
            return AgreementVerdict::Inconclusive;
        }
        let normed: Vec<String> = candidates.iter().map(|s| normalise(s)).collect();
        // Reject if any candidate is empty after normalisation.
        if normed.iter().any(|s| s.is_empty()) {
            return AgreementVerdict::Inconclusive;
        }
        let first = &normed[0];
        if normed.iter().all(|s| s == first) {
            AgreementVerdict::Agree {
                // Use the *first original* (not the normalised form) so
                // callers see well-formatted text. The vote on
                // normalised strings is what gates agreement.
                consensus: candidates[0].clone(),
                // Exact string match across N≥2 distinct models is
                // strong evidence — we report 0.99 (not 1.0) to leave
                // a sliver of doubt for adversarial cases where the
                // models share a training shortcut.
                confidence: 0.99,
            }
        } else {
            AgreementVerdict::Disagree {
                reason: format!(
                    "string mismatch across {} candidates",
                    candidates.len()
                ),
            }
        }
    }
}

// ============================================================
// JaccardChecker — token-Jaccard similarity threshold
// ============================================================

/// Pairwise token-Jaccard must meet `threshold` across every pair of
/// candidates. The default threshold is `0.8` (per the donor); raise
/// it for stricter agreement, lower it for fuzzier prose.
///
/// Tokenisation is whitespace-split-then-lowercase — deliberately
/// dumb. The whole point is to be deterministic and cheap; smarter
/// tokenisation belongs in a custom checker.
#[derive(Debug, Clone)]
pub struct JaccardChecker {
    threshold: f32,
}

impl JaccardChecker {
    /// Construct with the canonical 0.8 threshold.
    pub fn new() -> Self {
        Self { threshold: 0.8 }
    }

    /// Construct with a custom threshold in `[0.0, 1.0]`. Values
    /// outside the range are clamped.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

impl Default for JaccardChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Token-set Jaccard similarity of two strings. Public so consumers
/// can sanity-check thresholds in their own tests.
pub fn jaccard(a: &str, b: &str) -> f32 {
    let ta: HashSet<&str> = a.split_whitespace().collect();
    let tb: HashSet<&str> = b.split_whitespace().collect();
    if ta.is_empty() && tb.is_empty() {
        return 1.0;
    }
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let inter = ta.intersection(&tb).count();
    let union = ta.union(&tb).count();
    if union == 0 {
        return 0.0;
    }
    inter as f32 / union as f32
}

impl AgreementChecker for JaccardChecker {
    fn check(&self, candidates: &[String]) -> AgreementVerdict {
        if candidates.len() < 2 {
            return AgreementVerdict::Inconclusive;
        }
        let normed: Vec<String> = candidates.iter().map(|s| normalise(s)).collect();
        if normed.iter().any(|s| s.is_empty()) {
            return AgreementVerdict::Inconclusive;
        }

        // Pairwise check: every pair must clear the threshold. The
        // minimum pairwise score becomes the confidence floor.
        let mut min_score: f32 = 1.0;
        for i in 0..normed.len() {
            for j in (i + 1)..normed.len() {
                let s = jaccard(&normed[i], &normed[j]);
                if s < min_score {
                    min_score = s;
                }
            }
        }

        if min_score >= self.threshold {
            // Confidence scales with the agreement margin: at the
            // threshold we report 0.9, climbing to 0.99 as the score
            // saturates at 1.0.
            let margin = (min_score - self.threshold) / (1.0 - self.threshold).max(1e-6);
            let confidence = 0.9 + 0.09 * margin.clamp(0.0, 1.0);
            AgreementVerdict::Agree {
                consensus: candidates[0].clone(),
                confidence,
            }
        } else {
            AgreementVerdict::Disagree {
                reason: format!(
                    "jaccard {:.3} below threshold {:.3}",
                    min_score, self.threshold
                ),
            }
        }
    }
}

// ============================================================
// Tests for the checkers
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalise_collapses_whitespace_and_lowercases() {
        assert_eq!(normalise("  Hello   World  "), "hello world");
        assert_eq!(normalise("\tA\nB\rC"), "a b c");
    }

    #[test]
    fn jaccard_identical_is_one() {
        assert!((jaccard("a b c", "a b c") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_disjoint_is_zero() {
        assert_eq!(jaccard("a b c", "x y z"), 0.0);
    }

    #[test]
    fn jaccard_partial_overlap() {
        // {a,b,c} vs {b,c,d} → intersection 2, union 4 → 0.5.
        assert!((jaccard("a b c", "b c d") - 0.5).abs() < 1e-6);
    }

    #[test]
    fn string_match_agree_on_equal_inputs() {
        let chk = StringMatchChecker::new();
        let v = chk.check(&["Paris".into(), "paris".into(), "  PARIS  ".into()]);
        match v {
            AgreementVerdict::Agree { consensus, confidence } => {
                assert_eq!(consensus, "Paris");
                assert!(confidence > 0.9);
            }
            _ => panic!("expected agree, got {:?}", v),
        }
    }

    #[test]
    fn string_match_disagree() {
        let chk = StringMatchChecker::new();
        let v = chk.check(&["Paris".into(), "London".into()]);
        assert!(matches!(v, AgreementVerdict::Disagree { .. }));
    }

    #[test]
    fn string_match_inconclusive_on_singleton() {
        let chk = StringMatchChecker::new();
        let v = chk.check(&["Paris".into()]);
        assert!(matches!(v, AgreementVerdict::Inconclusive));
    }

    #[test]
    fn string_match_inconclusive_on_empty_candidate() {
        let chk = StringMatchChecker::new();
        let v = chk.check(&["".into(), "".into()]);
        assert!(matches!(v, AgreementVerdict::Inconclusive));
    }

    #[test]
    fn jaccard_agree_above_threshold() {
        let chk = JaccardChecker::with_threshold(0.5);
        let v = chk.check(&[
            "the quick brown fox".into(),
            "the quick brown dog".into(),
        ]);
        match v {
            AgreementVerdict::Agree { confidence, .. } => {
                assert!(confidence >= 0.9);
            }
            _ => panic!("expected agree, got {:?}", v),
        }
    }

    #[test]
    fn jaccard_disagree_below_threshold() {
        let chk = JaccardChecker::with_threshold(0.8);
        let v = chk.check(&[
            "alpha beta gamma".into(),
            "delta epsilon zeta".into(),
        ]);
        assert!(matches!(v, AgreementVerdict::Disagree { .. }));
    }

    #[test]
    fn jaccard_threshold_edge_exact() {
        // Exact-equal candidates → score 1.0, threshold 1.0 → agree.
        let chk = JaccardChecker::with_threshold(1.0);
        let v = chk.check(&["a b c".into(), "a b c".into()]);
        assert!(matches!(v, AgreementVerdict::Agree { .. }));
    }

    #[test]
    fn jaccard_threshold_clamps_to_unit_range() {
        let chk = JaccardChecker::with_threshold(5.0);
        assert!((chk.threshold() - 1.0).abs() < 1e-6);
        let chk2 = JaccardChecker::with_threshold(-1.0);
        assert!((chk2.threshold() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_inconclusive_on_singleton() {
        let chk = JaccardChecker::new();
        let v = chk.check(&["Paris".into()]);
        assert!(matches!(v, AgreementVerdict::Inconclusive));
    }
}
