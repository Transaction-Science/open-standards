//! NSFW (text) detection.
//!
//! Lightweight lexicon-based filter for sexually explicit and
//! graphic-violence text. Pattern bucketing follows the OpenAI
//! moderation categories (sexual, sexual/minors, violence,
//! violence/graphic). Image / video NSFW is **out of scope** for
//! this text-first detector; the trait is the integration point.

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// NSFW sub-category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NsfwCategory {
    /// Adult sexual content.
    Sexual,
    /// Sexual content involving minors — always reject.
    SexualMinors,
    /// Violent content.
    Violence,
    /// Graphic / gore-level violence.
    ViolenceGraphic,
}

/// Verdict from the NSFW classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NsfwVerdict {
    /// Overall NSFW score in `[0.0, 1.0]`.
    pub score: f32,
    /// Per-category hits.
    pub categories: Vec<(NsfwCategory, f32)>,
    /// True if the verdict exceeds threshold (or contains a hard-reject
    /// category like [`NsfwCategory::SexualMinors`]).
    pub reject: bool,
}

impl NsfwVerdict {
    /// Clean verdict.
    pub fn clean() -> Self {
        Self {
            score: 0.0,
            categories: Vec::new(),
            reject: false,
        }
    }
}

/// Plug-in interface for an NSFW backend (text or multimodal).
#[async_trait::async_trait]
pub trait NsfwClassifier: Send + Sync {
    /// Score `input` for NSFW content.
    async fn classify(&self, input: &str) -> Result<NsfwVerdict>;
}

/// Lexicon-based baseline classifier.
pub struct HeuristicNsfwClassifier {
    threshold: f32,
}

impl HeuristicNsfwClassifier {
    /// Build with the default threshold (`0.5`).
    pub fn new() -> Self {
        Self { threshold: 0.5 }
    }

    /// Build with a custom threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Synchronous classification.
    pub fn classify_sync(&self, input: &str) -> NsfwVerdict {
        let lower = input.to_lowercase();
        let mut hits: Vec<(NsfwCategory, f32)> = Vec::new();
        let mut hard_reject = false;

        let minors = count_hits(&lower, SEXUAL_MINORS_TERMS);
        if minors > 0 {
            hits.push((NsfwCategory::SexualMinors, 1.0));
            hard_reject = true;
        }
        let sexual = count_hits(&lower, SEXUAL_TERMS);
        if sexual > 0 {
            hits.push((NsfwCategory::Sexual, saturating(sexual)));
        }
        let violence = count_hits(&lower, VIOLENCE_TERMS);
        if violence > 0 {
            hits.push((NsfwCategory::Violence, saturating(violence)));
        }
        let graphic = count_hits(&lower, GRAPHIC_VIOLENCE_TERMS);
        if graphic > 0 {
            hits.push((NsfwCategory::ViolenceGraphic, saturating(graphic)));
        }

        let score = hits.iter().map(|(_, s)| *s).fold(0.0_f32, f32::max);
        NsfwVerdict {
            score,
            reject: hard_reject || score >= self.threshold,
            categories: hits,
        }
    }
}

#[async_trait::async_trait]
impl NsfwClassifier for HeuristicNsfwClassifier {
    async fn classify(&self, input: &str) -> Result<NsfwVerdict> {
        Ok(self.classify_sync(input))
    }
}

fn count_hits(h: &str, n: &[&str]) -> usize {
    n.iter().filter(|t| h.contains(*t)).count()
}

fn saturating(n: usize) -> f32 {
    1.0 - (-0.7 * n as f32).exp()
}

// Hard-reject category. Trigger phrases here are intentionally non-graphic.
const SEXUAL_MINORS_TERMS: &[&str] = &["sexual content involving minors", "child sexual"];
const SEXUAL_TERMS: &[&str] = &["explicit sexual content", "graphic sexual content"];
const VIOLENCE_TERMS: &[&str] = &["graphic violence", "depictions of violence", "violent threat"];
const GRAPHIC_VIOLENCE_TERMS: &[&str] = &["torture scene", "graphic gore", "dismemberment"];
