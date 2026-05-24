//! Toxicity classification.
//!
//! Defines the [`ToxicityClassifier`] trait and a deterministic
//! lexicon-based baseline ([`HeuristicToxicityClassifier`]). The
//! lexicon is a small public-domain set of bucketed terms (slurs,
//! threats, insults) suitable as a first-line filter; real deployments
//! should plug in a learned classifier (Perspective API, Detoxify,
//! Llama Guard) behind the same trait.

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Toxicity sub-category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToxCategory {
    /// Direct slurs / hateful language.
    Hate,
    /// Threats of violence.
    Threat,
    /// General insults / profanity.
    Insult,
    /// Sexually explicit language.
    Sexual,
    /// Self-harm references.
    SelfHarm,
}

/// Verdict from a toxicity classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToxicityVerdict {
    /// Overall toxicity score in `[0.0, 1.0]`.
    pub score: f32,
    /// Per-category contributing scores.
    pub categories: Vec<(ToxCategory, f32)>,
    /// True if the verdict exceeds the configured threshold.
    pub reject: bool,
}

impl ToxicityVerdict {
    /// Clean verdict.
    pub fn clean() -> Self {
        Self {
            score: 0.0,
            categories: Vec::new(),
            reject: false,
        }
    }
}

/// Plug-in interface for any toxicity backend.
#[async_trait::async_trait]
pub trait ToxicityClassifier: Send + Sync {
    /// Score `input` for toxicity.
    async fn classify(&self, input: &str) -> Result<ToxicityVerdict>;
}

/// Lexicon-based baseline classifier — fast, deterministic, no model.
pub struct HeuristicToxicityClassifier {
    threshold: f32,
}

impl HeuristicToxicityClassifier {
    /// Build with the default threshold (`0.5`).
    pub fn new() -> Self {
        Self { threshold: 0.5 }
    }

    /// Build with a custom threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Synchronous lookup (`async` wrapper is also provided).
    pub fn classify_sync(&self, input: &str) -> ToxicityVerdict {
        let lower = input.to_lowercase();
        let mut buckets: Vec<(ToxCategory, f32)> = Vec::new();

        let hate = count_hits(&lower, HATE_TERMS);
        if hate > 0 {
            buckets.push((ToxCategory::Hate, saturating(hate)));
        }
        let threat = count_hits(&lower, THREAT_TERMS);
        if threat > 0 {
            buckets.push((ToxCategory::Threat, saturating(threat)));
        }
        let insult = count_hits(&lower, INSULT_TERMS);
        if insult > 0 {
            buckets.push((ToxCategory::Insult, saturating(insult)));
        }
        let sexual = count_hits(&lower, SEXUAL_TERMS);
        if sexual > 0 {
            buckets.push((ToxCategory::Sexual, saturating(sexual)));
        }
        let selfharm = count_hits(&lower, SELFHARM_TERMS);
        if selfharm > 0 {
            buckets.push((ToxCategory::SelfHarm, saturating(selfharm)));
        }

        let score = buckets
            .iter()
            .map(|(_, s)| *s)
            .fold(0.0_f32, |a, b| a.max(b));
        ToxicityVerdict {
            score,
            reject: score >= self.threshold,
            categories: buckets,
        }
    }
}

#[async_trait::async_trait]
impl ToxicityClassifier for HeuristicToxicityClassifier {
    async fn classify(&self, input: &str) -> Result<ToxicityVerdict> {
        Ok(self.classify_sync(input))
    }
}

fn count_hits(haystack: &str, needles: &[&str]) -> usize {
    needles.iter().filter(|n| haystack.contains(*n)).count()
}

fn saturating(n: usize) -> f32 {
    1.0 - (-0.7 * n as f32).exp()
}

// Intentionally a thin, non-exhaustive starter lexicon. Real deployments
// plug in a learned classifier behind `ToxicityClassifier`.
const HATE_TERMS: &[&str] = &["hate group", "subhuman", "ethnic slur"];
const THREAT_TERMS: &[&str] = &["i will kill", "i'll kill", "i will hurt", "going to murder", "shoot you", "burn down"];
const INSULT_TERMS: &[&str] = &["idiot", "moron", "stupid", "loser", "scumbag"];
const SEXUAL_TERMS: &[&str] = &["explicit sexual", "graphic sexual"];
const SELFHARM_TERMS: &[&str] = &["kill myself", "end my life", "self-harm", "self harm"];
