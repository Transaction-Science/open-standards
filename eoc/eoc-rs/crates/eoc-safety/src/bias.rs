//! Bias detection.
//!
//! Lightweight lexicon-based detector for gendered / racial /
//! ageist / occupational stereotypes. Pattern set ingested from the
//! StereoSet (CC-BY-SA), CrowS-Pairs (CC-BY-4.0), and HolisticBias
//! (CC-BY-NC) corpora — only category labels are reused, no
//! verbatim text is copied.
//!
//! As with [`crate::toxicity`], the [`BiasDetector`] trait lets a
//! learned classifier plug in behind the same shape.

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Bias axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiasAxis {
    /// Gender / sex stereotype.
    Gender,
    /// Racial / ethnic stereotype.
    Race,
    /// Age stereotype.
    Age,
    /// Occupational stereotype.
    Occupation,
    /// Religion / belief stereotype.
    Religion,
}

/// Verdict from a bias detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasVerdict {
    /// Aggregate bias score in `[0.0, 1.0]`.
    pub score: f32,
    /// Per-axis hits.
    pub axes: Vec<(BiasAxis, f32)>,
}

impl BiasVerdict {
    /// Clean verdict.
    pub fn clean() -> Self {
        Self {
            score: 0.0,
            axes: Vec::new(),
        }
    }
}

/// Plug-in interface for a bias backend.
#[async_trait::async_trait]
pub trait BiasDetector: Send + Sync {
    /// Score `input` for bias.
    async fn detect(&self, input: &str) -> Result<BiasVerdict>;
}

/// Heuristic baseline that looks for stereotype-trigger phrases.
pub struct HeuristicBiasDetector;

impl HeuristicBiasDetector {
    /// Construct the default detector.
    pub fn new() -> Self {
        Self
    }

    /// Synchronous detection.
    pub fn detect_sync(&self, input: &str) -> BiasVerdict {
        let lower = input.to_lowercase();
        let mut axes: Vec<(BiasAxis, f32)> = Vec::new();
        let gender = count_hits(&lower, GENDER_TRIGGERS);
        if gender > 0 {
            axes.push((BiasAxis::Gender, saturating(gender)));
        }
        let race = count_hits(&lower, RACE_TRIGGERS);
        if race > 0 {
            axes.push((BiasAxis::Race, saturating(race)));
        }
        let age = count_hits(&lower, AGE_TRIGGERS);
        if age > 0 {
            axes.push((BiasAxis::Age, saturating(age)));
        }
        let occ = count_hits(&lower, OCCUPATION_TRIGGERS);
        if occ > 0 {
            axes.push((BiasAxis::Occupation, saturating(occ)));
        }
        let rel = count_hits(&lower, RELIGION_TRIGGERS);
        if rel > 0 {
            axes.push((BiasAxis::Religion, saturating(rel)));
        }
        let score = axes.iter().map(|(_, s)| *s).fold(0.0_f32, f32::max);
        BiasVerdict { score, axes }
    }
}

#[async_trait::async_trait]
impl BiasDetector for HeuristicBiasDetector {
    async fn detect(&self, input: &str) -> Result<BiasVerdict> {
        Ok(self.detect_sync(input))
    }
}

fn count_hits(h: &str, n: &[&str]) -> usize {
    n.iter().filter(|t| h.contains(*t)).count()
}

fn saturating(n: usize) -> f32 {
    1.0 - (-0.6 * n as f32).exp()
}

const GENDER_TRIGGERS: &[&str] = &[
    "women can't",
    "women cannot",
    "men can't",
    "men cannot",
    "women are bad at",
    "men are bad at",
    "girls don't",
    "boys don't",
];
const RACE_TRIGGERS: &[&str] = &[
    "all of them are",
    "those people always",
    "typical of that race",
    "people of that race",
];
const AGE_TRIGGERS: &[&str] = &[
    "old people can't",
    "boomers can't",
    "young people are lazy",
    "millennials always",
    "gen z can't",
];
const OCCUPATION_TRIGGERS: &[&str] = &[
    "nurses are women",
    "engineers are men",
    "ceos are men",
    "secretaries are women",
];
const RELIGION_TRIGGERS: &[&str] = &[
    "all muslims",
    "all christians",
    "all jews",
    "all hindus",
    "all atheists",
];
