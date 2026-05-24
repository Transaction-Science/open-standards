//! IFEval — verifiable instruction-following evaluation (Zhou et al. 2023).
//!
//! Each case has one or more programmatically verifiable constraints
//! (e.g. "use the word X at least N times", "respond in exactly 2
//! paragraphs", "no commas"). The score for a case is the fraction of
//! constraints satisfied; for a multi-case run the harness reports the
//! mean.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::builtin_samples;
use crate::error::Result;
use crate::harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, IfEvalConstraint, Metric, Response, load_raw,
};

/// IFEval harness.
pub struct IfEval;

impl IfEval {
    /// Create a new IFEval harness.
    pub fn new() -> Self {
        Self
    }
}

impl Default for IfEval {
    fn default() -> Self {
        Self::new()
    }
}

/// Deserialised raw row from the builtin / hub JSON. Constraints are
/// flattened with `#[serde(tag = "type")]`.
#[derive(Debug, Deserialize, Serialize)]
struct RawRow {
    id: String,
    prompt: String,
    constraints: Vec<RawConstraint>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RawConstraint {
    KeywordAtLeast { keyword: String, min_count: usize },
    ExactParagraphs { count: usize },
    WordCount { min: usize, max: usize },
    NoComma,
    AllCaps,
    AllLowercase,
    StartsWith { prefix: String },
    EndsWith { suffix: String },
    ExactBullets { count: usize },
    ForbidKeywords { keywords: Vec<String> },
}

impl From<RawConstraint> for IfEvalConstraint {
    fn from(c: RawConstraint) -> Self {
        match c {
            RawConstraint::KeywordAtLeast { keyword, min_count } => {
                IfEvalConstraint::KeywordAtLeast { keyword, min_count }
            }
            RawConstraint::ExactParagraphs { count } => IfEvalConstraint::ExactParagraphs(count),
            RawConstraint::WordCount { min, max } => IfEvalConstraint::WordCount { min, max },
            RawConstraint::NoComma => IfEvalConstraint::NoComma,
            RawConstraint::AllCaps => IfEvalConstraint::AllCaps,
            RawConstraint::AllLowercase => IfEvalConstraint::AllLowercase,
            RawConstraint::StartsWith { prefix } => IfEvalConstraint::StartsWith(prefix),
            RawConstraint::EndsWith { suffix } => IfEvalConstraint::EndsWith(suffix),
            RawConstraint::ExactBullets { count } => IfEvalConstraint::ExactBullets(count),
            RawConstraint::ForbidKeywords { keywords } => {
                IfEvalConstraint::ForbidKeywords(keywords)
            }
        }
    }
}

/// Return `true` if `output` satisfies `constraint`.
pub fn verify_constraint(output: &str, constraint: &IfEvalConstraint) -> bool {
    match constraint {
        IfEvalConstraint::KeywordAtLeast { keyword, min_count } => {
            let needle = keyword.to_lowercase();
            let hay = output.to_lowercase();
            let mut count = 0usize;
            let mut idx = 0usize;
            while let Some(pos) = hay[idx..].find(&needle) {
                count += 1;
                idx += pos + needle.len();
            }
            count >= *min_count
        }
        IfEvalConstraint::ExactParagraphs(count) => {
            let trimmed = output.trim();
            let paragraphs: Vec<&str> = trimmed
                .split("\n\n")
                .map(|p| p.trim())
                .filter(|p| !p.is_empty())
                .collect();
            paragraphs.len() == *count
        }
        IfEvalConstraint::WordCount { min, max } => {
            let n = output.split_whitespace().count();
            n >= *min && n <= *max
        }
        IfEvalConstraint::NoComma => !output.contains(','),
        IfEvalConstraint::AllCaps => output
            .chars()
            .filter(|c| c.is_alphabetic())
            .all(|c| c.is_uppercase()),
        IfEvalConstraint::AllLowercase => output
            .chars()
            .filter(|c| c.is_alphabetic())
            .all(|c| c.is_lowercase()),
        IfEvalConstraint::StartsWith(p) => output.trim_start().starts_with(p),
        IfEvalConstraint::EndsWith(s) => output.trim_end().ends_with(s),
        IfEvalConstraint::ExactBullets(count) => {
            let n = output
                .lines()
                .map(|l| l.trim_start())
                .filter(|l| l.starts_with("- ") || l.starts_with("* "))
                .count();
            n == *count
        }
        IfEvalConstraint::ForbidKeywords(words) => {
            let hay = output.to_lowercase();
            !words.iter().any(|w| hay.contains(&w.to_lowercase()))
        }
    }
}

#[async_trait]
impl Harness for IfEval {
    fn name(&self) -> &'static str {
        "ifeval"
    }

    fn metric(&self) -> Metric {
        Metric::Custom("instruction_following")
    }

    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>> {
        let raw = load_raw(source, builtin_samples::IFEVAL).await?;
        let rows: Vec<RawRow> = serde_json::from_str(&raw)?;
        Ok(rows
            .into_iter()
            .map(|r| {
                let constraints = r.constraints.into_iter().map(Into::into).collect();
                EvalCase {
                    id: r.id,
                    prompt: r.prompt,
                    expected: ExpectedAnswer::Constraints(constraints),
                    dataset: "ifeval".to_string(),
                    subject: None,
                }
            })
            .collect())
    }

    async fn score(&self, response: &Response, expected: &ExpectedAnswer) -> f64 {
        let ExpectedAnswer::Constraints(cs) = expected else {
            return 0.0;
        };
        if cs.is_empty() {
            return 0.0;
        }
        let satisfied = cs
            .iter()
            .filter(|c| verify_constraint(&response.payload, c))
            .count();
        (satisfied as f64) / (cs.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eoc_core::JouleCost;

    fn resp(s: &str) -> Response {
        Response {
            payload: s.to_string(),
            latency_ms: 1,
            joule_cost: JouleCost::estimated(1),
        }
    }

    #[tokio::test]
    async fn loads_builtin() {
        let cases = IfEval::new().load(DatasetSource::BuiltinSample).await.unwrap();
        assert!(!cases.is_empty());
    }

    #[test]
    fn keyword_at_least() {
        let c = IfEvalConstraint::KeywordAtLeast {
            keyword: "chloroplast".into(),
            min_count: 3,
        };
        assert!(verify_constraint(
            "chloroplast Chloroplast CHLOROPLAST",
            &c
        ));
        assert!(!verify_constraint("chloroplast chloroplast", &c));
    }

    #[test]
    fn exact_paragraphs() {
        let c = IfEvalConstraint::ExactParagraphs(2);
        assert!(verify_constraint("first.\n\nsecond.", &c));
        assert!(!verify_constraint("just one", &c));
        assert!(!verify_constraint("a\n\nb\n\nc", &c));
    }

    #[test]
    fn word_count() {
        let c = IfEvalConstraint::WordCount { min: 3, max: 5 };
        assert!(verify_constraint("a b c d", &c));
        assert!(!verify_constraint("a b", &c));
        assert!(!verify_constraint("a b c d e f", &c));
    }

    #[test]
    fn no_comma() {
        assert!(verify_constraint("hello world", &IfEvalConstraint::NoComma));
        assert!(!verify_constraint("hello, world", &IfEvalConstraint::NoComma));
    }

    #[test]
    fn all_caps_and_lower() {
        assert!(verify_constraint("PARIS", &IfEvalConstraint::AllCaps));
        assert!(!verify_constraint("Paris", &IfEvalConstraint::AllCaps));
        assert!(verify_constraint("tokyo", &IfEvalConstraint::AllLowercase));
        assert!(!verify_constraint("Tokyo", &IfEvalConstraint::AllLowercase));
    }

    #[test]
    fn starts_ends() {
        assert!(verify_constraint(
            "The result is: 96",
            &IfEvalConstraint::StartsWith("The result is:".into())
        ));
        assert!(verify_constraint(
            "...That is all.",
            &IfEvalConstraint::EndsWith("That is all.".into())
        ));
    }

    #[test]
    fn bullets_and_forbid() {
        let three = IfEvalConstraint::ExactBullets(3);
        assert!(verify_constraint("- a\n- b\n- c", &three));
        assert!(!verify_constraint("- a\n- b", &three));
        let forbid = IfEvalConstraint::ForbidKeywords(vec!["beach".into(), "sand".into()]);
        assert!(verify_constraint("palm trees and coconuts", &forbid));
        assert!(!verify_constraint("the beach is lovely", &forbid));
    }

    #[tokio::test]
    async fn scores_fractional() {
        let h = IfEval::new();
        let exp = ExpectedAnswer::Constraints(vec![
            IfEvalConstraint::NoComma,
            IfEvalConstraint::WordCount { min: 1, max: 100 },
            IfEvalConstraint::AllCaps,
        ]);
        // No comma + word count ok, but not all caps => 2/3.
        let s = h.score(&resp("hello world"), &exp).await;
        assert!((s - 2.0 / 3.0).abs() < 1e-9);
    }
}
