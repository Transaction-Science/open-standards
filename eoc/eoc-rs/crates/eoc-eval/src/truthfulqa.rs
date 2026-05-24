//! TruthfulQA — adversarial questions about common misconceptions
//! (Lin et al. 2021).
//!
//! Reports both MC1 (single-correct accuracy) and MC2 (probability mass on
//! correct answer, here approximated as 1.0 if the model selects the
//! correct letter and 0.0 otherwise — full MC2 requires per-choice
//! logprobs from the backend, which isn't part of the `NeuralBackend`
//! surface). MC1 is the default metric.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::builtin_samples;
use crate::error::Result;
use crate::harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, Metric, Response, load_raw,
};
use crate::mcq::{extract_letter, format_mcq_prompt, LETTERS};

/// TruthfulQA harness (MC1).
pub struct TruthfulQa;

impl TruthfulQa {
    /// Create a new TruthfulQA harness.
    pub fn new() -> Self {
        Self
    }
}

impl Default for TruthfulQa {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct RawRow {
    id: String,
    question: String,
    choices: Vec<String>,
    answer: String,
}

#[async_trait]
impl Harness for TruthfulQa {
    fn name(&self) -> &'static str {
        "truthfulqa"
    }

    fn metric(&self) -> Metric {
        Metric::Custom("mc1")
    }

    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>> {
        let raw = load_raw(source, builtin_samples::TRUTHFULQA).await?;
        let rows: Vec<RawRow> = serde_json::from_str(&raw)?;
        Ok(rows
            .into_iter()
            .map(|r| {
                let n = r.choices.len();
                let distractors: Vec<String> = LETTERS
                    .iter()
                    .take(n)
                    .filter(|l| !l.eq_ignore_ascii_case(&r.answer))
                    .map(|s| s.to_string())
                    .collect();
                EvalCase {
                    id: r.id,
                    prompt: format_mcq_prompt(&r.question, &r.choices),
                    expected: ExpectedAnswer::Truthful {
                        correct: r.answer,
                        distractors,
                    },
                    dataset: "truthfulqa".to_string(),
                    subject: None,
                }
            })
            .collect())
    }

    async fn score(&self, response: &Response, expected: &ExpectedAnswer) -> f64 {
        let ExpectedAnswer::Truthful { correct, .. } = expected else {
            return 0.0;
        };
        match extract_letter(&response.payload, 10) {
            Some(l) if l.eq_ignore_ascii_case(correct) => 1.0,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eoc_core::JouleCost;

    #[tokio::test]
    async fn loads_builtin() {
        let cases = TruthfulQa::new()
            .load(DatasetSource::BuiltinSample)
            .await
            .unwrap();
        assert!(!cases.is_empty());
        for c in &cases {
            assert!(matches!(c.expected, ExpectedAnswer::Truthful { .. }));
        }
    }

    #[tokio::test]
    async fn scores_letter() {
        let h = TruthfulQa::new();
        let exp = ExpectedAnswer::Truthful {
            correct: "A".to_string(),
            distractors: vec!["B".into(), "C".into(), "D".into()],
        };
        let r = Response {
            payload: "A".to_string(),
            latency_ms: 1,
            joule_cost: JouleCost::estimated(1),
        };
        assert_eq!(h.score(&r, &exp).await, 1.0);
        let bad = Response {
            payload: "C".to_string(),
            latency_ms: 1,
            joule_cost: JouleCost::estimated(1),
        };
        assert_eq!(h.score(&bad, &exp).await, 0.0);
    }
}
