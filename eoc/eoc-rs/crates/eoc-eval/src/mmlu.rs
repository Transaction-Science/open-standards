//! MMLU — Massive Multitask Language Understanding (Hendrycks et al. 2020).
//!
//! 57 academic subjects, 4-way multiple choice. Score: exact letter match
//! after canonicalisation.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::builtin_samples;
use crate::error::Result;
use crate::harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, Metric, Response, load_raw,
};
use crate::mcq::{extract_letter, format_mcq_prompt};

/// MMLU harness.
pub struct Mmlu;

impl Mmlu {
    /// Create a new MMLU harness instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Mmlu {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct RawRow {
    id: String,
    subject: String,
    question: String,
    choices: Vec<String>,
    answer: String,
}

#[async_trait]
impl Harness for Mmlu {
    fn name(&self) -> &'static str {
        "mmlu"
    }

    fn metric(&self) -> Metric {
        Metric::Accuracy
    }

    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>> {
        let raw = load_raw(source, builtin_samples::MMLU).await?;
        let rows: Vec<RawRow> = serde_json::from_str(&raw)?;
        Ok(rows
            .into_iter()
            .map(|r| EvalCase {
                id: r.id,
                prompt: format_mcq_prompt(&r.question, &r.choices),
                expected: ExpectedAnswer::Choice(r.answer),
                dataset: "mmlu".to_string(),
                subject: Some(r.subject),
            })
            .collect())
    }

    async fn score(&self, response: &Response, expected: &ExpectedAnswer) -> f64 {
        let ExpectedAnswer::Choice(correct) = expected else {
            return 0.0;
        };
        match extract_letter(&response.payload, 4) {
            Some(l) if l.eq_ignore_ascii_case(correct) => 1.0,
            _ => 0.0,
        }
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
    async fn loads_builtin_sample() {
        let cases = Mmlu::new()
            .load(DatasetSource::BuiltinSample)
            .await
            .expect("builtin load");
        assert_eq!(cases.len(), 20);
        assert!(cases.iter().any(|c| c.subject.as_deref() == Some("abstract_algebra")));
    }

    #[tokio::test]
    async fn scores_letter_responses() {
        let h = Mmlu::new();
        let expected = ExpectedAnswer::Choice("B".to_string());
        assert_eq!(h.score(&resp("B"), &expected).await, 1.0);
        assert_eq!(h.score(&resp("(B)"), &expected).await, 1.0);
        assert_eq!(h.score(&resp("**B**"), &expected).await, 1.0);
        assert_eq!(h.score(&resp("B."), &expected).await, 1.0);
        assert_eq!(h.score(&resp("The answer is B."), &expected).await, 1.0);
        assert_eq!(h.score(&resp("A"), &expected).await, 0.0);
        assert_eq!(h.score(&resp("nonsense"), &expected).await, 0.0);
    }
}
