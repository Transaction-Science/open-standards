//! BIG-Bench Hard — 23 challenging BIG-Bench tasks.
//!
//! Each task is multiple-choice with a variable number of options (2-6).
//! Score: accuracy.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::builtin_samples;
use crate::error::Result;
use crate::harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, Metric, Response, load_raw,
};
use crate::mcq::{extract_letter, format_mcq_prompt};

/// BIG-Bench Hard harness.
pub struct Bbh;

impl Bbh {
    /// Create a new BBH harness.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Bbh {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct RawRow {
    id: String,
    task: String,
    question: String,
    choices: Vec<String>,
    answer: String,
}

#[async_trait]
impl Harness for Bbh {
    fn name(&self) -> &'static str {
        "bbh"
    }

    fn metric(&self) -> Metric {
        Metric::Accuracy
    }

    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>> {
        let raw = load_raw(source, builtin_samples::BBH).await?;
        let rows: Vec<RawRow> = serde_json::from_str(&raw)?;
        Ok(rows
            .into_iter()
            .map(|r| EvalCase {
                id: r.id,
                prompt: format_mcq_prompt(&r.question, &r.choices),
                expected: ExpectedAnswer::Choice(r.answer),
                dataset: "bbh".to_string(),
                subject: Some(r.task),
            })
            .collect())
    }

    async fn score(&self, response: &Response, expected: &ExpectedAnswer) -> f64 {
        let ExpectedAnswer::Choice(correct) = expected else {
            return 0.0;
        };
        // BBH max options = 6.
        match extract_letter(&response.payload, 6) {
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
    async fn loads_all_23_tasks() {
        let cases = Bbh::new().load(DatasetSource::BuiltinSample).await.unwrap();
        let mut tasks: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for c in &cases {
            if let Some(s) = &c.subject {
                tasks.insert(s.clone());
            }
        }
        assert_eq!(tasks.len(), 23, "BBH should cover 23 distinct tasks");
    }

    #[tokio::test]
    async fn scores_letter() {
        let h = Bbh::new();
        let expected = ExpectedAnswer::Choice("C".to_string());
        assert_eq!(h.score(&resp("C"), &expected).await, 1.0);
        assert_eq!(h.score(&resp("A"), &expected).await, 0.0);
    }
}
