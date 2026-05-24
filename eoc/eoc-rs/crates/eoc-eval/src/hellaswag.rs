//! HellaSwag — commonsense sentence completion (Zellers et al. 2019).
//!
//! 4-way multiple choice; the model picks the most plausible continuation.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::builtin_samples;
use crate::error::Result;
use crate::harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, Metric, Response, load_raw,
};
use crate::mcq::{extract_letter, format_mcq_prompt};

/// HellaSwag harness.
pub struct HellaSwag;

impl HellaSwag {
    /// Create a new HellaSwag harness.
    pub fn new() -> Self {
        Self
    }
}

impl Default for HellaSwag {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct RawRow {
    id: String,
    context: String,
    choices: Vec<String>,
    answer: String,
}

#[async_trait]
impl Harness for HellaSwag {
    fn name(&self) -> &'static str {
        "hellaswag"
    }

    fn metric(&self) -> Metric {
        Metric::Accuracy
    }

    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>> {
        let raw = load_raw(source, builtin_samples::HELLASWAG).await?;
        let rows: Vec<RawRow> = serde_json::from_str(&raw)?;
        Ok(rows
            .into_iter()
            .map(|r| {
                let q = format!(
                    "Choose the most plausible continuation of the following:\n\n{}",
                    r.context
                );
                EvalCase {
                    id: r.id,
                    prompt: format_mcq_prompt(&q, &r.choices),
                    expected: ExpectedAnswer::Choice(r.answer),
                    dataset: "hellaswag".to_string(),
                    subject: None,
                }
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

    #[tokio::test]
    async fn loads_builtin() {
        let cases = HellaSwag::new().load(DatasetSource::BuiltinSample).await.unwrap();
        assert!(!cases.is_empty());
    }

    #[tokio::test]
    async fn scores_letter() {
        let h = HellaSwag::new();
        let r = Response { payload: "A".to_string(), latency_ms: 1, joule_cost: JouleCost::estimated(1) };
        assert_eq!(h.score(&r, &ExpectedAnswer::Choice("A".into())).await, 1.0);
        assert_eq!(h.score(&r, &ExpectedAnswer::Choice("B".into())).await, 0.0);
    }
}
