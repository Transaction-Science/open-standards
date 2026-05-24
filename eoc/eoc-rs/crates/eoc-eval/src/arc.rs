//! ARC — AI2 Reasoning Challenge (Clark et al. 2018).
//!
//! Two subsets: Easy and Challenge. 4-way multiple choice grade-school
//! science questions.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::builtin_samples;
use crate::error::Result;
use crate::harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, Metric, Response, load_raw,
};
use crate::mcq::{extract_letter, format_mcq_prompt};

/// ARC harness — mixed Easy + Challenge by default.
pub struct Arc;

impl Arc {
    /// Create a new ARC harness.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Arc {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct RawRow {
    id: String,
    subset: String,
    question: String,
    choices: Vec<String>,
    answer: String,
}

#[async_trait]
impl Harness for Arc {
    fn name(&self) -> &'static str {
        "arc"
    }

    fn metric(&self) -> Metric {
        Metric::Accuracy
    }

    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>> {
        let raw = load_raw(source, builtin_samples::ARC).await?;
        let rows: Vec<RawRow> = serde_json::from_str(&raw)?;
        Ok(rows
            .into_iter()
            .map(|r| EvalCase {
                id: r.id,
                prompt: format_mcq_prompt(&r.question, &r.choices),
                expected: ExpectedAnswer::Choice(r.answer),
                dataset: "arc".to_string(),
                subject: Some(r.subset),
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
    async fn loads_both_subsets() {
        let cases = Arc::new().load(DatasetSource::BuiltinSample).await.unwrap();
        assert!(cases.iter().any(|c| c.subject.as_deref() == Some("easy")));
        assert!(cases.iter().any(|c| c.subject.as_deref() == Some("challenge")));
    }

    #[tokio::test]
    async fn scores_letter() {
        let h = Arc::new();
        let r = Response { payload: "C".to_string(), latency_ms: 1, joule_cost: JouleCost::estimated(1) };
        assert_eq!(h.score(&r, &ExpectedAnswer::Choice("C".into())).await, 1.0);
    }
}
