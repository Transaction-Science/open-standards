//! AGIEval — academic / professional standardised tests (Zhong et al. 2023).
//!
//! Subset covering SAT, LSAT, GMAT, GRE and equivalent Chinese exams.
//! Multiple-choice with 4 or 5 options depending on the exam; we treat
//! the harness as accuracy on the option letter.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::builtin_samples;
use crate::error::Result;
use crate::harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, Metric, Response, load_raw,
};
use crate::mcq::{extract_letter, format_mcq_prompt};

/// AGIEval harness.
pub struct AgiEval;

impl AgiEval {
    /// Create a new AGIEval harness.
    pub fn new() -> Self {
        Self
    }
}

impl Default for AgiEval {
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
impl Harness for AgiEval {
    fn name(&self) -> &'static str {
        "agi_eval"
    }

    fn metric(&self) -> Metric {
        Metric::Accuracy
    }

    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>> {
        let raw = load_raw(source, builtin_samples::AGI_EVAL).await?;
        let rows: Vec<RawRow> = serde_json::from_str(&raw)?;
        Ok(rows
            .into_iter()
            .map(|r| EvalCase {
                id: r.id,
                prompt: format_mcq_prompt(&r.question, &r.choices),
                expected: ExpectedAnswer::Choice(r.answer),
                dataset: "agi_eval".to_string(),
                subject: Some(r.subject),
            })
            .collect())
    }

    async fn score(&self, response: &Response, expected: &ExpectedAnswer) -> f64 {
        let ExpectedAnswer::Choice(correct) = expected else {
            return 0.0;
        };
        match extract_letter(&response.payload, 5) {
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
        let cases = AgiEval::new().load(DatasetSource::BuiltinSample).await.unwrap();
        assert!(!cases.is_empty());
    }

    #[tokio::test]
    async fn scores_letter() {
        let h = AgiEval::new();
        let r = Response { payload: "C".to_string(), latency_ms: 1, joule_cost: JouleCost::estimated(1) };
        assert_eq!(h.score(&r, &ExpectedAnswer::Choice("C".into())).await, 1.0);
    }
}
