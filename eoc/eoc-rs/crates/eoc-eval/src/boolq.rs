//! BoolQ — yes/no reading comprehension (Clark et al. 2019).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::builtin_samples;
use crate::error::Result;
use crate::harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, Metric, Response, load_raw,
};

/// BoolQ harness.
pub struct BoolQ;

impl BoolQ {
    /// Create a new BoolQ harness.
    pub fn new() -> Self {
        Self
    }
}

impl Default for BoolQ {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct RawRow {
    id: String,
    passage: String,
    question: String,
    answer: bool,
}

#[async_trait]
impl Harness for BoolQ {
    fn name(&self) -> &'static str {
        "boolq"
    }

    fn metric(&self) -> Metric {
        Metric::Accuracy
    }

    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>> {
        let raw = load_raw(source, builtin_samples::BOOLQ).await?;
        let rows: Vec<RawRow> = serde_json::from_str(&raw)?;
        Ok(rows
            .into_iter()
            .map(|r| {
                let prompt = format!(
                    "Passage: {}\n\nQuestion: {}\n\nAnswer with 'yes' or 'no' only.",
                    r.passage.trim(),
                    r.question.trim()
                );
                EvalCase {
                    id: r.id,
                    prompt,
                    expected: ExpectedAnswer::Text(if r.answer { "yes" } else { "no" }.into()),
                    dataset: "boolq".to_string(),
                    subject: None,
                }
            })
            .collect())
    }

    async fn score(&self, response: &Response, expected: &ExpectedAnswer) -> f64 {
        let ExpectedAnswer::Text(want) = expected else {
            return 0.0;
        };
        let lower = response.payload.trim().to_lowercase();
        let want = want.to_lowercase();
        // First yes/no token wins.
        for tok in lower.split(|c: char| !c.is_alphanumeric()).filter(|s| !s.is_empty()) {
            match tok {
                "yes" | "true" => return if want == "yes" { 1.0 } else { 0.0 },
                "no" | "false" => return if want == "no" { 1.0 } else { 0.0 },
                _ => continue,
            }
        }
        0.0
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
        let cases = BoolQ::new().load(DatasetSource::BuiltinSample).await.unwrap();
        assert!(!cases.is_empty());
    }

    #[tokio::test]
    async fn scores_yes_no() {
        let h = BoolQ::new();
        let yes = ExpectedAnswer::Text("yes".into());
        let no = ExpectedAnswer::Text("no".into());
        assert_eq!(h.score(&resp("yes"), &yes).await, 1.0);
        assert_eq!(h.score(&resp("YES."), &yes).await, 1.0);
        assert_eq!(h.score(&resp("The answer is yes."), &yes).await, 1.0);
        assert_eq!(h.score(&resp("no"), &yes).await, 0.0);
        assert_eq!(h.score(&resp("false"), &no).await, 1.0);
    }
}
