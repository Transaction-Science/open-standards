//! Winogrande — large-scale Winograd Schema Challenge (Sakaguchi et al. 2019).
//!
//! Binary pronoun resolution: pick which of two options fits the
//! blanked `_` in a sentence.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::builtin_samples;
use crate::error::Result;
use crate::harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, Metric, Response, load_raw,
};

/// Winogrande harness.
pub struct Winogrande;

impl Winogrande {
    /// Create a new Winogrande harness.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Winogrande {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct RawRow {
    id: String,
    sentence: String,
    option1: String,
    option2: String,
    answer: String,
}

#[async_trait]
impl Harness for Winogrande {
    fn name(&self) -> &'static str {
        "winogrande"
    }

    fn metric(&self) -> Metric {
        Metric::Accuracy
    }

    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>> {
        let raw = load_raw(source, builtin_samples::WINOGRANDE).await?;
        let rows: Vec<RawRow> = serde_json::from_str(&raw)?;
        Ok(rows
            .into_iter()
            .map(|r| {
                let prompt = format!(
                    "Fill the blank in the sentence below.\n\nSentence: {}\n\nOption 1: {}\nOption 2: {}\n\nAnswer with '1' or '2' only.",
                    r.sentence, r.option1, r.option2
                );
                EvalCase {
                    id: r.id,
                    prompt,
                    expected: ExpectedAnswer::Text(r.answer),
                    dataset: "winogrande".to_string(),
                    subject: None,
                }
            })
            .collect())
    }

    async fn score(&self, response: &Response, expected: &ExpectedAnswer) -> f64 {
        let ExpectedAnswer::Text(want) = expected else {
            return 0.0;
        };
        for tok in response
            .payload
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
        {
            if tok == "1" || tok == "2" {
                return if tok == want { 1.0 } else { 0.0 };
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
        let cases = Winogrande::new()
            .load(DatasetSource::BuiltinSample)
            .await
            .unwrap();
        assert!(!cases.is_empty());
    }

    #[tokio::test]
    async fn scores_one_two() {
        let h = Winogrande::new();
        let exp = ExpectedAnswer::Text("1".into());
        assert_eq!(h.score(&resp("1"), &exp).await, 1.0);
        assert_eq!(h.score(&resp("Option 1"), &exp).await, 1.0);
        assert_eq!(h.score(&resp("2"), &exp).await, 0.0);
    }
}
