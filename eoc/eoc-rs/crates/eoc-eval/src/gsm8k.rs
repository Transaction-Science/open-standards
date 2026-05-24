//! GSM8K — grade-school math word problems (Cobbe et al. 2021).
//!
//! Metric: exact-match on the final numerical answer. The canonical
//! reference solution ends in `#### <number>`; we accept any response
//! whose last integer-or-decimal token equals the gold answer.

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use crate::builtin_samples;
use crate::error::Result;
use crate::harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, Metric, Response, load_raw,
};

/// GSM8K harness.
pub struct Gsm8K;

impl Gsm8K {
    /// Create a new GSM8K harness.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Gsm8K {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct RawRow {
    id: String,
    question: String,
    answer: String,
}

/// Pull the last numeric token (allowing thousands separators and
/// decimals) out of `text`, returning a normalised string with no
/// commas and a stripped trailing `.0`.
pub fn extract_final_number(text: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"-?\d[\d,]*(?:\.\d+)?").unwrap());
    let last = re.find_iter(text).last()?;
    let mut s = last.as_str().replace(',', "");
    if let Some(stripped) = s.strip_suffix(".0") {
        s = stripped.to_string();
    }
    Some(s)
}

#[async_trait]
impl Harness for Gsm8K {
    fn name(&self) -> &'static str {
        "gsm8k"
    }

    fn metric(&self) -> Metric {
        Metric::ExactMatch
    }

    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>> {
        let raw = load_raw(source, builtin_samples::GSM8K).await?;
        let rows: Vec<RawRow> = serde_json::from_str(&raw)?;
        Ok(rows
            .into_iter()
            .map(|r| {
                let prompt = format!(
                    "{}\n\nGive your final numeric answer on the last line.",
                    r.question.trim()
                );
                EvalCase {
                    id: r.id,
                    prompt,
                    expected: ExpectedAnswer::Number(r.answer),
                    dataset: "gsm8k".to_string(),
                    subject: None,
                }
            })
            .collect())
    }

    async fn score(&self, response: &Response, expected: &ExpectedAnswer) -> f64 {
        let ExpectedAnswer::Number(want) = expected else {
            return 0.0;
        };
        let want = want.replace(',', "");
        let want = want.strip_suffix(".0").unwrap_or(&want);
        match extract_final_number(&response.payload) {
            Some(got) if got == want => 1.0,
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
    async fn loads_builtin() {
        let cases = Gsm8K::new().load(DatasetSource::BuiltinSample).await.unwrap();
        assert!(!cases.is_empty());
    }

    #[tokio::test]
    async fn scores_final_number() {
        let h = Gsm8K::new();
        let exp = ExpectedAnswer::Number("72".into());
        assert_eq!(h.score(&resp("The answer is 72."), &exp).await, 1.0);
        assert_eq!(h.score(&resp("...\n#### 72"), &exp).await, 1.0);
        assert_eq!(h.score(&resp("intermediate 48, then 72.0"), &exp).await, 1.0);
        assert_eq!(h.score(&resp("Final: 1,000 - 928 = 72"), &exp).await, 1.0);
        assert_eq!(h.score(&resp("the answer is 24"), &exp).await, 0.0);
    }
}
