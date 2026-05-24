//! MATH — competition mathematics (Hendrycks et al. 2021).
//!
//! Metric: equivalence on the final boxed answer (`\boxed{...}`). We
//! accept either a boxed expression in the response or — as a fallback
//! — the last numeric token, after a light normalisation that strips
//! whitespace, dollar signs, and trailing punctuation.

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use crate::builtin_samples;
use crate::error::Result;
use crate::harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, Metric, Response, load_raw,
};

/// MATH harness.
pub struct Math;

impl Math {
    /// Create a new MATH harness.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Math {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct RawRow {
    id: String,
    subject: String,
    #[allow(dead_code)]
    level: u32,
    question: String,
    answer: String,
}

/// Strip a `\boxed{...}` wrapper out of `text` and return the inner
/// expression with surrounding whitespace removed.
pub fn extract_boxed(text: &str) -> Option<String> {
    // Find the last occurrence of \boxed{ and balance-match its braces.
    let idx = text.rfind("\\boxed{")?;
    let start = idx + "\\boxed{".len();
    let mut depth = 1i32;
    let bytes = text.as_bytes();
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..i].trim().to_string());
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Normalise a MATH answer string for equivalence comparison.
pub fn normalise_math(answer: &str) -> String {
    let mut s = answer.trim().to_string();
    // Strip enclosing dollars (display math).
    while s.starts_with('$') && s.ends_with('$') && s.len() > 1 {
        s = s[1..s.len() - 1].trim().to_string();
    }
    // Drop common cosmetic differences.
    s.retain(|c| !c.is_whitespace());
    // Trailing period.
    if s.ends_with('.') {
        s.pop();
    }
    s
}

#[async_trait]
impl Harness for Math {
    fn name(&self) -> &'static str {
        "math"
    }

    fn metric(&self) -> Metric {
        Metric::ExactMatch
    }

    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>> {
        let raw = load_raw(source, builtin_samples::MATH).await?;
        let rows: Vec<RawRow> = serde_json::from_str(&raw)?;
        Ok(rows
            .into_iter()
            .map(|r| {
                let prompt = format!(
                    "{}\n\nPut your final answer in \\boxed{{...}}.",
                    r.question.trim()
                );
                EvalCase {
                    id: r.id,
                    prompt,
                    expected: ExpectedAnswer::Number(r.answer),
                    dataset: "math".to_string(),
                    subject: Some(r.subject),
                }
            })
            .collect())
    }

    async fn score(&self, response: &Response, expected: &ExpectedAnswer) -> f64 {
        let ExpectedAnswer::Number(want) = expected else {
            return 0.0;
        };
        let want_norm = normalise_math(want);
        // Prefer a boxed expression in the response.
        if let Some(boxed) = extract_boxed(&response.payload)
            && normalise_math(&boxed) == want_norm
        {
            return 1.0;
        }
        // Fallback: last bare integer / decimal.
        static RE: OnceLock<Regex> = OnceLock::new();
        let re = RE.get_or_init(|| Regex::new(r"-?\d+(?:\.\d+)?").unwrap());
        if let Some(m) = re.find_iter(&response.payload).last()
            && normalise_math(m.as_str()) == want_norm
        {
            return 1.0;
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
        let cases = Math::new().load(DatasetSource::BuiltinSample).await.unwrap();
        assert!(!cases.is_empty());
    }

    #[test]
    fn boxed_extraction() {
        assert_eq!(
            extract_boxed("So the answer is \\boxed{42}.").as_deref(),
            Some("42"),
        );
        assert_eq!(
            extract_boxed("\\boxed{\\frac{5}{6}}").as_deref(),
            Some("\\frac{5}{6}"),
        );
    }

    #[tokio::test]
    async fn scores_boxed_and_bare() {
        let h = Math::new();
        let exp = ExpectedAnswer::Number("12".into());
        assert_eq!(h.score(&resp("\\boxed{12}"), &exp).await, 1.0);
        assert_eq!(h.score(&resp("So 6*2 = 12"), &exp).await, 1.0);
        assert_eq!(h.score(&resp("13"), &exp).await, 0.0);

        let frac = ExpectedAnswer::Number("\\frac{5}{6}".into());
        assert_eq!(h.score(&resp("\\boxed{\\frac{5}{6}}"), &frac).await, 1.0);
    }
}
