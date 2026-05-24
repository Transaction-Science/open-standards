//! AlpacaEval — head-to-head LLM-as-judge against a reference response.
//!
//! Unlike the rule-based harnesses in this crate, AlpacaEval requires an
//! external judge model (Anthropic Claude, GPT-4, etc.). We ship the
//! trait surface, the dataset loader, and a reference scoring rule that
//! compares responses to the reference via ROUGE-L-style token overlap;
//! the canonical leaderboard score requires plugging a real `Judge`
//! into [`AlpacaEval::with_judge`].
//!
//! ## How to use a real judge
//!
//! ```ignore
//! use eoc_eval::alpaca_eval::{AlpacaEval, Judge, JudgeVerdict};
//! use async_trait::async_trait;
//!
//! struct AnthropicJudge { /* api client */ }
//! #[async_trait]
//! impl Judge for AnthropicJudge {
//!     async fn judge(&self, prompt: &str, candidate: &str, reference: &str) -> JudgeVerdict {
//!         // call the API, return a JudgeVerdict::CandidatePreferred / TiePreferred / ReferencePreferred
//!         unimplemented!()
//!     }
//! }
//! let alpaca = AlpacaEval::new().with_judge(Box::new(AnthropicJudge { /* ... */ }));
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::builtin_samples;
use crate::error::Result;
use crate::harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, Metric, Response, load_raw,
};

/// Verdict returned by an external judge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JudgeVerdict {
    /// The candidate response was preferred.
    CandidatePreferred,
    /// The reference was preferred.
    ReferencePreferred,
    /// The judge could not decide.
    Tie,
}

/// External LLM-as-judge interface.
#[async_trait]
pub trait Judge: Send + Sync {
    /// Judge a candidate against a reference for `prompt`.
    async fn judge(
        &self,
        prompt: &str,
        candidate: &str,
        reference: &str,
    ) -> JudgeVerdict;
}

/// AlpacaEval harness.
pub struct AlpacaEval {
    judge: Option<Box<dyn Judge>>,
}

impl AlpacaEval {
    /// Create a new AlpacaEval harness without an external judge.
    pub fn new() -> Self {
        Self { judge: None }
    }

    /// Attach an external judge.
    pub fn with_judge(mut self, judge: Box<dyn Judge>) -> Self {
        self.judge = Some(judge);
        self
    }
}

impl Default for AlpacaEval {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct RawRow {
    id: String,
    prompt: String,
    reference: String,
}

/// Token-overlap fallback when no external judge is configured. Returns
/// the Jaccard similarity between lowercased word sets, which is a
/// pragmatic stand-in for ROUGE-L F1 on short responses.
pub fn token_overlap(candidate: &str, reference: &str) -> f64 {
    let cset: HashSet<String> = candidate
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();
    let rset: HashSet<String> = reference
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();
    if cset.is_empty() && rset.is_empty() {
        return 0.0;
    }
    let inter = cset.intersection(&rset).count() as f64;
    let union = cset.union(&rset).count() as f64;
    inter / union
}

#[async_trait]
impl Harness for AlpacaEval {
    fn name(&self) -> &'static str {
        "alpaca_eval"
    }

    fn metric(&self) -> Metric {
        Metric::Custom("win_rate")
    }

    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>> {
        let raw = load_raw(source, builtin_samples::ALPACA_EVAL).await?;
        let rows: Vec<RawRow> = serde_json::from_str(&raw)?;
        Ok(rows
            .into_iter()
            .map(|r| EvalCase {
                id: r.id,
                prompt: r.prompt,
                expected: ExpectedAnswer::Reference(r.reference),
                dataset: "alpaca_eval".to_string(),
                subject: None,
            })
            .collect())
    }

    async fn score(&self, response: &Response, expected: &ExpectedAnswer) -> f64 {
        let ExpectedAnswer::Reference(reference) = expected else {
            return 0.0;
        };
        if let Some(judge) = &self.judge {
            return match judge.judge("", &response.payload, reference).await {
                JudgeVerdict::CandidatePreferred => 1.0,
                JudgeVerdict::Tie => 0.5,
                JudgeVerdict::ReferencePreferred => 0.0,
            };
        }
        // Fallback: token overlap, thresholded so a hostile-or-blank
        // response scores 0 and an exact match scores 1.
        token_overlap(&response.payload, reference)
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
        let cases = AlpacaEval::new().load(DatasetSource::BuiltinSample).await.unwrap();
        assert!(!cases.is_empty());
    }

    #[tokio::test]
    async fn fallback_overlap_grades() {
        let h = AlpacaEval::new();
        let exp = ExpectedAnswer::Reference("the cat sat on the mat".into());
        let perfect = h.score(&resp("the cat sat on the mat"), &exp).await;
        assert!((perfect - 1.0).abs() < 1e-9);
        let none = h.score(&resp("totally unrelated XYZ"), &exp).await;
        assert!(none < 0.2);
    }

    struct StubJudge(JudgeVerdict);
    #[async_trait]
    impl Judge for StubJudge {
        async fn judge(&self, _p: &str, _c: &str, _r: &str) -> JudgeVerdict {
            self.0
        }
    }

    #[tokio::test]
    async fn judge_drives_score() {
        let h = AlpacaEval::new().with_judge(Box::new(StubJudge(JudgeVerdict::CandidatePreferred)));
        let exp = ExpectedAnswer::Reference("ref".into());
        assert_eq!(h.score(&resp("cand"), &exp).await, 1.0);

        let h = AlpacaEval::new().with_judge(Box::new(StubJudge(JudgeVerdict::ReferencePreferred)));
        assert_eq!(h.score(&resp("cand"), &exp).await, 0.0);

        let h = AlpacaEval::new().with_judge(Box::new(StubJudge(JudgeVerdict::Tie)));
        assert_eq!(h.score(&resp("cand"), &exp).await, 0.5);
    }
}
