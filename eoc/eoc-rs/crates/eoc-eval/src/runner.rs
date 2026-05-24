//! End-to-end eval runner: drives a `Harness` through a `NeuralBackend`
//! and emits an `EvalReport` carrying score + joules-per-correct +
//! latency percentiles.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use eoc_core::Query;
use eoc_neural::NeuralBackend;

use crate::error::Result;
use crate::harness::{DatasetSource, Harness, Metric, Response};

/// Configuration for running a harness against a backend.
pub struct EvalRunner {
    /// The harness to run.
    pub harness: Box<dyn Harness>,
    /// The backend under test.
    pub backend: Arc<dyn NeuralBackend>,
    /// Number of cases to evaluate in parallel. `1` is sequential.
    pub batch_size: usize,
    /// Cap on the number of cases (`None` = run all).
    pub max_cases: Option<usize>,
    /// Identifier for the model under test. Pure label, used in the
    /// report.
    pub model: String,
    /// Source to load the harness's dataset from.
    pub source: DatasetSource,
}

impl EvalRunner {
    /// Build a sequential runner with the builtin sample dataset.
    pub fn new(harness: Box<dyn Harness>, backend: Arc<dyn NeuralBackend>) -> Self {
        Self {
            harness,
            backend,
            batch_size: 1,
            max_cases: None,
            model: "unknown".to_string(),
            source: DatasetSource::BuiltinSample,
        }
    }

    /// Set the model label.
    pub fn with_model(mut self, m: impl Into<String>) -> Self {
        self.model = m.into();
        self
    }

    /// Cap the number of cases.
    pub fn with_max_cases(mut self, n: usize) -> Self {
        self.max_cases = Some(n);
        self
    }

    /// Override the dataset source.
    pub fn with_source(mut self, s: DatasetSource) -> Self {
        self.source = s;
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, b: usize) -> Self {
        self.batch_size = b.max(1);
        self
    }

    /// Drive every case through the backend and aggregate the report.
    pub async fn run(self) -> Result<EvalReport> {
        let cases = self.harness.load(self.source.clone()).await?;
        let mut cases = cases;
        if let Some(n) = self.max_cases {
            cases.truncate(n);
        }
        let total_cases = cases.len();

        let mut total_score = 0.0f64;
        let mut total_microjoules: u128 = 0;
        let mut latencies: Vec<u64> = Vec::with_capacity(total_cases);

        for case in &cases {
            let q = Query::new(case.prompt.clone());
            let start = std::time::Instant::now();
            let core_resp = self.backend.infer(&q).await;
            let latency_ms = start.elapsed().as_millis().min(u64::MAX as u128) as u64;
            total_microjoules += u128::from(core_resp.joule_cost.microjoules);
            latencies.push(latency_ms);
            let response = Response::from_core(core_resp, latency_ms);
            let s = self.harness.score(&response, &case.expected).await;
            total_score += s.clamp(0.0, 1.0);
        }

        let metric = self.harness.metric();
        let score = if total_cases == 0 {
            0.0
        } else {
            total_score / (total_cases as f64)
        };
        let correct = total_score.round() as usize;
        // Joules-per-correct in *micro-joules*-per-correct cast to u64,
        // matching the rest of EOC's accounting which reports raw uJ
        // and converts at the display layer.
        let joules_per_correct = if correct == 0 {
            u64::MAX
        } else {
            (total_microjoules / (correct as u128)).min(u64::MAX as u128) as u64
        };

        latencies.sort_unstable();
        let latency_p50_ms = percentile(&latencies, 0.50);
        let latency_p95_ms = percentile(&latencies, 0.95);

        Ok(EvalReport {
            harness: self.harness.name().to_string(),
            model: self.model,
            total_cases,
            correct,
            score,
            metric,
            joules_per_correct,
            latency_p50_ms,
            latency_p95_ms,
        })
    }
}

/// Aggregated report for one harness run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// Stable harness id (`"mmlu"`, ...).
    pub harness: String,
    /// Label for the model under test.
    pub model: String,
    /// Number of cases scored.
    pub total_cases: usize,
    /// Number of cases scored at >=0.5. For binary harnesses this is the
    /// correct count; for graded harnesses (IFEval, TruthfulQA MC2) it
    /// is the rounded count of "mostly-correct" cases.
    pub correct: usize,
    /// Mean per-case score in `[0.0, 1.0]`.
    pub score: f64,
    /// The metric the harness reports.
    pub metric: Metric,
    /// Micro-joules per correct case. `u64::MAX` when zero correct.
    pub joules_per_correct: u64,
    /// Latency p50 in milliseconds.
    pub latency_p50_ms: u64,
    /// Latency p95 in milliseconds.
    pub latency_p95_ms: u64,
}

/// Nearest-rank percentile (no interpolation).
fn percentile(sorted: &[u64], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let n = sorted.len();
    let rank = (q * n as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(n - 1);
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{EvalCase, ExpectedAnswer};
    use async_trait::async_trait;
    use eoc_core::{JouleCost, Query as CoreQuery, Response as CoreResponse, Stage};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // A NeuralBackend that returns the correct answer ("B") for the
    // first half of the calls and the wrong answer ("A") for the rest.
    struct HalfRight {
        n: AtomicUsize,
        switchpoint: usize,
        correct: String,
        wrong: String,
    }
    #[async_trait]
    impl NeuralBackend for HalfRight {
        async fn infer(&self, q: &CoreQuery) -> CoreResponse {
            let idx = self.n.fetch_add(1, Ordering::Relaxed);
            let payload = if idx < self.switchpoint { &self.correct } else { &self.wrong };
            CoreResponse::new(
                q.id,
                payload.clone(),
                Stage::Neural,
                JouleCost::estimated(1_000_000),
            )
        }
    }

    // A trivial harness that grades exact-match on a fixed answer.
    struct FixedHarness {
        n_cases: usize,
        correct: &'static str,
    }
    #[async_trait]
    impl Harness for FixedHarness {
        fn name(&self) -> &'static str { "fixed" }
        fn metric(&self) -> Metric { Metric::Accuracy }
        async fn load(&self, _src: DatasetSource) -> Result<Vec<EvalCase>> {
            Ok((0..self.n_cases)
                .map(|i| EvalCase {
                    id: format!("c-{i}"),
                    prompt: format!("q{i}"),
                    expected: ExpectedAnswer::Text(self.correct.into()),
                    dataset: "fixed".into(),
                    subject: None,
                })
                .collect())
        }
        async fn score(&self, resp: &Response, expected: &ExpectedAnswer) -> f64 {
            match expected {
                ExpectedAnswer::Text(t) if resp.payload == *t => 1.0,
                _ => 0.0,
            }
        }
    }

    #[tokio::test]
    async fn runner_lands_in_expected_range() {
        let harness = Box::new(FixedHarness { n_cases: 10, correct: "yes" });
        let backend = Arc::new(HalfRight {
            n: AtomicUsize::new(0),
            switchpoint: 5,
            correct: "yes".into(),
            wrong: "no".into(),
        });
        let runner = EvalRunner::new(harness, backend).with_model("test");
        let report = runner.run().await.unwrap();
        assert_eq!(report.total_cases, 10);
        assert_eq!(report.correct, 5);
        assert!((report.score - 0.5).abs() < 1e-9);
        assert!(report.score >= 0.4 && report.score <= 0.6);
        // 10 cases * 1_000_000 uJ = 10_000_000 uJ; / 5 correct = 2_000_000.
        assert_eq!(report.joules_per_correct, 2_000_000);
    }

    #[tokio::test]
    async fn report_roundtrips_through_serde_json() {
        let report = EvalReport {
            harness: "mmlu".into(),
            model: "test".into(),
            total_cases: 1,
            correct: 1,
            score: 1.0,
            metric: Metric::Accuracy,
            joules_per_correct: 100,
            latency_p50_ms: 1,
            latency_p95_ms: 1,
        };
        let json = serde_json::to_string(&report).unwrap();
        let back: EvalReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.harness, "mmlu");
        assert_eq!(back.correct, 1);
    }
}
