//! Benchmark runner — drives [`BenchCase`]s through the EOC cascade and
//! produces an aggregated [`BenchReport`].
//!
//! The runner is intentionally cascade-agnostic. When this crate is built
//! with the `eoc-rs` feature, the [`run`] function accepts a real
//! `eoc_cascade::Cascade`. Without that feature, only the data types and
//! aggregator are available — useful for ingesting JSON reports produced
//! by other deployments and for unit testing the aggregation logic.
//!
//! # The metric the harness exists to compute
//!
//! `joules-per-correct = total_joules / max(correct_answers, 1)`
//!
//! Lower is better. The cascade's job is to drive this number down by
//! resolving as many queries as possible at cache/kv/graph before falling
//! through to the (expensive) neural stage. The aggregator also reports
//! the share of resolutions per stage so a deployment can see *where* its
//! joules are going.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Stage abstraction
// ---------------------------------------------------------------------------
//
// The aggregator and serialised report don't depend on the eoc-rs
// implementation crates — they round-trip a stage by its stable string id
// (`"cache" | "kv" | "graph" | "neural"`). When the `eoc-rs` feature is
// on we import `eoc_core::Stage` directly and `From`/`Into` it to the
// string form at the wire boundary; when it's off we use this local mirror
// so the report type stays usable.

/// Stable string identifiers for the four cascade stages. Mirror of
/// `eoc_core::Stage::as_str()` — kept here so this crate compiles
/// stand-alone without `eoc-rs`.
pub mod stage_id {
    /// LRU / content-addressed cache.
    pub const CACHE: &str = "cache";
    /// Key-value / embedding-similarity lookup.
    pub const KV: &str = "kv";
    /// Graph / triple-store retrieval.
    pub const GRAPH: &str = "graph";
    /// Neural inference (last resort).
    pub const NEURAL: &str = "neural";
}

#[cfg(feature = "eoc-rs")]
pub use eoc_core::Stage;

/// A single benchmark case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BenchCase {
    /// Stable identifier for the case (e.g. `"mt-arith-01"`).
    pub id: String,
    /// The prompt fed to the cascade.
    pub prompt: String,
    /// Optional expected answer — used only for accuracy scoring. When
    /// `None`, the case is treated as inspection-only (counts towards
    /// joules but never towards `accuracy_pct`).
    pub expected: Option<String>,
    /// Dataset/suite label (e.g. `"mt"`, `"router"`). Lets a single
    /// report combine multiple suites without losing provenance.
    pub dataset: String,
}

impl BenchCase {
    /// Convenience constructor for tests / fixtures.
    pub fn new(
        id: impl Into<String>,
        prompt: impl Into<String>,
        expected: Option<String>,
        dataset: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            prompt: prompt.into(),
            expected,
            dataset: dataset.into(),
        }
    }
}

/// Result of running one [`BenchCase`] through the cascade.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchResult {
    /// Which case this result is for.
    pub case_id: String,
    /// The response payload returned by the cascade.
    pub response: String,
    /// Which stage resolved the query (stable string id).
    pub stage: String,
    /// Energy cost in micro-joules.
    pub joule_cost_microjoules: u64,
    /// Wall-clock latency for the resolution.
    pub latency_ms: u64,
    /// Whether the response matched the expected answer. `None` when
    /// the case had no `expected` field.
    pub accuracy: Option<bool>,
}

/// Aggregated metrics for a [`run`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchReport {
    /// Number of cases in the report.
    pub case_count: usize,
    /// Sum of all `joule_cost_microjoules`, expressed in joules.
    pub total_joules: f64,
    /// `total_joules / correct_answers`. `f64::INFINITY` when there are
    /// no correct answers — that result is dimensioned but uninformative.
    pub joules_per_correct: f64,
    /// Percentage of cases (with an `expected`) that matched.
    pub accuracy_pct: f64,
    /// Latency p50 in milliseconds.
    pub latency_p50_ms: u64,
    /// Latency p95 in milliseconds.
    pub latency_p95_ms: u64,
    /// Latency p99 in milliseconds.
    pub latency_p99_ms: u64,
    /// Count of cases resolved at each stage, keyed by stable stage id.
    pub stage_distribution: BTreeMap<String, usize>,
}

// ---------------------------------------------------------------------------
// run() — only available with the eoc-rs feature
// ---------------------------------------------------------------------------

/// Run a slice of [`BenchCase`]s through an `eoc_cascade::Cascade` and
/// collect a [`BenchResult`] per case.
///
/// Each case is resolved sequentially; the harness deliberately avoids
/// per-case parallelism so that the joule attribution reported by the
/// cascade is not polluted by overlapping work on the same accelerator.
#[cfg(feature = "eoc-rs")]
pub async fn run(cases: &[BenchCase], cascade: &eoc_cascade::Cascade) -> Vec<BenchResult> {
    let mut out = Vec::with_capacity(cases.len());
    for case in cases {
        let query = eoc_core::Query::new(case.prompt.clone());
        let started = std::time::Instant::now();
        let response = cascade.resolve(query).await;
        let latency_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let accuracy = case
            .expected
            .as_ref()
            .map(|e| matches_expected(&response.payload, e));
        out.push(BenchResult {
            case_id: case.id.clone(),
            response: response.payload,
            stage: response.stage.as_str().to_string(),
            joule_cost_microjoules: response.joule_cost.microjoules,
            latency_ms,
            accuracy,
        });
    }
    out
}

/// Default accuracy predicate: case-insensitive substring match between
/// the response payload and the expected answer. Bench suites with
/// stricter scoring rules can post-process [`BenchResult`]s before
/// calling [`aggregate`].
pub fn matches_expected(response: &str, expected: &str) -> bool {
    let r = response.to_lowercase();
    let e = expected.to_lowercase();
    r.contains(&e)
}

/// Aggregate raw results into a [`BenchReport`].
pub fn aggregate(results: &[BenchResult]) -> BenchReport {
    let case_count = results.len();
    let total_microjoules: u128 = results
        .iter()
        .map(|r| u128::from(r.joule_cost_microjoules))
        .sum();
    let total_joules = (total_microjoules as f64) / 1_000_000.0;

    let scored: Vec<bool> = results.iter().filter_map(|r| r.accuracy).collect();
    let correct = scored.iter().filter(|x| **x).count();
    let accuracy_pct = if scored.is_empty() {
        0.0
    } else {
        (correct as f64) * 100.0 / (scored.len() as f64)
    };
    let joules_per_correct = if correct == 0 {
        f64::INFINITY
    } else {
        total_joules / (correct as f64)
    };

    let mut latencies: Vec<u64> = results.iter().map(|r| r.latency_ms).collect();
    latencies.sort_unstable();
    let latency_p50_ms = percentile(&latencies, 0.50);
    let latency_p95_ms = percentile(&latencies, 0.95);
    let latency_p99_ms = percentile(&latencies, 0.99);

    let mut stage_distribution: BTreeMap<String, usize> = BTreeMap::new();
    for r in results {
        *stage_distribution.entry(r.stage.clone()).or_insert(0) += 1;
    }

    BenchReport {
        case_count,
        total_joules,
        joules_per_correct,
        accuracy_pct,
        latency_p50_ms,
        latency_p95_ms,
        latency_p99_ms,
        stage_distribution,
    }
}

/// Nearest-rank percentile (no interpolation). Returns 0 on empty input.
fn percentile(sorted: &[u64], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    // Nearest-rank: rank = ceil(q * n), 1-indexed.
    let n = sorted.len();
    let rank = (q * n as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(n - 1);
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn syn(id: &str, stage: &str, microjoules: u64, latency_ms: u64, correct: Option<bool>) -> BenchResult {
        BenchResult {
            case_id: id.to_string(),
            response: format!("response for {id}"),
            stage: stage.to_string(),
            joule_cost_microjoules: microjoules,
            latency_ms,
            accuracy: correct,
        }
    }

    #[test]
    fn aggregate_counts_stages() {
        let results = vec![
            syn("a", stage_id::CACHE, 0, 1, Some(true)),
            syn("b", stage_id::KV, 100, 2, Some(true)),
            syn("c", stage_id::GRAPH, 500, 3, Some(false)),
            syn("d", stage_id::NEURAL, 50_000_000, 80, Some(true)),
            syn("e", stage_id::CACHE, 0, 1, Some(true)),
        ];
        let report = aggregate(&results);
        assert_eq!(report.case_count, 5);
        assert_eq!(report.stage_distribution[stage_id::CACHE], 2);
        assert_eq!(report.stage_distribution[stage_id::KV], 1);
        assert_eq!(report.stage_distribution[stage_id::GRAPH], 1);
        assert_eq!(report.stage_distribution[stage_id::NEURAL], 1);
    }

    #[test]
    fn aggregate_joules_arithmetic() {
        let results = vec![
            syn("a", stage_id::CACHE, 0, 1, Some(true)),
            syn("b", stage_id::KV, 250_000, 2, Some(true)),
            syn("c", stage_id::NEURAL, 1_750_000, 10, Some(true)),
        ];
        let report = aggregate(&results);
        // (0 + 250_000 + 1_750_000) microjoules = 2 joules.
        assert!((report.total_joules - 2.0).abs() < 1e-9);
        // 2 joules / 3 correct ≈ 0.6667 J/correct.
        assert!((report.joules_per_correct - (2.0 / 3.0)).abs() < 1e-9);
        assert!((report.accuracy_pct - 100.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_accuracy_excludes_unscored_cases() {
        let results = vec![
            syn("scored-pass", stage_id::CACHE, 0, 1, Some(true)),
            syn("scored-fail", stage_id::NEURAL, 100, 2, Some(false)),
            syn("unscored", stage_id::KV, 50, 3, None),
        ];
        let report = aggregate(&results);
        // 1 of 2 scored cases passed.
        assert!((report.accuracy_pct - 50.0).abs() < 1e-9);
        // Stage distribution still counts the unscored case.
        assert_eq!(report.case_count, 3);
        assert_eq!(report.stage_distribution[stage_id::KV], 1);
    }

    #[test]
    fn joules_per_correct_is_infinity_with_zero_correct() {
        let results = vec![
            syn("fail-1", stage_id::NEURAL, 1_000_000, 50, Some(false)),
            syn("fail-2", stage_id::NEURAL, 1_000_000, 60, Some(false)),
        ];
        let report = aggregate(&results);
        assert!(report.joules_per_correct.is_infinite());
        assert!((report.accuracy_pct - 0.0).abs() < 1e-9);
    }

    #[test]
    fn latency_percentiles_are_nearest_rank() {
        // 100 cases with latency = 1..=100 ms; p50≈50, p95≈95, p99≈99.
        let results: Vec<BenchResult> = (1..=100)
            .map(|ms| syn(&format!("c-{ms}"), stage_id::CACHE, 0, ms, Some(true)))
            .collect();
        let report = aggregate(&results);
        assert_eq!(report.latency_p50_ms, 50);
        assert_eq!(report.latency_p95_ms, 95);
        assert_eq!(report.latency_p99_ms, 99);
    }

    #[test]
    fn empty_input_is_well_defined() {
        let report = aggregate(&[]);
        assert_eq!(report.case_count, 0);
        assert_eq!(report.total_joules, 0.0);
        assert!(report.joules_per_correct.is_infinite());
        assert_eq!(report.accuracy_pct, 0.0);
        assert_eq!(report.latency_p50_ms, 0);
        assert!(report.stage_distribution.is_empty());
    }

    #[test]
    fn matches_expected_is_case_insensitive_substring() {
        assert!(matches_expected("The answer is 4.", "4"));
        assert!(matches_expected("PARIS", "paris"));
        assert!(!matches_expected("the answer is five", "4"));
    }

    #[test]
    fn report_roundtrips_through_serde_json() {
        let results = vec![
            syn("a", stage_id::CACHE, 0, 1, Some(true)),
            syn("b", stage_id::NEURAL, 1_000, 2, Some(false)),
        ];
        let report = aggregate(&results);
        let json = serde_json::to_string(&report).unwrap();
        let back: BenchReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report, back);
    }
}
