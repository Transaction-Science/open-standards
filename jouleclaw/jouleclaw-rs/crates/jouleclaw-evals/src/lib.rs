//! # jouleclaw-evals
//!
//! Eval-driven skill engineering — the four-mode pipeline Anthropic
//! pinned in skill-creator 2.0 (Create / Eval / Improve / Benchmark)
//! with executor / grader / comparator sub-roles. The crate ships
//! the *traits* and *reference graders*; the Executor is consumer-
//! supplied because what is being eval'd varies (a model, a skill,
//! a tier, a whole cascade).
//!
//! ## Energy as the orthogonal pass criterion
//!
//! The classic eval trap is "100/100 means nothing." If your eval
//! set is too easy, even a bad executor passes. This crate
//! addresses that by treating **energy spend as a first-class
//! grader axis** — an executor that gets the answer right but
//! costs 10× more joules has *regressed*. The
//! [`EnergyBudgetGrader`] makes that explicit: pass iff
//! `(correct AND joules_uj <= budget_uj)`. The [`Comparator`]
//! distinguishes correctness regressions from energy regressions
//! so you can see them separately.
//!
//! ## Honest scope (v1)
//!
//! - Measures the EvalCase set, not the world. Cases must
//!   discriminate; the field has admitted that easy eval sets
//!   teach nothing.
//! - No statistical significance testing, no cross-fold
//!   validation, no automatic case discovery.
//! - Grader trait returns `bool` — pass/fail. Three-valued
//!   "pass/warn/fail" is the consumer's composition (run two
//!   graders).
//! - Executor returns `EvalRun` synchronously; async backends
//!   bridge through their own runtime.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(unexpected_cfgs)]

use jouleclaw_critic::{Artifact, CritiqueReport, DeterministicCritic, Critic, Rubric, Verdict};
use jouleclaw_energy::Provenance;
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────

/// One eval case the executor is asked to handle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalCase {
    /// Stable id for the case. Used for regression / improvement
    /// flip detection in [`ComparisonReport`].
    pub id: String,
    /// Input to the executor. Shape is consumer-defined.
    pub input: serde_json::Value,
    /// Optional expected output for exact / subset graders.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<serde_json::Value>,
}

/// One executor run on one case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalRun {
    /// `EvalCase::id` this run is for.
    pub case_id: String,
    /// What the executor produced.
    pub actual: serde_json::Value,
    /// Whether the grader passed it.
    pub passed: bool,
    /// Microjoules the executor reports having spent.
    pub joules_uj: u64,
    /// Honesty tier of `joules_uj`.
    pub energy_provenance: Provenance,
    /// Optional reason — typically a one-line grader/executor note.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Aggregate report over an eval pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    /// Per-case runs in input order.
    pub runs: Vec<EvalRun>,
    /// `passed / total` as a fraction in `[0, 1]`. NaN if no runs.
    pub pass_rate: f64,
    /// Mean microjoules across runs. 0 if no runs.
    pub mean_joules_uj: u64,
    /// Worst observed provenance across runs — the floor of the
    /// honesty ladder, same rule as `jouleclaw-prov::Receipt`.
    pub energy_provenance: Provenance,
}

impl EvalReport {
    /// Build a report from a run set, computing the rollups.
    pub fn from_runs(runs: Vec<EvalRun>) -> Self {
        let n = runs.len();
        let pass_rate = if n == 0 {
            f64::NAN
        } else {
            runs.iter().filter(|r| r.passed).count() as f64 / n as f64
        };
        let mean_joules_uj = if n == 0 {
            0
        } else {
            runs.iter().map(|r| r.joules_uj).sum::<u64>() / n as u64
        };
        let energy_provenance = runs
            .iter()
            .map(|r| r.energy_provenance)
            .fold(Provenance::HwShunt, worst_provenance);
        Self {
            runs,
            pass_rate,
            mean_joules_uj,
            energy_provenance,
        }
    }

    /// How many runs were observed.
    pub fn len(&self) -> usize {
        self.runs.len()
    }
    /// `true` if no runs were observed.
    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }
}

fn worst_provenance(a: Provenance, b: Provenance) -> Provenance {
    use Provenance::*;
    let rank = |p: Provenance| match p {
        HwShunt => 2,
        ModelBased => 1,
        Estimator => 0,
    };
    if rank(a) <= rank(b) {
        a
    } else {
        b
    }
}

// ─────────────────────────────────────────────────────────────────────
// Trait surface
// ─────────────────────────────────────────────────────────────────────

/// Runs an [`EvalCase`] and emits a [`EvalRun`]. Consumer-supplied
/// because what is being eval'd varies.
pub trait Executor: Send + Sync {
    /// Execute one case.
    fn execute(&self, case: &EvalCase) -> EvalRun;
}

/// Grades an `actual` value against an [`EvalCase`]. Pure pass/fail;
/// compose two graders for "warn" / multi-axis.
pub trait Grader: Send + Sync {
    /// Return `true` iff the run passes.
    fn grade(
        &self,
        case: &EvalCase,
        actual: &serde_json::Value,
        run: &EvalRun,
    ) -> bool;
}

/// Compares two [`EvalReport`]s — typically a baseline (V_n) and a
/// candidate (V_n+1).
pub trait Comparator: Send + Sync {
    /// Produce a structured report.
    fn compare(
        &self,
        baseline: &EvalReport,
        candidate: &EvalReport,
    ) -> ComparisonReport;
}

/// Result of comparing two reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// `candidate.pass_rate - baseline.pass_rate`.
    pub pass_rate_delta: f64,
    /// `(candidate - baseline) / baseline * 100.0`. NaN if baseline
    /// mean is 0.
    pub joules_delta_pct: f64,
    /// Cases that flipped pass → fail. Correctness regressions.
    pub regressions: Vec<String>,
    /// Cases that flipped fail → pass.
    pub improvements: Vec<String>,
    /// Cases that still pass but now cost more joules than baseline
    /// (above [`ComparisonReport::ENERGY_REGRESSION_MIN_RATIO`]).
    /// The "100/100 with 10× energy" trap.
    pub energy_regressions: Vec<EnergyRegression>,
}

impl ComparisonReport {
    /// A run that costs at least this much more energy than its
    /// baseline counterpart counts as an energy regression. 1.10 =
    /// 10% over.
    pub const ENERGY_REGRESSION_MIN_RATIO: f64 = 1.10;
}

/// One energy regression — a case that passed in both reports but
/// cost more in the candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyRegression {
    /// `EvalCase::id`.
    pub case_id: String,
    /// Baseline run joules.
    pub baseline_uj: u64,
    /// Candidate run joules.
    pub candidate_uj: u64,
    /// `candidate / baseline`. Always `>= 1.0` per filter.
    pub ratio: f64,
}

// ─────────────────────────────────────────────────────────────────────
// Reference impls
// ─────────────────────────────────────────────────────────────────────

/// Grader: exact equality between `actual` and `EvalCase::expected`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ExactMatchGrader;

impl Grader for ExactMatchGrader {
    fn grade(
        &self,
        case: &EvalCase,
        actual: &serde_json::Value,
        _run: &EvalRun,
    ) -> bool {
        match &case.expected {
            Some(e) => actual == e,
            None => false,
        }
    }
}

/// Grader: `actual` is an object containing every key in
/// `case.expected` with equal values (a structural subset). Useful
/// when the executor emits more fields than the test pins.
#[derive(Debug, Default, Clone, Copy)]
pub struct JsonSubsetGrader;

impl Grader for JsonSubsetGrader {
    fn grade(
        &self,
        case: &EvalCase,
        actual: &serde_json::Value,
        _run: &EvalRun,
    ) -> bool {
        let Some(expected) = &case.expected else {
            return false;
        };
        json_contains(actual, expected)
    }
}

fn json_contains(haystack: &serde_json::Value, needle: &serde_json::Value) -> bool {
    use serde_json::Value::*;
    match (haystack, needle) {
        (Object(h), Object(n)) => n
            .iter()
            .all(|(k, v)| h.get(k).map_or(false, |hv| json_contains(hv, v))),
        (a, b) => a == b,
    }
}

/// Grader: passes iff `run.joules_uj <= budget_uj`. Composes with
/// correctness graders via [`AndGrader`].
#[derive(Debug, Clone, Copy)]
pub struct EnergyBudgetGrader {
    /// Maximum allowed microjoules.
    pub budget_uj: u64,
}

impl EnergyBudgetGrader {
    /// New grader with `budget_uj` ceiling.
    pub fn new(budget_uj: u64) -> Self {
        Self { budget_uj }
    }
}

impl Grader for EnergyBudgetGrader {
    fn grade(
        &self,
        _case: &EvalCase,
        _actual: &serde_json::Value,
        run: &EvalRun,
    ) -> bool {
        run.joules_uj <= self.budget_uj
    }
}

/// Grader: passes iff every child grader passes. The composition
/// rule for multi-axis grading (e.g. correctness AND energy).
pub struct AndGrader {
    graders: Vec<Box<dyn Grader>>,
}

impl AndGrader {
    /// Empty AND grader.
    pub fn new() -> Self {
        Self { graders: Vec::new() }
    }
    /// Append a child grader.
    pub fn with<G: Grader + 'static>(mut self, g: G) -> Self {
        self.graders.push(Box::new(g));
        self
    }
    /// Number of registered graders.
    pub fn len(&self) -> usize {
        self.graders.len()
    }
    /// Whether no graders are registered.
    pub fn is_empty(&self) -> bool {
        self.graders.is_empty()
    }
}

impl Default for AndGrader {
    fn default() -> Self {
        Self::new()
    }
}

impl Grader for AndGrader {
    fn grade(
        &self,
        case: &EvalCase,
        actual: &serde_json::Value,
        run: &EvalRun,
    ) -> bool {
        self.graders.iter().all(|g| g.grade(case, actual, run))
    }
}

/// Grader: delegate to a `jouleclaw-critic` [`Critic`] over the
/// run's stringified `actual` and a fixed [`Rubric`]. The
/// `DeterministicCritic` reference lives in jouleclaw-critic; pass
/// `LlmCritic` for LLM-graded eval rubrics.
pub struct RubricGrader<C: Critic> {
    critic: C,
    rubric: Rubric,
}

impl<C: Critic> RubricGrader<C> {
    /// Build a rubric grader from a critic + rubric.
    pub fn new(critic: C, rubric: Rubric) -> Self {
        Self { critic, rubric }
    }
}

impl<C: Critic> Grader for RubricGrader<C> {
    fn grade(
        &self,
        _case: &EvalCase,
        actual: &serde_json::Value,
        _run: &EvalRun,
    ) -> bool {
        let bytes = serde_json::to_vec(actual).unwrap_or_default();
        let artifact = Artifact::json(bytes);
        let report: CritiqueReport = self.critic.critique(&artifact, &self.rubric);
        // Any failing finding blocks; warn-only does not block.
        matches!(report.overall, Verdict::Pass | Verdict::Warn)
    }
}

/// Convenience: a `RubricGrader` over the `DeterministicCritic`.
pub fn deterministic_rubric_grader(rubric: Rubric) -> RubricGrader<DeterministicCritic> {
    RubricGrader::new(DeterministicCritic, rubric)
}

/// Reference comparator — produces a full [`ComparisonReport`].
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultComparator;

impl Comparator for DefaultComparator {
    fn compare(
        &self,
        baseline: &EvalReport,
        candidate: &EvalReport,
    ) -> ComparisonReport {
        let pass_rate_delta = candidate.pass_rate - baseline.pass_rate;
        let joules_delta_pct = if baseline.mean_joules_uj == 0 {
            f64::NAN
        } else {
            (candidate.mean_joules_uj as f64 - baseline.mean_joules_uj as f64)
                / baseline.mean_joules_uj as f64
                * 100.0
        };

        // Build case-id → run lookups.
        use std::collections::BTreeMap;
        let by_id_b: BTreeMap<&str, &EvalRun> =
            baseline.runs.iter().map(|r| (r.case_id.as_str(), r)).collect();
        let by_id_c: BTreeMap<&str, &EvalRun> =
            candidate.runs.iter().map(|r| (r.case_id.as_str(), r)).collect();

        let mut regressions = Vec::new();
        let mut improvements = Vec::new();
        let mut energy_regressions = Vec::new();

        for (id, c) in &by_id_c {
            let Some(b) = by_id_b.get(id) else { continue };
            // Correctness flips.
            match (b.passed, c.passed) {
                (true, false) => regressions.push(id.to_string()),
                (false, true) => improvements.push(id.to_string()),
                _ => {}
            }
            // Energy regression: both passed AND candidate costs more
            // by at least ENERGY_REGRESSION_MIN_RATIO.
            if b.passed && c.passed && b.joules_uj > 0 {
                let ratio = c.joules_uj as f64 / b.joules_uj as f64;
                if ratio >= ComparisonReport::ENERGY_REGRESSION_MIN_RATIO {
                    energy_regressions.push(EnergyRegression {
                        case_id: id.to_string(),
                        baseline_uj: b.joules_uj,
                        candidate_uj: c.joules_uj,
                        ratio,
                    });
                }
            }
        }
        regressions.sort();
        improvements.sort();
        energy_regressions.sort_by(|a, b| a.case_id.cmp(&b.case_id));
        ComparisonReport {
            pass_rate_delta,
            joules_delta_pct,
            regressions,
            improvements,
            energy_regressions,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Pipeline runner
// ─────────────────────────────────────────────────────────────────────

/// Run every case in `cases` through `executor`, then grade each
/// using `grader`, then assemble an [`EvalReport`].
pub fn run_eval<E: Executor + ?Sized, G: Grader + ?Sized>(
    executor: &E,
    grader: &G,
    cases: &[EvalCase],
) -> EvalReport {
    let mut runs = Vec::with_capacity(cases.len());
    for case in cases {
        let mut run = executor.execute(case);
        // Grader can OVERRIDE the executor's reported pass — the
        // grader is authoritative, not the executor (the executor
        // might lie / hallucinate).
        let pass = grader.grade(case, &run.actual, &run);
        run.passed = pass;
        runs.push(run);
    }
    EvalReport::from_runs(runs)
}

// ─────────────────────────────────────────────────────────────────────
// Kani proofs
// ─────────────────────────────────────────────────────────────────────

/// Pass rate is in `[0, 1]` for any non-empty run set.
#[cfg(kani)]
#[kani::proof]
fn kani_pass_rate_in_range() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n < 8);
    let mut runs = Vec::new();
    for i in 0..n {
        let passed: bool = kani::any();
        runs.push(EvalRun {
            case_id: format!("c{i}"),
            actual: serde_json::Value::Null,
            passed,
            joules_uj: 0,
            energy_provenance: Provenance::Estimator,
            reason: None,
        });
    }
    let rep = EvalReport::from_runs(runs);
    kani::assert(rep.pass_rate >= 0.0 && rep.pass_rate <= 1.0, "pass rate in [0,1]");
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn case(id: &str, expected: serde_json::Value) -> EvalCase {
        EvalCase {
            id: id.into(),
            input: serde_json::json!({"q": id}),
            expected: Some(expected),
        }
    }

    fn run_at(id: &str, actual: serde_json::Value, joules: u64, passed: bool) -> EvalRun {
        EvalRun {
            case_id: id.into(),
            actual,
            passed,
            joules_uj: joules,
            energy_provenance: Provenance::HwShunt,
            reason: None,
        }
    }

    struct EchoExecutor;
    impl Executor for EchoExecutor {
        fn execute(&self, case: &EvalCase) -> EvalRun {
            EvalRun {
                case_id: case.id.clone(),
                actual: case.expected.clone().unwrap_or(serde_json::Value::Null),
                passed: false, // grader decides
                joules_uj: 100,
                energy_provenance: Provenance::HwShunt,
                reason: None,
            }
        }
    }

    struct BadExecutor;
    impl Executor for BadExecutor {
        fn execute(&self, case: &EvalCase) -> EvalRun {
            EvalRun {
                case_id: case.id.clone(),
                actual: serde_json::json!({"bad": true}),
                passed: false,
                joules_uj: 10_000,
                energy_provenance: Provenance::Estimator,
                reason: None,
            }
        }
    }

    #[test]
    fn report_rollups_pass_rate_and_mean_joules() {
        let runs = vec![
            run_at("a", serde_json::json!({}), 100, true),
            run_at("b", serde_json::json!({}), 200, false),
            run_at("c", serde_json::json!({}), 300, true),
        ];
        let r = EvalReport::from_runs(runs);
        assert_eq!(r.pass_rate, 2.0 / 3.0);
        assert_eq!(r.mean_joules_uj, 200);
        assert_eq!(r.energy_provenance, Provenance::HwShunt);
    }

    #[test]
    fn exact_match_grader_compares_to_expected() {
        let g = ExactMatchGrader;
        let c = case("a", serde_json::json!({"x": 1}));
        let r = run_at("a", serde_json::json!({"x": 1}), 0, false);
        assert!(g.grade(&c, &r.actual, &r));
        let r2 = run_at("a", serde_json::json!({"x": 2}), 0, false);
        assert!(!g.grade(&c, &r2.actual, &r2));
    }

    #[test]
    fn json_subset_grader_allows_extra_keys_in_actual() {
        let g = JsonSubsetGrader;
        let c = case("a", serde_json::json!({"x": 1}));
        let r = run_at("a", serde_json::json!({"x": 1, "y": 2}), 0, false);
        assert!(g.grade(&c, &r.actual, &r));
    }

    #[test]
    fn energy_budget_grader_passes_under_budget() {
        let g = EnergyBudgetGrader::new(150);
        let c = case("a", serde_json::json!({}));
        let r_pass = run_at("a", serde_json::json!({}), 100, true);
        let r_fail = run_at("a", serde_json::json!({}), 200, true);
        assert!(g.grade(&c, &r_pass.actual, &r_pass));
        assert!(!g.grade(&c, &r_fail.actual, &r_fail));
    }

    #[test]
    fn and_grader_requires_all_children() {
        let c = case("a", serde_json::json!({"x": 1}));
        let g = AndGrader::new()
            .with(ExactMatchGrader)
            .with(EnergyBudgetGrader::new(150));
        let r_pass = run_at("a", serde_json::json!({"x": 1}), 100, false);
        let r_correct_but_expensive = run_at("a", serde_json::json!({"x": 1}), 9999, false);
        let r_cheap_but_wrong = run_at("a", serde_json::json!({"x": 2}), 50, false);
        assert!(g.grade(&c, &r_pass.actual, &r_pass));
        assert!(!g.grade(&c, &r_correct_but_expensive.actual, &r_correct_but_expensive));
        assert!(!g.grade(&c, &r_cheap_but_wrong.actual, &r_cheap_but_wrong));
    }

    #[test]
    fn run_eval_lets_grader_override_executor_reported_pass() {
        let cases = vec![case("a", serde_json::json!({"x": 1}))];
        let report = run_eval(&EchoExecutor, &ExactMatchGrader, &cases);
        assert_eq!(report.len(), 1);
        assert!(report.runs[0].passed, "grader overrode executor's false");
    }

    #[test]
    fn comparator_detects_correctness_flips() {
        let baseline = EvalReport::from_runs(vec![
            run_at("a", serde_json::json!({}), 100, true),
            run_at("b", serde_json::json!({}), 100, true),
            run_at("c", serde_json::json!({}), 100, false),
        ]);
        let candidate = EvalReport::from_runs(vec![
            run_at("a", serde_json::json!({}), 100, true),
            run_at("b", serde_json::json!({}), 100, false), // regression
            run_at("c", serde_json::json!({}), 100, true),  // improvement
        ]);
        let r = DefaultComparator.compare(&baseline, &candidate);
        assert_eq!(r.regressions, vec!["b".to_string()]);
        assert_eq!(r.improvements, vec!["c".to_string()]);
    }

    #[test]
    fn comparator_detects_energy_regression_even_when_both_pass() {
        let baseline = EvalReport::from_runs(vec![
            run_at("a", serde_json::json!({}), 100, true),
            run_at("b", serde_json::json!({}), 100, true),
        ]);
        let candidate = EvalReport::from_runs(vec![
            run_at("a", serde_json::json!({}), 100, true), // no change
            run_at("b", serde_json::json!({}), 500, true), // 5x energy
        ]);
        let r = DefaultComparator.compare(&baseline, &candidate);
        assert!(r.regressions.is_empty());
        assert_eq!(r.energy_regressions.len(), 1);
        assert_eq!(r.energy_regressions[0].case_id, "b");
        assert_eq!(r.energy_regressions[0].ratio, 5.0);
    }

    #[test]
    fn comparator_no_false_energy_regression_below_min_ratio() {
        // Just 5% above baseline — below the 10% min ratio.
        let baseline = EvalReport::from_runs(vec![run_at("a", serde_json::json!({}), 1000, true)]);
        let candidate = EvalReport::from_runs(vec![run_at("a", serde_json::json!({}), 1050, true)]);
        let r = DefaultComparator.compare(&baseline, &candidate);
        assert!(r.energy_regressions.is_empty(), "5% over should not flag");
    }

    #[test]
    fn comparator_reports_joules_delta_percent() {
        let baseline = EvalReport::from_runs(vec![run_at("a", serde_json::json!({}), 100, true)]);
        let candidate = EvalReport::from_runs(vec![run_at("a", serde_json::json!({}), 200, true)]);
        let r = DefaultComparator.compare(&baseline, &candidate);
        assert_eq!(r.joules_delta_pct, 100.0);
    }

    #[test]
    fn worst_provenance_picks_estimator_over_hwshunt() {
        let runs = vec![
            run_at("a", serde_json::json!({}), 100, true),
            EvalRun {
                case_id: "b".into(),
                actual: serde_json::json!({}),
                passed: true,
                joules_uj: 100,
                energy_provenance: Provenance::Estimator,
                reason: None,
            },
        ];
        let r = EvalReport::from_runs(runs);
        assert_eq!(r.energy_provenance, Provenance::Estimator);
    }

    #[test]
    fn eval_run_round_trips_through_json() {
        let r = run_at("a", serde_json::json!({"x": 1}), 100, true);
        let bytes = serde_json::to_vec(&r).unwrap();
        let back: EvalRun = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn bad_executor_with_strict_grader_flags_failure() {
        let cases = vec![case("a", serde_json::json!({"good": true}))];
        let report = run_eval(&BadExecutor, &ExactMatchGrader, &cases);
        assert!(!report.runs[0].passed);
        assert_eq!(report.pass_rate, 0.0);
    }
}
