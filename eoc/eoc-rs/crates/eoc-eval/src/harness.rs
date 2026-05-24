//! Common types for evaluation harnesses.
//!
//! A `Harness` is the canonical loader-plus-grader for one published
//! evaluation suite (MMLU, GPQA, HumanEval, ...). It hides the dataset
//! schema and the scoring rule behind a small trait surface so that the
//! `EvalRunner` can drive any harness through any `NeuralBackend` and
//! report `joules-per-correct` on a comparable axis.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Scoring rule the harness uses to grade a `Response` against an
/// `ExpectedAnswer`. Each variant is documented with the canonical
/// reference for the suite that uses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Multiple-choice accuracy (MMLU, MMLU-Pro, GPQA, ARC, HellaSwag,
    /// BoolQ, Winogrande, AGIEval, ...). Exact letter / option match
    /// after canonicalisation.
    Accuracy,
    /// String exact-match after light normalisation (GSM8K, MATH).
    ExactMatch,
    /// Code execution unit-test pass rate at k=1 (HumanEval).
    Pass1,
    /// ROUGE recall (summary tasks). Computed as ROUGE-L F1 against the
    /// reference text.
    Rouge,
    /// BLEU score (translation tasks).
    BleuScore,
    /// Harness-specific metric (label kept for reporting). Stored as a
    /// `&'static str` so the variant stays `Copy`; round-trips through
    /// serde as the plain string (deserialised values land in
    /// [`Metric::Custom`] via a small set of known labels).
    Custom(&'static str),
}

impl Metric {
    /// Stable string id for serialisation / CLI output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Metric::Accuracy => "accuracy",
            Metric::ExactMatch => "exact_match",
            Metric::Pass1 => "pass@1",
            Metric::Rouge => "rouge",
            Metric::BleuScore => "bleu",
            Metric::Custom(s) => s,
        }
    }

    /// Parse a stable string id back into a `Metric`. Unknown ids
    /// become [`Metric::Custom`] with one of the well-known suite
    /// labels (`mc1`, `win_rate`, `instruction_following`); anything
    /// else falls back to [`Metric::Custom("unknown")`].
    pub fn parse_id(s: &str) -> Self {
        match s {
            "accuracy" => Metric::Accuracy,
            "exact_match" => Metric::ExactMatch,
            "pass@1" => Metric::Pass1,
            "rouge" => Metric::Rouge,
            "bleu" => Metric::BleuScore,
            "mc1" => Metric::Custom("mc1"),
            "mc2" => Metric::Custom("mc2"),
            "win_rate" => Metric::Custom("win_rate"),
            "instruction_following" => Metric::Custom("instruction_following"),
            _ => Metric::Custom("unknown"),
        }
    }
}

impl serde::Serialize for Metric {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for Metric {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let s = <String as serde::Deserialize>::deserialize(d)?;
        Ok(Metric::parse_id(&s))
    }
}

/// Where a harness should load its cases from.
#[derive(Debug, Clone)]
pub enum DatasetSource {
    /// A local file already on disk. JSON or JSONL per the harness's
    /// `parse_local` implementation.
    Local(PathBuf),
    /// HuggingFace Hub dataset. Only resolvable when the crate is built
    /// with the `download` feature.
    HuggingFaceHub {
        /// `org/dataset` style repo identifier.
        repo: String,
        /// Split to fetch (`train`, `test`, `validation`, ...).
        split: String,
    },
    /// Small hand-curated sample embedded in the binary. Always available.
    BuiltinSample,
}

/// What a harness considers a "correct" answer for one case.
///
/// Different suites disagree on the shape of ground truth, so we keep a
/// small typed union instead of forcing every harness to stuff its data
/// into a single string.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExpectedAnswer {
    /// A single canonical letter (e.g. `"A"`) for an MCQ.
    Choice(String),
    /// A single canonical string the answer must equal (after
    /// normalisation).
    Text(String),
    /// A numeric answer (GSM8K's final integer; MATH's boxed value).
    Number(String),
    /// HumanEval-style: a unit-test program that must pass when
    /// concatenated with the candidate code.
    UnitTest {
        /// Function name the candidate must define.
        entry_point: String,
        /// Suite of tests to evaluate against the candidate.
        test_program: String,
    },
    /// AlpacaEval-style: a reference response the judge compares against.
    Reference(String),
    /// TruthfulQA: full label payload — single correct answer plus
    /// distractors used by MC1/MC2 scoring.
    Truthful {
        /// The correct option (canonical letter).
        correct: String,
        /// All other options' canonical letters.
        distractors: Vec<String>,
    },
    /// IFEval: list of constraint specs every output must satisfy.
    Constraints(Vec<IfEvalConstraint>),
}

/// One IFEval constraint. The verifier is dispatched by the variant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum IfEvalConstraint {
    /// Output must contain `keyword` at least `min_count` times
    /// (case-insensitive).
    KeywordAtLeast {
        /// Keyword to look for.
        keyword: String,
        /// Minimum number of occurrences.
        min_count: usize,
    },
    /// Output must contain exactly `count` paragraphs (separated by a
    /// blank line).
    ExactParagraphs(usize),
    /// Output must contain at least `min` and at most `max` words.
    WordCount {
        /// Inclusive lower bound.
        min: usize,
        /// Inclusive upper bound.
        max: usize,
    },
    /// Output must not contain a comma.
    NoComma,
    /// Output must be entirely upper-case (letters only).
    AllCaps,
    /// Output must be entirely lower-case (letters only).
    AllLowercase,
    /// Output must start with `prefix` (case-sensitive).
    StartsWith(String),
    /// Output must end with `suffix` (case-sensitive).
    EndsWith(String),
    /// Output must contain exactly `count` bullet points (lines starting
    /// with `- ` or `* `).
    ExactBullets(usize),
    /// Forbid the listed substrings (case-insensitive).
    ForbidKeywords(Vec<String>),
}

/// A single graded case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalCase {
    /// Stable identifier (e.g. `"mmlu/abstract_algebra/0"`).
    pub id: String,
    /// The prompt fed to the backend. Already includes any choices /
    /// formatting required by the suite.
    pub prompt: String,
    /// The labelled ground truth.
    pub expected: ExpectedAnswer,
    /// Provenance — `"mmlu"`, `"gpqa"`, etc. Useful when a `BenchReport`
    /// combines multiple suites.
    pub dataset: String,
    /// Optional subject / sub-task tag (`"high_school_physics"`,
    /// `"competition_math"`, ...).
    pub subject: Option<String>,
}

/// What a backend gave back for one case. Wraps an `eoc_core::Response`
/// so we keep joule attribution intact.
#[derive(Debug, Clone)]
pub struct Response {
    /// The raw payload text.
    pub payload: String,
    /// Wall-clock latency in milliseconds.
    pub latency_ms: u64,
    /// Joule cost reported by the backend.
    pub joule_cost: eoc_core::JouleCost,
}

impl Response {
    /// Lift an `eoc_core::Response` (plus measured latency) into the
    /// harness-friendly shape.
    pub fn from_core(r: eoc_core::Response, latency_ms: u64) -> Self {
        Self {
            payload: r.payload,
            latency_ms,
            joule_cost: r.joule_cost,
        }
    }
}

/// Helper: read the raw text for any [`DatasetSource`], using `builtin`
/// for [`DatasetSource::BuiltinSample`] and erroring with a stable
/// message when the `download` feature is required but absent.
pub async fn load_raw(source: DatasetSource, builtin: &'static str) -> Result<String> {
    match source {
        DatasetSource::BuiltinSample => Ok(builtin.to_string()),
        DatasetSource::Local(path) => tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| crate::error::EvalError::Io { path, source: e }),
        DatasetSource::HuggingFaceHub { repo, split } => {
            #[cfg(feature = "download")]
            {
                crate::dataset_loader::fetch_huggingface(&repo, &split).await
            }
            #[cfg(not(feature = "download"))]
            {
                let _ = (repo, split);
                Err(crate::error::EvalError::FeatureDisabled { feature: "download" })
            }
        }
    }
}

/// One canonical evaluation suite.
#[async_trait]
pub trait Harness: Send + Sync {
    /// Stable lowercase id for this suite (e.g. `"mmlu"`).
    fn name(&self) -> &'static str;

    /// Default metric this suite reports.
    fn metric(&self) -> Metric;

    /// Load the cases from `source`. Builtin samples must always work
    /// without any external dependency.
    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>>;

    /// Grade one response. Returns a value in `[0.0, 1.0]` so suites
    /// with graded credit (TruthfulQA MC2, IFEval per-constraint) can
    /// report fractional scores. Hard 0/1 suites simply return 0.0 or
    /// 1.0.
    async fn score(&self, response: &Response, expected: &ExpectedAnswer) -> f64;
}
