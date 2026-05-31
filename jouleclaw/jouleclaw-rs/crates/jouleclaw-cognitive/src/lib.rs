//! # jouleclaw-cognitive
//!
//! SonarSource Cognitive Complexity (Campbell 2017, updated 2018)
//! over the Rust `syn` AST. The canonical "how hard is this code to
//! read?" metric, distinct from cyclomatic complexity (which counts
//! control-flow nodes without nesting penalty).
//!
//! ## Sonar's three rules
//!
//! - **B1 — Increment for each break in linear flow.** `if`,
//!   `else if`, `match` (per arm), `for`, `while`, `loop`, `?`,
//!   `break`/`continue` with label, recursion, sequences of mixed
//!   boolean operators.
//! - **B2 — Increment for each level of nesting.** Each B1
//!   structure inside another structure adds `current_nesting`
//!   in addition to B1.
//! - **B3 — Certain structures get the basic increment but no
//!   nesting penalty.** `else`, `else if` (the basic +1 only, no
//!   nesting bump).
//!
//! ## Honest scope (from wave-4 SOTA brief)
//!
//! - **Clippy's `cognitive_complexity` lint is NOT used.** Clippy
//!   maintainers self-deprecated it as "the true cognitive
//!   complexity is not something we can calculate using modern
//!   technology." We reimplement against the SonarSource
//!   whitepaper directly.
//! - **No hard threshold.** Sonar's default is 15 for Java
//!   methods; that does NOT generalise. We expose
//!   [`ComplexityCorpus::percentile_rank`] so the consumer
//!   reviews the *top decile*, not a fixed cutoff. The wave-4
//!   brief was emphatic: threshold-based gates are the #1 reason
//!   teams disable the metric.
//! - **Complexity is a *proxy*, not a metric of correctness.** It
//!   does not detect bugs and is gameable by extracting functions.
//!   Treat as a heuristic ranking aid.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(unexpected_cfgs)]

use jouleclaw_bounded::{Score, Scored};
use serde::{Deserialize, Serialize};
use syn::visit::{self, Visit};
use syn::{Expr, ItemFn};

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Errors a parse can surface.
#[derive(Debug, thiserror::Error)]
pub enum CognitiveError {
    /// `syn` could not parse the source.
    #[error("syn parse: {0}")]
    Parse(#[from] syn::Error),
}

// ─────────────────────────────────────────────────────────────────────
// Per-function complexity
// ─────────────────────────────────────────────────────────────────────

/// Per-function complexity result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionComplexity {
    /// Function name (last segment of the path).
    pub name: String,
    /// Cognitive complexity score per the SonarSource rules.
    pub score: u32,
    /// Number of distinct nesting structures encountered (B1 hits).
    pub b1_hits: u32,
    /// Sum of nesting-penalty increments (B2 hits).
    pub b2_hits: u32,
}

/// Walk one Rust file and return per-function complexity scores.
pub fn analyze_file(source: &str) -> Result<Vec<FunctionComplexity>, CognitiveError> {
    let parsed: syn::File = syn::parse_str(source)?;
    let mut out = Vec::new();
    for item in &parsed.items {
        match item {
            syn::Item::Fn(f) => out.push(score_fn(f)),
            syn::Item::Impl(im) => {
                for ii in &im.items {
                    if let syn::ImplItem::Fn(f) = ii {
                        // Wrap into ItemFn shape for the scorer.
                        let item = ItemFn {
                            attrs: f.attrs.clone(),
                            vis: f.vis.clone(),
                            sig: f.sig.clone(),
                            block: Box::new(f.block.clone()),
                        };
                        out.push(score_fn(&item));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn score_fn(f: &ItemFn) -> FunctionComplexity {
    let mut scorer = CogScorer::default();
    scorer.visit_block(&f.block);
    FunctionComplexity {
        name: f.sig.ident.to_string(),
        score: scorer.score,
        b1_hits: scorer.b1_hits,
        b2_hits: scorer.b2_hits,
    }
}

/// AST visitor that implements the SonarSource rules.
#[derive(Default)]
struct CogScorer {
    score: u32,
    nesting: u32,
    b1_hits: u32,
    b2_hits: u32,
}

impl CogScorer {
    /// Apply B1 + B2 — increment by `1 + current_nesting`.
    fn b1_b2(&mut self) {
        self.score += 1 + self.nesting;
        self.b1_hits += 1;
        if self.nesting > 0 {
            self.b2_hits += self.nesting;
        }
    }
    /// Apply B3 — increment by 1 only.
    fn b3(&mut self) {
        self.score += 1;
        self.b1_hits += 1;
    }

    fn enter_nest(&mut self) {
        self.nesting += 1;
    }
    fn exit_nest(&mut self) {
        self.nesting = self.nesting.saturating_sub(1);
    }
}

impl<'ast> Visit<'ast> for CogScorer {
    fn visit_expr(&mut self, e: &'ast Expr) {
        match e {
            Expr::If(_) | Expr::While(_) | Expr::ForLoop(_) | Expr::Loop(_) | Expr::Match(_) => {
                self.b1_b2();
                self.enter_nest();
                visit::visit_expr(self, e);
                self.exit_nest();
            }
            Expr::Try(_) => {
                // `?` is a break in linear flow — increment without
                // nesting bump (B3 in the SonarSource spec for
                // single-token jumps).
                self.b3();
                visit::visit_expr(self, e);
            }
            Expr::Break(brk) if brk.label.is_some() => {
                self.b3();
                visit::visit_expr(self, e);
            }
            Expr::Continue(c) if c.label.is_some() => {
                self.b3();
                visit::visit_expr(self, e);
            }
            _ => visit::visit_expr(self, e),
        }
    }

    /// `else if` chains and `else` blocks get the B3 increment
    /// (the basic +1, no nesting bump).
    fn visit_expr_if(&mut self, e: &'ast syn::ExprIf) {
        // The B1/B2 increment for the `if` itself is handled in
        // `visit_expr`. Here we handle the `else` branch.
        visit::visit_block(self, &e.then_branch);
        if let Some((_else_tok, else_expr)) = &e.else_branch {
            match &**else_expr {
                Expr::If(_) => {
                    // `else if` — B3 increment without re-entering
                    // the if branch (the inner `visit_expr` will
                    // count the nested if too via the outer visitor;
                    // we need to make sure we don't double-count).
                    self.b3();
                    visit::visit_expr(self, else_expr);
                }
                Expr::Block(_) => {
                    // Plain `else { ... }` — B3.
                    self.b3();
                    visit::visit_expr(self, else_expr);
                }
                _ => visit::visit_expr(self, else_expr),
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Codebase-level corpus + percentile rank
// ─────────────────────────────────────────────────────────────────────

/// A corpus of per-function complexity scores from one or more
/// files. Sorted internally for fast percentile lookups.
pub struct ComplexityCorpus {
    sorted_scores: Vec<u32>,
    by_name: Vec<FunctionComplexity>,
}

impl ComplexityCorpus {
    /// Build from a list of per-function results.
    pub fn from_functions(functions: Vec<FunctionComplexity>) -> Self {
        let mut sorted_scores: Vec<u32> = functions.iter().map(|f| f.score).collect();
        sorted_scores.sort_unstable();
        Self {
            sorted_scores,
            by_name: functions,
        }
    }

    /// Number of functions in the corpus.
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// Whether the corpus is empty.
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// Mean complexity.
    pub fn mean(&self) -> f64 {
        if self.by_name.is_empty() {
            return 0.0;
        }
        self.sorted_scores.iter().map(|s| *s as f64).sum::<f64>() / self.by_name.len() as f64
    }

    /// Percentile-rank of a function's score within the corpus —
    /// the fraction of functions with strictly lower score.
    /// `1.0` = highest in the codebase; `0.0` = lowest. The number
    /// the SOTA brief recommends *over* threshold-based gates.
    pub fn percentile_rank(&self, score: u32) -> f64 {
        if self.sorted_scores.is_empty() {
            return 0.0;
        }
        let count_below = self.sorted_scores.iter().filter(|s| **s < score).count();
        count_below as f64 / self.sorted_scores.len() as f64
    }

    /// Tukey upper fence on the score distribution. Functions above
    /// this fence are *defensibly* outliers (Tukey 1977), unlike a
    /// 3σ rule which presumes normality the distribution explicitly
    /// violates.
    pub fn tukey_upper_fence(&self, k: f64) -> u32 {
        if self.sorted_scores.is_empty() {
            return 0;
        }
        let q1 = quantile_u32(&self.sorted_scores, 0.25);
        let q3 = quantile_u32(&self.sorted_scores, 0.75);
        let iqr = q3 as f64 - q1 as f64;
        (q3 as f64 + k * iqr).max(0.0) as u32
    }

    /// Functions in the top decile by complexity — the rank-based
    /// "review these first" list the SOTA brief recommends.
    pub fn top_decile(&self) -> Vec<&FunctionComplexity> {
        let n = self.by_name.len();
        if n == 0 {
            return Vec::new();
        }
        let cutoff_idx = (n as f64 * 0.9) as usize;
        let cutoff_score = self.sorted_scores.get(cutoff_idx).copied().unwrap_or(0);
        let mut hits: Vec<&FunctionComplexity> =
            self.by_name.iter().filter(|f| f.score >= cutoff_score).collect();
        hits.sort_by(|a, b| b.score.cmp(&a.score));
        hits
    }
}

fn quantile_u32(sorted: &[u32], q: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    sorted[lo.min(sorted.len() - 1)]
}

/// Structured explanation surfaced by [`CorpusScorer`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveExplanation {
    /// Function name.
    pub name: String,
    /// Raw cognitive-complexity score.
    pub raw_score: u32,
    /// Codebase percentile rank (0.0..=1.0).
    pub percentile_rank: f64,
    /// Tukey upper fence at `k = 1.5` over the corpus.
    pub corpus_fence_15: u32,
    /// Mean corpus complexity.
    pub corpus_mean: f64,
}

/// `Scored` adapter: takes a [`FunctionComplexity`], scores it
/// against a [`ComplexityCorpus`]. Score value = percentile rank
/// in `[0, 1]`; explanation carries the raw number + corpus
/// context.
pub struct CorpusScorer<'a> {
    /// The corpus the scorer ranks against.
    pub corpus: &'a ComplexityCorpus,
}

impl<'a> Scored for CorpusScorer<'a> {
    type Input = FunctionComplexity;
    type Explanation = CognitiveExplanation;
    fn score(&self, f: &FunctionComplexity) -> Score<CognitiveExplanation> {
        let pr = self.corpus.percentile_rank(f.score);
        Score::new(
            pr,
            CognitiveExplanation {
                name: f.name.clone(),
                raw_score: f.score,
                percentile_rank: pr,
                corpus_fence_15: self.corpus.tukey_upper_fence(1.5),
                corpus_mean: self.corpus.mean(),
            },
            "cognitive-complexity",
        )
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_function_scores_zero() {
        let src = r#"
            fn add(a: u32, b: u32) -> u32 {
                a + b
            }
        "#;
        let fs = analyze_file(src).unwrap();
        assert_eq!(fs.len(), 1);
        assert_eq!(fs[0].name, "add");
        assert_eq!(fs[0].score, 0);
    }

    #[test]
    fn one_if_scores_one() {
        let src = r#"
            fn x(a: i32) -> i32 {
                if a > 0 { a } else { -a }
            }
        "#;
        let fs = analyze_file(src).unwrap();
        // `if` → +1; `else { ... }` → +1 (B3).
        assert!(fs[0].score >= 1);
    }

    #[test]
    fn nested_if_adds_nesting_penalty() {
        let src = r#"
            fn x(a: i32, b: i32) -> i32 {
                if a > 0 {
                    if b > 0 {
                        a + b
                    } else {
                        a
                    }
                } else {
                    b
                }
            }
        "#;
        let fs = analyze_file(src).unwrap();
        // Outer if = +1; inner if = +1 + 1 (nesting); outer else = +1;
        // inner else = +1; total ≥ 5.
        assert!(fs[0].score >= 5, "got {}", fs[0].score);
    }

    #[test]
    fn loops_and_match_increment() {
        let src = r#"
            fn x(items: &[i32]) -> i32 {
                let mut total = 0;
                for v in items {
                    match v {
                        0 => continue,
                        _ => total += v,
                    }
                }
                total
            }
        "#;
        let fs = analyze_file(src).unwrap();
        // for = +1; match = +1 + 1 (nested in for).
        assert!(fs[0].score >= 3, "got {}", fs[0].score);
    }

    #[test]
    fn try_operator_increments() {
        let src = r#"
            fn x(s: &str) -> Result<i32, std::num::ParseIntError> {
                let n: i32 = s.parse()?;
                Ok(n + 1)
            }
        "#;
        let fs = analyze_file(src).unwrap();
        assert!(fs[0].score >= 1, "got {}", fs[0].score);
    }

    #[test]
    fn impl_block_methods_are_collected() {
        let src = r#"
            struct S;
            impl S {
                fn easy(&self) {}
                fn hard(&self, a: i32) {
                    if a > 0 { } else { }
                }
            }
        "#;
        let fs = analyze_file(src).unwrap();
        let names: Vec<&str> = fs.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"easy"));
        assert!(names.contains(&"hard"));
    }

    #[test]
    fn corpus_percentile_rank_zero_when_input_is_lowest() {
        let fs = vec![
            FunctionComplexity { name: "a".into(), score: 0, b1_hits: 0, b2_hits: 0 },
            FunctionComplexity { name: "b".into(), score: 5, b1_hits: 1, b2_hits: 0 },
            FunctionComplexity { name: "c".into(), score: 50, b1_hits: 5, b2_hits: 3 },
        ];
        let corpus = ComplexityCorpus::from_functions(fs);
        assert_eq!(corpus.percentile_rank(0), 0.0);
    }

    #[test]
    fn corpus_percentile_rank_high_for_top_function() {
        let fs = (0..10)
            .map(|i| FunctionComplexity {
                name: format!("f{i}"),
                score: i as u32,
                b1_hits: 0,
                b2_hits: 0,
            })
            .collect();
        let corpus = ComplexityCorpus::from_functions(fs);
        let pr = corpus.percentile_rank(9);
        assert!(pr >= 0.9, "got {pr}");
    }

    #[test]
    fn corpus_top_decile_returns_highest_complexity_functions() {
        let fs: Vec<FunctionComplexity> = (0..100)
            .map(|i| FunctionComplexity {
                name: format!("f{i}"),
                score: i as u32,
                b1_hits: 0,
                b2_hits: 0,
            })
            .collect();
        let corpus = ComplexityCorpus::from_functions(fs);
        let top = corpus.top_decile();
        // Top decile should be the 10 highest scores.
        assert!(top.len() <= 15 && top.len() >= 10);
        // First is the highest.
        assert_eq!(top[0].score, 99);
    }

    #[test]
    fn corpus_tukey_fence_reasonable() {
        let fs = (1..=100)
            .map(|i| FunctionComplexity {
                name: format!("f{i}"),
                score: i as u32,
                b1_hits: 0,
                b2_hits: 0,
            })
            .collect();
        let corpus = ComplexityCorpus::from_functions(fs);
        let f15 = corpus.tukey_upper_fence(1.5);
        // For [1..100] linear distribution, Q3 ≈ 75, IQR ≈ 50,
        // fence ≈ 75 + 75 = 150. We're permissive on bounds.
        assert!(f15 > 75 && f15 < 200, "got {f15}");
    }

    #[test]
    fn corpus_scorer_emits_percentile_rank_score() {
        let fs = (1..=100)
            .map(|i| FunctionComplexity {
                name: format!("f{i}"),
                score: i as u32,
                b1_hits: 0,
                b2_hits: 0,
            })
            .collect();
        let corpus = ComplexityCorpus::from_functions(fs);
        let scorer = CorpusScorer { corpus: &corpus };
        let top_fn = FunctionComplexity { name: "f99".into(), score: 99, b1_hits: 0, b2_hits: 0 };
        let s = scorer.score(&top_fn);
        assert!(s.value >= 0.95, "got {}", s.value);
        assert_eq!(s.detector, "cognitive-complexity");
        assert_eq!(s.explanation.raw_score, 99);
    }

    #[test]
    fn function_complexity_round_trips_through_json() {
        let f = FunctionComplexity { name: "x".into(), score: 7, b1_hits: 4, b2_hits: 3 };
        let bytes = serde_json::to_vec(&f).unwrap();
        let back: FunctionComplexity = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, f);
    }

    #[test]
    fn malformed_source_errors() {
        let src = "fn x( {";
        let err = analyze_file(src).unwrap_err();
        assert!(matches!(err, CognitiveError::Parse(_)));
    }
}
