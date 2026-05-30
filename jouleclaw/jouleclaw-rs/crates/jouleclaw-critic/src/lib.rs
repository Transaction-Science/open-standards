//! Writer-critic pattern — fresh-context adversarial gate over an
//! `(artifact, rubric)` pair, with a falsification pass before promotion.
//!
//! The field has converged on a specific shape for reliability in
//! agentic systems: a **second, fresh-context** agent inspects the
//! writer's output against an explicit rubric, then a **falsification
//! step** tries to refute each finding before it becomes a verdict.
//! The trick is twofold — the critic must *not* see the writer's trace
//! (no self-justification leaks), and tool-grounded graders (tests,
//! schemas, lints) beat free-form LLM scoring whenever they exist.
//!
//! This crate is a **pattern crate, not a tier**. Any caller can wrap a
//! resolution in a critique step; the critic does not slot into the
//! cascade's first-non-refused-wins walk. That keeps the doctrinal
//! clarity (one critic shape, used where it earns its energy) without
//! committing every resolution to the cost.
//!
//! ## Architecture
//!
//! ```text
//!   artifact ──────────┐
//!                      ▼
//!   rubric ────► Critic::critique ────► Vec<Finding>
//!                                            │
//!                                            ▼
//!                              Falsifier::try_falsify (per finding)
//!                                            │
//!                                            ▼
//!                              CritiqueReport { findings, overall }
//!                                            │
//!                                            ▼
//!                              promote_if_clean → Verdict
//! ```
//!
//! ## Honest scope (v1)
//!
//! - **Deterministic graders ground the critic.** Each rubric criterion
//!   either carries a [`jouleclaw_verify::OutputVerifier`] (which the
//!   reference [`DeterministicCritic`] runs) or it is marked
//!   [`GraderRef::LlmOnly`] and **skipped** in deterministic mode.
//!   Wiring an LLM critic for `LlmOnly` criteria is the L3 extension
//!   point — implement the [`Critic`] trait over your model backend.
//! - **The falsifier is conservative by default.** [`NoFalsifier`] never
//!   drops a finding; callers plug in their own [`Falsifier`] when they
//!   have a refutation strategy (e.g. re-run with a tighter verifier;
//!   re-grade after a whitespace trim).
//! - The critic does NOT call the writer's model, does NOT see the
//!   writer's trace, and does NOT promote anything by itself — the
//!   verdict is returned for the caller to act on.

#![forbid(unsafe_code)]

use jouleclaw_verify::{OutputVerifier, VerifyResult};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────
// Artifact + rubric
// ─────────────────────────────────────────────────────────────────────

/// What the artifact is — used by callers to route criteria
/// appropriately (e.g. only run a `cargo test` grader on `Code`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Text,
    Code,
    Json,
    Markdown,
    /// Caller-defined kind. The string is the discriminator (e.g.
    /// `"yaml"`, `"sql"`).
    Other,
}

/// The artifact under critique. The critic sees only this — never the
/// writer's prompt or reasoning trace.
#[derive(Debug, Clone)]
pub struct Artifact {
    pub kind: ArtifactKind,
    pub bytes: Vec<u8>,
}

impl Artifact {
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            kind: ArtifactKind::Text,
            bytes: s.into().into_bytes(),
        }
    }
    pub fn code(s: impl Into<String>) -> Self {
        Self {
            kind: ArtifactKind::Code,
            bytes: s.into().into_bytes(),
        }
    }
    pub fn json(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            kind: ArtifactKind::Json,
            bytes: bytes.into(),
        }
    }
}

/// How a criterion is graded.
///
/// `Verifier` is the deterministic L1 path — the reference
/// [`DeterministicCritic`] runs it and turns any failure into a finding.
/// `LlmOnly` is the open extension point: skipped in deterministic mode,
/// picked up by a consumer-supplied LLM critic.
pub enum GraderRef {
    Verifier(Box<dyn OutputVerifier>),
    LlmOnly,
}

impl std::fmt::Debug for GraderRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraderRef::Verifier(v) => write!(f, "Verifier({})", v.name()),
            GraderRef::LlmOnly => write!(f, "LlmOnly"),
        }
    }
}

/// One criterion in a rubric.
pub struct Criterion {
    /// Stable, short identifier — appears in findings and receipts.
    pub name: String,
    /// One-line description of what is being checked.
    pub description: String,
    /// How this criterion is graded.
    pub grader: GraderRef,
    /// Severity assigned when this criterion fails. Defaults to
    /// [`Severity::Fail`].
    pub severity: Severity,
}

impl Criterion {
    /// Build a deterministic criterion from any [`OutputVerifier`].
    pub fn verifier<V: OutputVerifier + 'static>(
        name: impl Into<String>,
        description: impl Into<String>,
        verifier: V,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            grader: GraderRef::Verifier(Box::new(verifier)),
            severity: Severity::Fail,
        }
    }
    /// Build a criterion that has no deterministic grader — the
    /// reference critic skips it; consumers route it to an LLM critic.
    pub fn llm_only(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            grader: GraderRef::LlmOnly,
            severity: Severity::Fail,
        }
    }
    /// Demote this criterion's failure severity to [`Severity::Warn`].
    pub fn warn_only(mut self) -> Self {
        self.severity = Severity::Warn;
        self
    }
}

/// A rubric is an ordered list of criteria. Order is stable so that
/// findings and the report are deterministic.
#[derive(Default)]
pub struct Rubric {
    criteria: Vec<Criterion>,
}

impl Rubric {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with(mut self, c: Criterion) -> Self {
        self.criteria.push(c);
        self
    }
    pub fn push(&mut self, c: Criterion) {
        self.criteria.push(c);
    }
    pub fn criteria(&self) -> &[Criterion] {
        &self.criteria
    }
    pub fn len(&self) -> usize {
        self.criteria.len()
    }
    pub fn is_empty(&self) -> bool {
        self.criteria.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Findings + report + verdict
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Advisory — does NOT block promotion by default.
    Warn,
    /// Blocking — failure means the overall verdict is [`Verdict::Fail`].
    Fail,
}

impl Default for Severity {
    fn default() -> Self {
        Severity::Fail
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Finding {
    pub criterion: String,
    pub severity: Severity,
    pub reason: String,
    /// `true` if a [`Falsifier`] failed to refute this finding. The
    /// reference [`NoFalsifier`] always returns `false` (refutation
    /// "could not be performed"); callers MUST treat absence of
    /// falsification as inconclusive — the finding stands.
    pub falsified_attempted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CritiqueReport {
    pub findings: Vec<Finding>,
    pub overall: Verdict,
    /// Sum of [`OutputVerifier::declared_cost_uj`] across criteria that
    /// were actually graded. LLM-only criteria contribute 0 (the
    /// deterministic critic skips them).
    pub joules_uj: u64,
}

impl CritiqueReport {
    /// Convenience: did the overall verdict pass (no blocking findings)?
    pub fn is_clean(&self) -> bool {
        matches!(self.overall, Verdict::Pass | Verdict::Warn)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Critic + Falsifier traits
// ─────────────────────────────────────────────────────────────────────

/// The critic: takes an artifact + rubric, returns findings + verdict.
/// MUST NOT see the writer's prompt, trace, or model identity — the
/// whole point is fresh context.
pub trait Critic: Send + Sync {
    fn critique(&self, artifact: &Artifact, rubric: &Rubric) -> CritiqueReport;
}

/// The falsifier: given a single finding, attempt to refute it. Returns
/// `true` only when the finding is *refuted* (drop it from the report);
/// returns `false` when the finding stands (the refutation either
/// failed or was not attempted).
pub trait Falsifier: Send + Sync {
    fn try_falsify(&self, artifact: &Artifact, finding: &Finding) -> bool;
}

// ─────────────────────────────────────────────────────────────────────
// Reference implementations
// ─────────────────────────────────────────────────────────────────────

/// Reference critic: runs each [`GraderRef::Verifier`] criterion; turns
/// any [`VerifyResult::Fail`] into a [`Finding`]; skips
/// [`GraderRef::LlmOnly`] criteria.
#[derive(Debug, Default, Clone, Copy)]
pub struct DeterministicCritic;

impl Critic for DeterministicCritic {
    fn critique(&self, artifact: &Artifact, rubric: &Rubric) -> CritiqueReport {
        let mut findings = Vec::new();
        let mut joules_uj: u64 = 0;
        for c in rubric.criteria() {
            let GraderRef::Verifier(v) = &c.grader else {
                continue;
            };
            joules_uj = joules_uj.saturating_add(v.declared_cost_uj());
            let r = v.verify(&artifact.bytes);
            if let VerifyResult::Fail { reason } = r {
                findings.push(Finding {
                    criterion: c.name.clone(),
                    severity: c.severity,
                    reason,
                    falsified_attempted: false,
                });
            }
        }
        let overall = overall_verdict(&findings);
        CritiqueReport {
            findings,
            overall,
            joules_uj,
        }
    }
}

/// Compute the overall verdict from a finding list. Any blocking failure
/// → `Fail`; any warning-only finding → `Warn`; otherwise `Pass`.
pub fn overall_verdict(findings: &[Finding]) -> Verdict {
    if findings.iter().any(|f| f.severity == Severity::Fail) {
        Verdict::Fail
    } else if !findings.is_empty() {
        Verdict::Warn
    } else {
        Verdict::Pass
    }
}

/// Reference falsifier — never refutes. Honest about not attempting
/// refutation, so callers treat the finding as it stands.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoFalsifier;

impl Falsifier for NoFalsifier {
    fn try_falsify(&self, _artifact: &Artifact, _finding: &Finding) -> bool {
        false
    }
}

/// A falsifier that re-checks a finding using a caller-supplied
/// secondary verifier list — if ANY secondary verifier *passes* the
/// artifact, the finding is refuted (false positive). Useful when the
/// primary grader is strict and a known-equivalent looser grader exists.
pub struct SecondaryVerifierFalsifier {
    secondaries: Vec<Box<dyn OutputVerifier>>,
}

impl SecondaryVerifierFalsifier {
    pub fn new() -> Self {
        Self {
            secondaries: Vec::new(),
        }
    }
    pub fn with<V: OutputVerifier + 'static>(mut self, v: V) -> Self {
        self.secondaries.push(Box::new(v));
        self
    }
}

impl Default for SecondaryVerifierFalsifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Falsifier for SecondaryVerifierFalsifier {
    fn try_falsify(&self, artifact: &Artifact, _finding: &Finding) -> bool {
        for v in &self.secondaries {
            if v.verify(&artifact.bytes).is_pass() {
                return true;
            }
        }
        false
    }
}

// ─────────────────────────────────────────────────────────────────────
// Full pipeline
// ─────────────────────────────────────────────────────────────────────

/// Run the full critique pipeline: critic → falsifier per finding →
/// recomputed verdict. Findings the falsifier refutes are dropped; the
/// rest carry `falsified_attempted = true`.
pub fn critique_and_falsify<C: Critic + ?Sized, F: Falsifier + ?Sized>(
    artifact: &Artifact,
    rubric: &Rubric,
    critic: &C,
    falsifier: &F,
) -> CritiqueReport {
    let mut report = critic.critique(artifact, rubric);
    let mut kept = Vec::with_capacity(report.findings.len());
    for mut f in report.findings.drain(..) {
        if falsifier.try_falsify(artifact, &f) {
            continue; // refuted, drop
        }
        f.falsified_attempted = true;
        kept.push(f);
    }
    report.findings = kept;
    report.overall = overall_verdict(&report.findings);
    report
}

/// Convenience: critique + falsify, return `Ok(())` on clean verdict,
/// `Err(report)` otherwise. Useful as a promotion gate.
pub fn promote_if_clean<C: Critic + ?Sized, F: Falsifier + ?Sized>(
    artifact: &Artifact,
    rubric: &Rubric,
    critic: &C,
    falsifier: &F,
) -> Result<CritiqueReport, CritiqueReport> {
    let report = critique_and_falsify(artifact, rubric, critic, falsifier);
    if report.is_clean() {
        Ok(report)
    } else {
        Err(report)
    }
}

// ─────────────────────────────────────────────────────────────────────
// LLM-backed critic
// ─────────────────────────────────────────────────────────────────────

/// LLM-backed critic — wraps a [`jouleclaw_llm_cheap::LlmBackend`] to
/// evaluate the artifact against the rubric in fresh context.
///
/// Doctrine — the same "fresh context, no writer state" rule the
/// deterministic critic enforces structurally is enforced here by
/// **prompt discipline**: the call only sees the artifact bytes and
/// the rubric's criterion text. It does NOT see the writer's prompt,
/// model name, or any trace. Lying via prompt smuggling would break
/// the contract; this implementation refuses to take such hints.
///
/// Output format — the LLM is asked to reply with JSON
/// `{"findings": [{"criterion": ..., "severity": "fail"|"warn",
///   "reason": ...}]}`. On parse failure, the critic emits a single
/// blocking [`Finding`] tagged `"parser"` so the artifact does NOT
/// silently pass when the model produced garbage. That is the honest
/// behaviour: an LLM that won't conform is a failed grader, not a
/// passing one.
///
/// Cost honesty — the LLM call's joule spend (from
/// [`jouleclaw_llm_cheap::LlmResponse::energy_joules`] if reported, or
/// the backend's `typical_joules_per_call` fallback) is summed into
/// the report's `joules_uj`. Deterministic verifier criteria handled
/// in the same rubric add their `declared_cost_uj` on top — this
/// adapter does not run them; consumers combine `LlmCritic` and
/// `DeterministicCritic` via `MultiCritic` (a follow-on).
///
/// Honest scope (v1):
///
/// - Only [`GraderRef::LlmOnly`] criteria are evaluated. Verifier
///   criteria are skipped, exactly inverting the deterministic
///   reference.
/// - One LLM call per critique, not one-per-criterion. Cheaper, but
///   means the prompt holds the whole rubric in context.
/// - No multi-turn dialogue, no chain-of-thought scaffolding. The
///   reply must be JSON.
pub struct LlmCritic {
    backend: std::sync::Arc<dyn jouleclaw_llm_cheap::LlmBackend>,
    max_tokens: u32,
    temperature: f32,
}

impl LlmCritic {
    /// Build an LLM critic with a backend and a default 512-token cap.
    pub fn new(backend: std::sync::Arc<dyn jouleclaw_llm_cheap::LlmBackend>) -> Self {
        Self {
            backend,
            max_tokens: 512,
            temperature: 0.0,
        }
    }

    /// Override the per-call token cap (default 512).
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens.max(1);
        self
    }

    /// Override the sampling temperature (default 0.0 — greedy, to keep
    /// critique deterministic across runs on a deterministic backend).
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    fn render_prompt(&self, artifact: &Artifact, rubric: &Rubric) -> String {
        let mut p = String::new();
        p.push_str("You are a strict critic. Evaluate the ARTIFACT against the RUBRIC.\n");
        p.push_str(
            "Reply with ONLY a JSON object: \
             {\"findings\": [{\"criterion\": \"<name>\", \"severity\": \"fail\"|\"warn\", \"reason\": \"<one sentence>\"}]}.\n",
        );
        p.push_str("Emit no finding for a criterion the artifact satisfies. Do not invent criteria.\n\n");
        p.push_str("RUBRIC:\n");
        for c in rubric.criteria() {
            if !matches!(c.grader, GraderRef::LlmOnly) {
                continue;
            }
            p.push_str(&format!(
                "- {} ({}): {}\n",
                c.name,
                match c.severity {
                    Severity::Fail => "fail",
                    Severity::Warn => "warn",
                },
                c.description
            ));
        }
        p.push_str("\nARTIFACT:\n");
        let text = match artifact.kind {
            ArtifactKind::Text | ArtifactKind::Code | ArtifactKind::Json => {
                String::from_utf8_lossy(&artifact.bytes).into_owned()
            }
            _ => format!("<binary, {} bytes>", artifact.bytes.len()),
        };
        p.push_str(&text);
        p
    }

    /// Estimate the joule spend of a single LLM critique call, using
    /// the backend's `typical_joules_per_call` (converted to µJ). This
    /// is the *estimator* — the actual spend, if reported via
    /// [`jouleclaw_llm_cheap::LlmResponse::energy_joules`], overrides
    /// it during `critique`.
    pub fn estimated_call_cost_uj(&self) -> u64 {
        (self.backend.typical_joules_per_call() * 1_000_000.0) as u64
    }
}

impl Critic for LlmCritic {
    fn critique(&self, artifact: &Artifact, rubric: &Rubric) -> CritiqueReport {
        // Quick exit: rubric has no LLM-only criteria — empty critique.
        let llm_only_count = rubric
            .criteria()
            .iter()
            .filter(|c| matches!(c.grader, GraderRef::LlmOnly))
            .count();
        if llm_only_count == 0 {
            return CritiqueReport {
                findings: Vec::new(),
                overall: Verdict::Pass,
                joules_uj: 0,
            };
        }

        let prompt = self.render_prompt(artifact, rubric);
        let mut req = jouleclaw_llm_cheap::LlmRequest::from_prompt(prompt, self.max_tokens);
        req.temperature = self.temperature;

        let resp = match self.backend.complete(&req) {
            Ok(r) => r,
            Err(e) => {
                // Backend failure → one blocking finding so the
                // artifact does not pass silently.
                return CritiqueReport {
                    findings: vec![Finding {
                        criterion: "backend".into(),
                        severity: Severity::Fail,
                        reason: format!("LLM critic backend error: {e}"),
                        falsified_attempted: false,
                    }],
                    overall: Verdict::Fail,
                    joules_uj: self.estimated_call_cost_uj(),
                };
            }
        };

        let joules_uj = match resp.energy_joules {
            Some(j) => (j * 1_000_000.0) as u64,
            None => self.estimated_call_cost_uj(),
        };

        // Parse the reply. On failure → one blocking finding.
        let parsed: serde_json::Value = match serde_json::from_str(resp.text.trim()) {
            Ok(v) => v,
            Err(_) => {
                return CritiqueReport {
                    findings: vec![Finding {
                        criterion: "parser".into(),
                        severity: Severity::Fail,
                        reason: format!(
                            "LLM critic reply was not valid JSON: {}",
                            resp.text.chars().take(120).collect::<String>()
                        ),
                        falsified_attempted: false,
                    }],
                    overall: Verdict::Fail,
                    joules_uj,
                };
            }
        };

        let findings = parsed
            .get("findings")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        // Build the set of LLM-only criterion names so we reject
        // hallucinated criteria (the model invented a check we didn't
        // ask for).
        let known: std::collections::HashSet<&str> = rubric
            .criteria()
            .iter()
            .filter(|c| matches!(c.grader, GraderRef::LlmOnly))
            .map(|c| c.name.as_str())
            .collect();

        let mut out = Vec::with_capacity(findings.len());
        for f in findings {
            let criterion = match f.get("criterion").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            if !known.contains(criterion.as_str()) {
                // Hallucinated criterion — silently drop. Honest
                // because the contract is "evaluate the rubric," not
                // "invent more checks."
                continue;
            }
            let severity = match f.get("severity").and_then(|v| v.as_str()) {
                Some("warn") => Severity::Warn,
                _ => Severity::Fail,
            };
            let reason = f
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            out.push(Finding {
                criterion,
                severity,
                reason,
                falsified_attempted: false,
            });
        }

        let overall = overall_verdict(&out);
        CritiqueReport {
            findings: out,
            overall,
            joules_uj,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifier that passes iff the output equals a fixed expected bytes.
    struct EqVerifier {
        expected: Vec<u8>,
    }
    impl EqVerifier {
        fn new(s: &str) -> Self {
            Self {
                expected: s.as_bytes().to_vec(),
            }
        }
    }
    impl OutputVerifier for EqVerifier {
        fn name(&self) -> &str {
            "verify:eq"
        }
        fn verify(&self, output: &[u8]) -> VerifyResult {
            if output == self.expected.as_slice() {
                VerifyResult::Pass
            } else {
                VerifyResult::fail("not equal to expected")
            }
        }
        fn declared_cost_uj(&self) -> u64 {
            1
        }
    }

    /// Verifier that passes iff the output starts with a given prefix.
    struct PrefixVerifier {
        prefix: Vec<u8>,
        name: String,
    }
    impl PrefixVerifier {
        fn new(name: &str, prefix: &str) -> Self {
            Self {
                prefix: prefix.as_bytes().to_vec(),
                name: format!("verify:prefix/{name}"),
            }
        }
    }
    impl OutputVerifier for PrefixVerifier {
        fn name(&self) -> &str {
            &self.name
        }
        fn verify(&self, output: &[u8]) -> VerifyResult {
            if output.starts_with(&self.prefix) {
                VerifyResult::Pass
            } else {
                VerifyResult::fail("prefix mismatch")
            }
        }
        fn declared_cost_uj(&self) -> u64 {
            1
        }
    }

    #[test]
    fn clean_artifact_passes() {
        let r = Rubric::new().with(Criterion::verifier(
            "exact",
            "must equal 'hello'",
            EqVerifier::new("hello"),
        ));
        let a = Artifact::text("hello");
        let report = critique_and_falsify(&a, &r, &DeterministicCritic, &NoFalsifier);
        assert_eq!(report.overall, Verdict::Pass);
        assert!(report.findings.is_empty());
        assert!(report.is_clean());
        assert_eq!(report.joules_uj, 1);
    }

    #[test]
    fn failing_criterion_blocks() {
        let r = Rubric::new().with(Criterion::verifier(
            "exact",
            "must equal 'hello'",
            EqVerifier::new("hello"),
        ));
        let a = Artifact::text("goodbye");
        let report = critique_and_falsify(&a, &r, &DeterministicCritic, &NoFalsifier);
        assert_eq!(report.overall, Verdict::Fail);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].criterion, "exact");
        assert!(report.findings[0].falsified_attempted);
    }

    #[test]
    fn llm_only_criterion_is_skipped_by_deterministic_critic() {
        let r = Rubric::new()
            .with(Criterion::llm_only("tone", "must be respectful"))
            .with(Criterion::verifier(
                "prefix",
                "must start with 'Hello'",
                PrefixVerifier::new("greeting", "Hello"),
            ));
        let a = Artifact::text("Hello, world");
        let report = critique_and_falsify(&a, &r, &DeterministicCritic, &NoFalsifier);
        // Only the deterministic criterion was graded; LLM-only skipped.
        assert_eq!(report.overall, Verdict::Pass);
        assert_eq!(report.joules_uj, 1);
    }

    #[test]
    fn warn_severity_does_not_block_promotion() {
        let r = Rubric::new().with(
            Criterion::verifier("optional", "nice to have", EqVerifier::new("ideal"))
                .warn_only(),
        );
        let a = Artifact::text("close enough");
        let report = critique_and_falsify(&a, &r, &DeterministicCritic, &NoFalsifier);
        assert_eq!(report.overall, Verdict::Warn);
        assert!(report.is_clean(), "Warn passes promote_if_clean");
        assert!(promote_if_clean(&a, &r, &DeterministicCritic, &NoFalsifier).is_ok());
    }

    #[test]
    fn falsifier_drops_a_finding_when_secondary_passes() {
        // Primary requires exact "hello"; secondary tolerates any "Hello*".
        let r = Rubric::new().with(Criterion::verifier(
            "exact",
            "strict equality",
            EqVerifier::new("hello"),
        ));
        let f =
            SecondaryVerifierFalsifier::new().with(PrefixVerifier::new("loose", "Hello"));
        let a = Artifact::text("Hello, world");
        let report = critique_and_falsify(&a, &r, &DeterministicCritic, &f);
        // Primary fails (artifact != "hello"); secondary passes (prefix
        // "Hello") so the finding is refuted; verdict flips to Pass.
        assert_eq!(report.overall, Verdict::Pass);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn promote_if_clean_returns_err_on_failure() {
        let r = Rubric::new().with(Criterion::verifier(
            "exact",
            "strict",
            EqVerifier::new("hello"),
        ));
        let a = Artifact::text("nope");
        match promote_if_clean(&a, &r, &DeterministicCritic, &NoFalsifier) {
            Err(report) => {
                assert_eq!(report.overall, Verdict::Fail);
                assert_eq!(report.findings.len(), 1);
            }
            Ok(_) => panic!("expected Err on failing critique"),
        }
    }

    #[test]
    fn multiple_failures_aggregate() {
        let r = Rubric::new()
            .with(Criterion::verifier(
                "eq",
                "exact",
                EqVerifier::new("hello"),
            ))
            .with(Criterion::verifier(
                "prefix",
                "must start with 'Goodbye'",
                PrefixVerifier::new("g", "Goodbye"),
            ));
        let a = Artifact::text("nothing fits");
        let report = critique_and_falsify(&a, &r, &DeterministicCritic, &NoFalsifier);
        assert_eq!(report.overall, Verdict::Fail);
        assert_eq!(report.findings.len(), 2);
    }

    #[test]
    fn empty_rubric_passes_trivially() {
        let r = Rubric::new();
        let a = Artifact::text("anything");
        let report = critique_and_falsify(&a, &r, &DeterministicCritic, &NoFalsifier);
        assert_eq!(report.overall, Verdict::Pass);
        assert_eq!(report.joules_uj, 0);
    }

    #[test]
    fn report_serializes_round_trip() {
        let r = Rubric::new().with(Criterion::verifier(
            "exact",
            "must equal 'hi'",
            EqVerifier::new("hi"),
        ));
        let a = Artifact::text("bye");
        let report = critique_and_falsify(&a, &r, &DeterministicCritic, &NoFalsifier);
        let json = serde_json::to_string(&report).expect("ser");
        let back: CritiqueReport = serde_json::from_str(&json).expect("deser");
        assert_eq!(back, report);
    }

    #[test]
    fn no_falsifier_is_honest_about_not_attempting() {
        let r = Rubric::new().with(Criterion::verifier(
            "exact",
            "strict",
            EqVerifier::new("hi"),
        ));
        let a = Artifact::text("bye");
        let report = critique_and_falsify(&a, &r, &DeterministicCritic, &NoFalsifier);
        // The finding stands; falsified_attempted is set so the audit
        // trail records that we tried (and got `false`, meaning "could
        // not refute") rather than silently skipping.
        assert_eq!(report.findings.len(), 1);
        assert!(report.findings[0].falsified_attempted);
    }

    // ─── LlmCritic ───────────────────────────────────────────────────

    /// Backend that returns a fixed text reply.
    struct ScriptedBackend(String);
    impl jouleclaw_llm_cheap::LlmBackend for ScriptedBackend {
        fn model_name(&self) -> &str {
            "test:scripted"
        }
        fn complete(
            &self,
            _req: &jouleclaw_llm_cheap::LlmRequest,
        ) -> Result<jouleclaw_llm_cheap::LlmResponse, jouleclaw_llm_cheap::LlmError> {
            Ok(jouleclaw_llm_cheap::LlmResponse {
                text: self.0.clone(),
                finish_reason: jouleclaw_llm_cheap::FinishReason::Stop,
                input_tokens: 0,
                output_tokens: self.0.len() as u32,
                energy_joules: Some(0.001),
            })
        }
    }

    struct ErrBackend;
    impl jouleclaw_llm_cheap::LlmBackend for ErrBackend {
        fn model_name(&self) -> &str {
            "test:err"
        }
        fn complete(
            &self,
            _req: &jouleclaw_llm_cheap::LlmRequest,
        ) -> Result<jouleclaw_llm_cheap::LlmResponse, jouleclaw_llm_cheap::LlmError> {
            Err(jouleclaw_llm_cheap::LlmError::Upstream("nope".into()))
        }
    }

    fn llm_rubric() -> Rubric {
        Rubric::new()
            .with(Criterion::llm_only("style", "Prose is clear and concise."))
            .with(Criterion::llm_only("tone", "No hype, no marketing language."))
    }

    #[test]
    fn llm_critic_skips_when_rubric_has_no_llm_criteria() {
        let r = Rubric::new().with(Criterion::verifier(
            "exact",
            "strict",
            EqVerifier::new("hi"),
        ));
        let lc = LlmCritic::new(std::sync::Arc::new(ScriptedBackend("{}".into())));
        let report = lc.critique(&Artifact::text("hi"), &r);
        assert!(report.findings.is_empty());
        assert!(matches!(report.overall, Verdict::Pass));
        assert_eq!(report.joules_uj, 0);
    }

    #[test]
    fn llm_critic_passes_when_reply_has_no_findings() {
        let lc = LlmCritic::new(std::sync::Arc::new(ScriptedBackend(
            r#"{"findings": []}"#.into(),
        )));
        let report = lc.critique(&Artifact::text("clean prose."), &llm_rubric());
        assert!(report.is_clean());
        assert_eq!(report.findings.len(), 0);
        assert!(report.joules_uj > 0, "joules accounted from response");
    }

    #[test]
    fn llm_critic_reports_findings_for_known_criteria() {
        let lc = LlmCritic::new(std::sync::Arc::new(ScriptedBackend(
            r#"{"findings": [{"criterion": "tone", "severity": "fail", "reason": "uses hype words"}]}"#.into(),
        )));
        let report = lc.critique(&Artifact::text("revolutionary breakthrough."), &llm_rubric());
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].criterion, "tone");
        assert!(matches!(report.findings[0].severity, Severity::Fail));
        assert!(matches!(report.overall, Verdict::Fail));
    }

    #[test]
    fn llm_critic_drops_hallucinated_criteria() {
        let lc = LlmCritic::new(std::sync::Arc::new(ScriptedBackend(
            r#"{"findings": [
                {"criterion": "made-up", "severity": "fail", "reason": "x"},
                {"criterion": "tone", "severity": "warn", "reason": "y"}
            ]}"#
            .into(),
        )));
        let report = lc.critique(&Artifact::text("text"), &llm_rubric());
        assert_eq!(report.findings.len(), 1, "hallucinated criterion dropped");
        assert_eq!(report.findings[0].criterion, "tone");
        assert!(matches!(report.findings[0].severity, Severity::Warn));
    }

    #[test]
    fn llm_critic_fails_on_unparseable_reply() {
        let lc = LlmCritic::new(std::sync::Arc::new(ScriptedBackend(
            "not json at all".into(),
        )));
        let report = lc.critique(&Artifact::text("x"), &llm_rubric());
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].criterion, "parser");
        assert!(matches!(report.overall, Verdict::Fail));
    }

    #[test]
    fn llm_critic_fails_on_backend_error() {
        let lc = LlmCritic::new(std::sync::Arc::new(ErrBackend));
        let report = lc.critique(&Artifact::text("x"), &llm_rubric());
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].criterion, "backend");
        assert!(matches!(report.overall, Verdict::Fail));
        assert!(report.joules_uj > 0, "estimated cost still accounted");
    }

    #[test]
    fn llm_critic_composes_through_the_full_pipeline() {
        let lc = LlmCritic::new(std::sync::Arc::new(ScriptedBackend(
            r#"{"findings": [{"criterion": "style", "severity": "warn", "reason": "wordy"}]}"#
                .into(),
        )));
        let report = critique_and_falsify(
            &Artifact::text("the the the"),
            &llm_rubric(),
            &lc,
            &NoFalsifier,
        );
        assert_eq!(report.findings.len(), 1);
        assert!(report.findings[0].falsified_attempted);
        assert!(matches!(report.overall, Verdict::Warn));
        assert!(report.is_clean(), "warn does not block promotion");
    }
}
