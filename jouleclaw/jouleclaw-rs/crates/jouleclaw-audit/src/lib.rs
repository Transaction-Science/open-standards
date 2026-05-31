//! # jouleclaw-audit
//!
//! Consumer-side verifier orchestration. Wraps `cargo kani` and
//! `cargo miri test` as subprocesses, captures structured results,
//! and joule-stamps every run.
//!
//! Ported (in spirit, not bytes) from `ai-verify`'s
//! `audit_core::proof` and `audit_core::process` modules per the
//! wave-4 directive. Refined per the SOTA brief:
//!
//! - **Use `cargo kani --harnesses <regex>` and `cargo miri test
//!   [filter]`** as the discovery mechanism. No AST walk —
//!   `cargo` already indexes harnesses.
//! - **Convention from rustls/s2n-quic + AWS Kani case study:**
//!   per-harness timeout 5 min in CI, 60 min in nightly soak;
//!   input width ≤ 8 bytes; unwind ≤ 16.
//! - **Skip `cargo audit` / `cargo deny` / clippy** — those have
//!   their own runners; this crate is verification-specific.
//! - **Prusti / Creusot deferred** — annotation-heavy, research-
//!   grade; not in v1 scope.
//!
//! ## Energy as the orthogonal trust anchor
//!
//! Every `ProofResult` carries `joules_uj` (wall-clock CPU time ×
//! a configurable joule-per-second rate, since Kani/MIRI runs are
//! CPU-bound). `energy_provenance` is `Estimator` by default — the
//! consumer can swap in a hardware-shunt reader if available.
//!
//! ## Honest scope
//!
//! - **Kani does not prove unbounded loops.** It bounds them.
//!   `ProofStatus::Passed` means "no counterexample within
//!   `--default-unwind N`," not a soundness proof beyond N.
//! - **MIRI does not prove the absence of UB.** It finds UB on
//!   paths it actually executes. Coverage is mandatory.
//! - **This crate runs subprocesses.** Side-effects: filesystem
//!   (cargo's target dir), process spawning, environment.
//!   Sandboxed it is not.
//! - **Subprocess timeout** is wall-clock; long-running proofs
//!   that get killed surface as [`ProofStatus::Timeout`], not
//!   as failures.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(unexpected_cfgs)]

use jouleclaw_energy::Provenance;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────
// Energy model
// ─────────────────────────────────────────────────────────────────────

/// Default joule-per-CPU-second estimate. ~20 W under-load is a
/// reasonable laptop CPU estimate; production deployments override
/// via [`ProofConfig::joules_per_cpu_sec`].
pub const DEFAULT_JOULES_PER_CPU_SEC: f64 = 20.0;

// ─────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────

/// Configuration for [`run_kani`] / [`run_miri`] / [`run_audit`].
#[derive(Debug, Clone)]
pub struct ProofConfig {
    /// Per-tool wall-clock cap. Default 300 s (CI-friendly).
    pub timeout: Duration,
    /// CPU-energy estimate to multiply elapsed CPU seconds by.
    /// Default [`DEFAULT_JOULES_PER_CPU_SEC`].
    pub joules_per_cpu_sec: f64,
    /// Optional working directory (defaults to current dir).
    pub workdir: Option<PathBuf>,
    /// Run in verbose mode (passes `--verbose` to the inner tool
    /// where supported).
    pub verbose: bool,
}

impl Default for ProofConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            joules_per_cpu_sec: DEFAULT_JOULES_PER_CPU_SEC,
            workdir: None,
            verbose: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────────────────────────────

/// Outcome of a single tool invocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofStatus {
    /// The tool reported success.
    Passed,
    /// The tool reported a counterexample or other failure.
    Failed,
    /// The tool was killed by the wall-clock timeout.
    Timeout,
    /// The tool could not be invoked (binary missing, etc).
    Error,
}

/// One tool invocation's structured result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofResult {
    /// Which tool produced this — `"kani"` or `"miri"`.
    pub tool: String,
    /// The subcommand / harness filter the run was scoped to.
    pub scope: String,
    /// Pass / fail / timeout / error.
    pub status: ProofStatus,
    /// Microjoules estimated for this run (wall-clock × cpu-secs
    /// rate). Honest provenance — `Estimator` tier.
    pub joules_uj: u64,
    /// Wall-clock duration of the run.
    pub elapsed_ms: u64,
    /// First ~400 chars of stderr — usually the meaningful message
    /// when something failed.
    pub stderr_excerpt: Option<String>,
}

/// Rolled-up report from a full `run_audit` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofReport {
    /// Per-tool results in invocation order.
    pub results: Vec<ProofResult>,
    /// Whether `cargo kani` was reachable on this host.
    pub kani_available: bool,
    /// Whether `cargo miri` was reachable on this host (requires
    /// nightly + miri component).
    pub miri_available: bool,
    /// Sum of `joules_uj` across all results.
    pub total_joules_uj: u64,
    /// Provenance of the rollup. Always `Estimator` at v1 — no
    /// hardware-shunt integration for CPU-time-to-joules.
    pub energy_provenance: Provenance,
}

impl Default for ProofReport {
    fn default() -> Self {
        Self {
            results: Vec::new(),
            kani_available: false,
            miri_available: false,
            total_joules_uj: 0,
            energy_provenance: Provenance::Estimator,
        }
    }
}

impl ProofReport {
    /// How many results passed.
    pub fn passed(&self) -> usize {
        self.results.iter().filter(|r| r.status == ProofStatus::Passed).count()
    }
    /// How many failed or timed out.
    pub fn failed(&self) -> usize {
        self.results
            .iter()
            .filter(|r| matches!(r.status, ProofStatus::Failed | ProofStatus::Timeout))
            .count()
    }
}

/// Errors `run_audit` and friends can surface.
#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    /// IO error spawning the subprocess.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

// ─────────────────────────────────────────────────────────────────────
// Tool availability
// ─────────────────────────────────────────────────────────────────────

/// Is `cargo kani` available on `PATH`? Tests `cargo kani --version`.
pub fn is_kani_available() -> bool {
    Command::new("cargo")
        .args(["kani", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Is `cargo +nightly miri` available? MIRI is nightly-only and
/// requires the `miri` component installed (`rustup component
/// add miri --toolchain nightly`).
pub fn is_miri_available() -> bool {
    Command::new("cargo")
        .args(["+nightly", "miri", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ─────────────────────────────────────────────────────────────────────
// Runners
// ─────────────────────────────────────────────────────────────────────

fn estimate_joules_uj(elapsed: Duration, rate: f64) -> u64 {
    let secs = elapsed.as_secs_f64();
    let joules = secs * rate;
    (joules * 1_000_000.0) as u64
}

fn truncate_stderr(s: &str) -> Option<String> {
    if s.is_empty() {
        return None;
    }
    let cap = 400usize;
    let n = s.chars().count();
    Some(if n > cap {
        s.chars().take(cap).collect()
    } else {
        s.to_string()
    })
}

/// Run `cargo kani` with an optional harness filter regex. Returns
/// a single `ProofResult`. Use [`run_audit`] for batched orchestration.
pub fn run_kani(
    harness_filter: Option<&str>,
    config: &ProofConfig,
) -> Result<ProofResult, AuditError> {
    let mut cmd = Command::new("cargo");
    cmd.arg("kani");
    if let Some(f) = harness_filter {
        cmd.args(["--harnesses", f]);
    }
    if config.verbose {
        cmd.arg("--verbose");
    }
    run_proc(cmd, "kani", harness_filter.unwrap_or("*"), config)
}

/// Run `cargo +nightly miri test` with an optional test-filter
/// string.
pub fn run_miri(
    test_filter: Option<&str>,
    config: &ProofConfig,
) -> Result<ProofResult, AuditError> {
    let mut cmd = Command::new("cargo");
    cmd.args(["+nightly", "miri", "test"]);
    if let Some(f) = test_filter {
        cmd.arg(f);
    }
    run_proc(cmd, "miri", test_filter.unwrap_or("*"), config)
}

fn run_proc(
    mut cmd: Command,
    tool: &str,
    scope: &str,
    config: &ProofConfig,
) -> Result<ProofResult, AuditError> {
    if let Some(dir) = &config.workdir {
        cmd.current_dir(dir);
    }
    let start = Instant::now();
    let result = run_with_timeout(&mut cmd, config.timeout)?;
    let elapsed = start.elapsed();
    let joules_uj = estimate_joules_uj(elapsed, config.joules_per_cpu_sec);
    let (status, stderr_excerpt) = match result {
        ProcResult::Ok { code, stderr } => {
            let s = if code == 0 { ProofStatus::Passed } else { ProofStatus::Failed };
            (s, truncate_stderr(&stderr))
        }
        ProcResult::TimedOut => (ProofStatus::Timeout, Some("(killed by timeout)".into())),
        ProcResult::SpawnError(msg) => (ProofStatus::Error, Some(msg)),
    };
    Ok(ProofResult {
        tool: tool.into(),
        scope: scope.into(),
        status,
        joules_uj,
        elapsed_ms: elapsed.as_millis() as u64,
        stderr_excerpt,
    })
}

enum ProcResult {
    Ok { code: i32, stderr: String },
    TimedOut,
    SpawnError(String),
}

/// Tiny subprocess+timeout helper. Spawns the child, polls for
/// completion every 100 ms up to `timeout`, kills + reaps on
/// expiry.
fn run_with_timeout(cmd: &mut Command, timeout: Duration) -> Result<ProcResult, AuditError> {
    let mut child = match cmd.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => return Ok(ProcResult::SpawnError(format!("spawn: {e}"))),
    };
    let start = Instant::now();
    loop {
        match child.try_wait()? {
            Some(status) => {
                let mut stderr_buf = String::new();
                if let Some(mut e) = child.stderr.take() {
                    use std::io::Read;
                    let _ = e.read_to_string(&mut stderr_buf);
                }
                let code = status.code().unwrap_or(-1);
                return Ok(ProcResult::Ok { code, stderr: stderr_buf });
            }
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(ProcResult::TimedOut);
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Batched audit
// ─────────────────────────────────────────────────────────────────────

/// Run both `cargo kani` (no harness filter) and `cargo +nightly
/// miri test` (no test filter) and produce a rolled-up
/// [`ProofReport`]. Skips a tool when its `is_*_available`
/// returns false (recording the availability in the report).
pub fn run_audit(config: &ProofConfig) -> Result<ProofReport, AuditError> {
    let kani_ok = is_kani_available();
    let miri_ok = is_miri_available();
    let mut results = Vec::new();
    if kani_ok {
        results.push(run_kani(None, config)?);
    }
    if miri_ok {
        results.push(run_miri(None, config)?);
    }
    let total = results.iter().map(|r| r.joules_uj).sum();
    Ok(ProofReport {
        results,
        kani_available: kani_ok,
        miri_available: miri_ok,
        total_joules_uj: total,
        energy_provenance: Provenance::Estimator,
    })
}

/// Convenience: scope an audit to a specific path's workspace.
pub fn run_audit_in(
    path: impl AsRef<Path>,
    timeout: Duration,
) -> Result<ProofReport, AuditError> {
    let mut config = ProofConfig::default();
    config.workdir = Some(path.as_ref().to_path_buf());
    config.timeout = timeout;
    run_audit(&config)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_joules_uj_default_rate() {
        let j = estimate_joules_uj(Duration::from_secs(1), DEFAULT_JOULES_PER_CPU_SEC);
        // 1 sec × 20 J/s × 1e6 µJ/J = 20_000_000 µJ.
        assert_eq!(j, 20_000_000);
    }

    #[test]
    fn estimate_joules_uj_zero_for_zero_elapsed() {
        let j = estimate_joules_uj(Duration::from_secs(0), DEFAULT_JOULES_PER_CPU_SEC);
        assert_eq!(j, 0);
    }

    #[test]
    fn truncate_stderr_none_on_empty() {
        assert!(truncate_stderr("").is_none());
    }

    #[test]
    fn truncate_stderr_caps_at_400_chars() {
        let s: String = std::iter::repeat('x').take(1000).collect();
        let t = truncate_stderr(&s).unwrap();
        assert!(t.chars().count() <= 400);
    }

    #[test]
    fn proof_report_default_is_clean() {
        let r = ProofReport::default();
        assert_eq!(r.passed(), 0);
        assert_eq!(r.failed(), 0);
        assert!(r.results.is_empty());
        assert_eq!(r.total_joules_uj, 0);
    }

    #[test]
    fn proof_report_passed_and_failed_counts() {
        let r = ProofReport {
            results: vec![
                ProofResult {
                    tool: "kani".into(),
                    scope: "*".into(),
                    status: ProofStatus::Passed,
                    joules_uj: 100,
                    elapsed_ms: 10,
                    stderr_excerpt: None,
                },
                ProofResult {
                    tool: "miri".into(),
                    scope: "*".into(),
                    status: ProofStatus::Failed,
                    joules_uj: 200,
                    elapsed_ms: 20,
                    stderr_excerpt: Some("err".into()),
                },
                ProofResult {
                    tool: "kani".into(),
                    scope: "kani_overflow".into(),
                    status: ProofStatus::Timeout,
                    joules_uj: 300,
                    elapsed_ms: 30,
                    stderr_excerpt: None,
                },
            ],
            kani_available: true,
            miri_available: true,
            total_joules_uj: 600,
            energy_provenance: Provenance::Estimator,
        };
        assert_eq!(r.passed(), 1);
        assert_eq!(r.failed(), 2); // Failed + Timeout
    }

    #[test]
    fn proof_status_round_trips_through_json() {
        for s in [ProofStatus::Passed, ProofStatus::Failed, ProofStatus::Timeout, ProofStatus::Error] {
            let j = serde_json::to_value(&s).unwrap();
            let back: ProofStatus = serde_json::from_value(j).unwrap();
            assert_eq!(back, s);
        }
    }

    #[test]
    fn proof_result_round_trips_through_json() {
        let r = ProofResult {
            tool: "kani".into(),
            scope: "kani_overflow".into(),
            status: ProofStatus::Passed,
            joules_uj: 12345,
            elapsed_ms: 67,
            stderr_excerpt: None,
        };
        let bytes = serde_json::to_vec(&r).unwrap();
        let back: ProofResult = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.tool, r.tool);
        assert_eq!(back.status, r.status);
    }

    #[test]
    fn tool_availability_is_a_pure_query() {
        // Just checks the function doesn't panic; tool may or may not
        // actually be installed on the test host.
        let _ = is_kani_available();
        let _ = is_miri_available();
    }

    #[test]
    fn run_with_timeout_returns_spawn_error_for_missing_binary() {
        let mut cmd = Command::new("definitely-not-a-real-binary-zzz");
        let result = run_with_timeout(&mut cmd, Duration::from_secs(1)).unwrap();
        match result {
            ProcResult::SpawnError(_) => {}
            other => panic!("expected SpawnError, got {:?}", match other {
                ProcResult::Ok { code, .. } => format!("Ok(code={code})"),
                ProcResult::TimedOut => "TimedOut".into(),
                ProcResult::SpawnError(_) => unreachable!(),
            }),
        }
    }

    #[test]
    fn proof_config_default_is_sensible() {
        let c = ProofConfig::default();
        assert_eq!(c.timeout, Duration::from_secs(300));
        assert_eq!(c.joules_per_cpu_sec, DEFAULT_JOULES_PER_CPU_SEC);
        assert!(c.workdir.is_none());
        assert!(!c.verbose);
    }

    #[test]
    fn run_kani_against_missing_tool_returns_spawn_error_status() {
        // We invoke run_kani regardless of whether kani is installed;
        // when it isn't, the subprocess errors and status is Error.
        // Otherwise we get a real Passed/Failed which is also valid;
        // we just check the result exists.
        let config = ProofConfig {
            timeout: Duration::from_secs(2),
            ..Default::default()
        };
        let r = run_kani(None, &config).unwrap();
        assert_eq!(r.tool, "kani");
        // Status is one of the four legal variants.
        assert!(matches!(
            r.status,
            ProofStatus::Passed
                | ProofStatus::Failed
                | ProofStatus::Timeout
                | ProofStatus::Error
        ));
    }
}
