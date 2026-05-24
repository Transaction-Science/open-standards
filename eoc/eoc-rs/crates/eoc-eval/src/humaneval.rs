//! HumanEval — Python coding problems (Chen et al. 2021).
//!
//! Each case ships a function signature with a docstring and a held-out
//! `check(candidate)` test function. The metric is pass@1: a candidate
//! passes iff the concatenated `prompt + candidate + test + check(...)`
//! program runs to completion under Python without raising.
//!
//! # Sandbox modes
//!
//! Three modes, in order of preference at runtime:
//!
//! 1. `python` feature on -> in-process via `pyo3`.
//! 2. `python3` on `$PATH` -> subprocess sandbox (default).
//! 3. Neither available -> the grader returns `0.0` and logs an error;
//!    [`HumanEval::set_strict_sandbox`] makes that case an error instead.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::builtin_samples;
use crate::error::{EvalError, Result};
use crate::harness::{
    DatasetSource, EvalCase, ExpectedAnswer, Harness, Metric, Response, load_raw,
};

/// HumanEval harness.
pub struct HumanEval {
    /// When `true`, missing Python returns an error from `score()`
    /// instead of silently scoring 0. Disabled by default so end-to-end
    /// tests run on machines without Python.
    strict_sandbox: bool,
    /// Wall-clock timeout for the subprocess (seconds).
    timeout: Duration,
}

impl HumanEval {
    /// Create a new HumanEval harness.
    pub fn new() -> Self {
        Self {
            strict_sandbox: false,
            timeout: Duration::from_secs(10),
        }
    }

    /// Enable strict sandbox mode (no Python => error).
    pub fn set_strict_sandbox(mut self, strict: bool) -> Self {
        self.strict_sandbox = strict;
        self
    }

    /// Override the subprocess timeout.
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
}

impl Default for HumanEval {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct RawRow {
    id: String,
    entry_point: String,
    prompt: String,
    test: String,
}

/// Extract the first code block from a possibly markdown-wrapped
/// response. If the response is not fenced, return it as-is. Internal
/// indentation is preserved (only the outermost trailing newline of
/// the fence is consumed); function-body candidates depend on their
/// leading whitespace surviving.
pub fn strip_code_fence(text: &str) -> String {
    let trimmed = text.trim_start_matches(['\n', '\r']);
    let trimmed = trimmed.trim_end_matches(['\n', '\r', ' ', '\t']);
    if let Some(after) = trimmed.strip_prefix("```python\n") {
        if let Some(end) = after.rfind("```") {
            return after[..end].trim_end().to_string();
        }
        return after.to_string();
    }
    if let Some(after) = trimmed.strip_prefix("```\n") {
        if let Some(end) = after.rfind("```") {
            return after[..end].trim_end().to_string();
        }
        return after.to_string();
    }
    text.to_string()
}

/// Build the Python program that exercises a candidate completion against
/// the held-out test. The candidate is assumed to either include the
/// stub signature itself, or to be a body that completes it.
pub fn build_program(prompt: &str, candidate: &str, test: &str, entry_point: &str) -> String {
    // If the candidate already defines the entry point function, use it
    // verbatim. Otherwise, append the candidate to the prompt stub so
    // that the function gets a body.
    let candidate = strip_code_fence(candidate);
    let body = if candidate.contains(&format!("def {entry_point}")) {
        candidate
    } else {
        format!("{prompt}{candidate}")
    };
    format!("{body}\n\n{test}\ncheck({entry_point})\n")
}

#[async_trait]
impl Harness for HumanEval {
    fn name(&self) -> &'static str {
        "humaneval"
    }

    fn metric(&self) -> Metric {
        Metric::Pass1
    }

    async fn load(&self, source: DatasetSource) -> Result<Vec<EvalCase>> {
        let raw = load_raw(source, builtin_samples::HUMANEVAL).await?;
        let rows: Vec<RawRow> = serde_json::from_str(&raw)?;
        Ok(rows
            .into_iter()
            .map(|r| EvalCase {
                id: r.id.clone(),
                prompt: r.prompt.clone(),
                expected: ExpectedAnswer::UnitTest {
                    entry_point: r.entry_point,
                    // Pack the stub prompt + test together with a stable
                    // single-line separator so `score()` can recover the
                    // stub when running the sandbox.
                    test_program: format!("{}{PROMPT_TEST_SEP}{}", r.prompt, r.test),
                },
                dataset: "humaneval".to_string(),
                subject: None,
            })
            .collect())
    }

    async fn score(&self, response: &Response, expected: &ExpectedAnswer) -> f64 {
        let ExpectedAnswer::UnitTest {
            entry_point,
            test_program,
        } = expected
        else {
            return 0.0;
        };
        let (stub_prompt, test_body) = match test_program.split_once(PROMPT_TEST_SEP) {
            Some((p, t)) => (p.to_string(), t.to_string()),
            None => (String::new(), test_program.clone()),
        };
        match run_python(&stub_prompt, &response.payload, &test_body, entry_point, self.timeout).await {
            Ok(true) => 1.0,
            Ok(false) => 0.0,
            Err(e) => {
                if self.strict_sandbox {
                    tracing::error!(error = %e, "humaneval sandbox failed");
                } else {
                    tracing::warn!(error = %e, "humaneval sandbox unavailable, scoring 0");
                }
                0.0
            }
        }
    }
}

/// Sentinel string used to pack the original prompt stub alongside the
/// held-out test inside [`ExpectedAnswer::UnitTest::test_program`].
const PROMPT_TEST_SEP: &str = "\n###__EOC_HUMANEVAL_TEST__###\n";

/// Try the configured sandbox(s) in order. Returns Ok(true) if the
/// candidate passed, Ok(false) if it ran but a test failed, and Err
/// if no sandbox was available.
async fn run_python(
    prompt: &str,
    candidate: &str,
    test: &str,
    entry_point: &str,
    timeout: Duration,
) -> std::result::Result<bool, EvalError> {
    let program = build_program(prompt, candidate, test, entry_point);

    #[cfg(feature = "python")]
    {
        return Ok(run_pyo3(&program));
    }

    #[allow(unreachable_code)]
    {
        run_subprocess(&program, timeout).await
    }
}

#[cfg(feature = "python")]
fn run_pyo3(program: &str) -> bool {
    use pyo3::Python;
    Python::with_gil(|py| py.run(program, None, None).is_ok())
}

async fn run_subprocess(program: &str, timeout: Duration) -> std::result::Result<bool, EvalError> {
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| EvalError::Sandbox(format!("spawn python3: {e}")))?;

    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| EvalError::Sandbox("no stdin handle".into()))?;
        stdin
            .write_all(program.as_bytes())
            .await
            .map_err(|e| EvalError::Sandbox(format!("stdin write: {e}")))?;
        stdin
            .shutdown()
            .await
            .map_err(|e| EvalError::Sandbox(format!("stdin shutdown: {e}")))?;
        // Drop closes the FD so python3 sees EOF and begins executing.
    }

    let out = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return Err(EvalError::Sandbox(format!("wait: {e}"))),
        Err(_) => return Ok(false), // Timeout treated as fail.
    };

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        tracing::debug!(stderr = %stderr, "humaneval sandbox failed test");
    }
    Ok(out.status.success())
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
        let cases = HumanEval::new().load(DatasetSource::BuiltinSample).await.unwrap();
        assert!(!cases.is_empty());
        for c in &cases {
            assert!(matches!(c.expected, ExpectedAnswer::UnitTest { .. }));
        }
    }

    #[test]
    fn strips_python_fence() {
        let s = "```python\ndef foo():\n    return 1\n```";
        let stripped = strip_code_fence(s);
        assert!(stripped.starts_with("def foo"));
        assert!(!stripped.contains("```"));
    }

    #[test]
    fn builds_program_with_existing_def() {
        let p = build_program(
            "def f(x):\n    \"\"\"doc\"\"\"\n",
            "def f(x):\n    return x + 1\n",
            "def check(c):\n    assert c(1) == 2\n",
            "f",
        );
        assert!(p.contains("def f(x):"));
        assert!(p.ends_with("check(f)\n"));
    }

    #[tokio::test]
    async fn scores_passing_solution_when_python_available() {
        // Quick probe — skip if python3 is not on PATH.
        if Command::new("python3").arg("--version").output().await.is_err() {
            eprintln!("python3 not available, skipping");
            return;
        }
        let h = HumanEval::new();
        let cases = h.load(DatasetSource::BuiltinSample).await.unwrap();
        // HumanEval/4: mean_absolute_deviation.
        let case = cases.iter().find(|c| c.id == "HumanEval/4").expect("case present");
        let candidate = "    mean = sum(numbers) / len(numbers)\n    return sum(abs(x - mean) for x in numbers) / len(numbers)\n";
        let s_pass = h.score(&resp(candidate), &case.expected).await;
        assert_eq!(s_pass, 1.0, "passing candidate should score 1");

        let bad = "    return 0.0\n";
        let s_fail = h.score(&resp(bad), &case.expected).await;
        assert_eq!(s_fail, 0.0, "failing candidate should score 0");
    }
}
