//! CodeAct — tool calls expressed as code.
//!
//! Instead of structured `tool_call` JSON, the model emits a small code
//! block that the executor parses. The reference implementation here
//! supports a minimal "linelang" of `name(arg)` invocations, one per
//! line, executed sequentially. Production deployments swap in a real
//! sandboxed interpreter (Python, JS, KCL …) by replacing the
//! [`CodeRuntime`] trait.

use std::sync::Arc;

use async_trait::async_trait;
use crate::agent::{Agent, AgentLoop, Budget, LlmProvider, StopReason, Tool, ToolBox};
use crate::error::{AgentError, AgentResult};
use crate::trace::SpanKind;

/// Executor for the code blocks the model emits.
#[async_trait]
pub trait CodeRuntime: Send + Sync {
    /// Run `code` and return its rendered output (everything the model
    /// will see as the observation).
    async fn run(&self, code: &str) -> AgentResult<String>;
}

/// The reference "linelang" runtime: each non-blank line is `name(arg)`
/// and is dispatched against a [`ToolBox`]. Arguments are passed as
/// `{"input": arg}`.
pub struct LineLangRuntime {
    tools: Arc<ToolBox>,
}

impl LineLangRuntime {
    /// Build.
    pub fn new(tools: Arc<ToolBox>) -> Self {
        Self { tools }
    }

    /// Parse a single line into `(name, arg)`. Lenient — a bare `name`
    /// is treated as `name()`.
    pub fn parse_line(line: &str) -> AgentResult<(String, String)> {
        let line = line.trim();
        if line.is_empty() {
            return Err(AgentError::Parse("empty line".into()));
        }
        if let (Some(lp), Some(rp)) = (line.find('('), line.rfind(')')) {
            if lp < rp {
                let name = line[..lp].trim().to_string();
                let arg = line[lp + 1..rp].trim().to_string();
                return Ok((name, arg));
            }
        }
        Ok((line.to_string(), String::new()))
    }
}

#[async_trait]
impl CodeRuntime for LineLangRuntime {
    async fn run(&self, code: &str) -> AgentResult<String> {
        let mut out = String::new();
        for line in code.lines() {
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let (name, arg) = match LineLangRuntime::parse_line(line) {
                Ok(p) => p,
                Err(e) => {
                    out.push_str(&format!("parse_error: {e}\n"));
                    continue;
                }
            };
            let tool = match self.tools.get(&name) {
                Some(t) => t,
                None => {
                    out.push_str(&format!("unknown_tool: {name}\n"));
                    continue;
                }
            };
            let args = serde_json::json!({ "input": arg });
            match Tool::run(tool.as_ref(), args).await {
                Ok(v) => out.push_str(&format!("{name} => {v}\n")),
                Err(e) => out.push_str(&format!("{name} error: {e}\n")),
            }
        }
        Ok(out)
    }
}

/// CodeAct driver.
pub struct CodeActLoop {
    provider: Arc<dyn LlmProvider>,
    runtime: Arc<dyn CodeRuntime>,
    inner: AgentLoop,
}

impl CodeActLoop {
    /// Build with a vendor-neutral code runtime.
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        runtime: Arc<dyn CodeRuntime>,
        budget: Budget,
    ) -> Self {
        Self {
            provider,
            runtime,
            inner: AgentLoop::new(budget),
        }
    }

    /// Extract the first ```code … ``` block. Returns `None` if there
    /// is no fenced block — callers may then treat the whole reply as
    /// the final answer.
    pub fn extract_code(text: &str) -> Option<String> {
        let start = text.find("```")?;
        let after = &text[start + 3..];
        // Skip an optional language tag on the same line.
        let body_start = after.find('\n').map(|n| n + 1).unwrap_or(0);
        let body = &after[body_start..];
        let end = body.find("```")?;
        Some(body[..end].to_string())
    }

    /// Detect a `FINAL: ...` line and return its body if present.
    pub fn extract_final(text: &str) -> Option<String> {
        for line in text.lines() {
            let l = line.trim();
            if let Some(rest) = l.strip_prefix("FINAL:") {
                return Some(rest.trim().to_string());
            }
        }
        None
    }
}

#[async_trait]
impl Agent for CodeActLoop {
    async fn run(&mut self, goal: &str) -> AgentResult<(String, StopReason)> {
        let mut transcript = String::new();
        let mut last_output = String::new();
        for iter in 0..self.inner.budget.max_iterations {
            if let Some(stop) = self.inner.iteration_check(iter) {
                return Ok((last_output, stop));
            }
            let prompt = if transcript.is_empty() {
                format!(
                    "Goal: {goal}\n\nReply with a ``` fenced code block of tool calls (one `name(arg)` per line), or `FINAL: answer` when done."
                )
            } else {
                format!(
                    "Goal: {goal}\n\nTranscript:\n{transcript}\n\nReply with a ``` fenced code block of tool calls (one `name(arg)` per line), or `FINAL: answer` when done."
                )
            };
            let reply = self.provider.complete(&prompt).await?;
            last_output = reply.text.clone();
            if let Some(stop) = self.inner.charge_llm_step(
                SpanKind::Think,
                "codeact.think",
                &prompt,
                &reply,
            ) {
                return Ok((last_output, stop));
            }
            if let Some(ans) = Self::extract_final(&reply.text) {
                return Ok((ans, StopReason::Done));
            }
            let code = match Self::extract_code(&reply.text) {
                Some(c) => c,
                None => {
                    return Ok((last_output, StopReason::Error("no code block in reply".into())));
                }
            };
            let (out_res, microjoules) = self
                .inner
                .measured(0, || async { self.runtime.run(&code).await })
                .await;
            let out = match out_res {
                Ok(o) => o,
                Err(e) => format!("runtime_error: {e}"),
            };
            if let Some(stop) = self.inner.budget.charge_tool(microjoules) {
                return Ok((last_output, stop));
            }
            self.inner.push_span(
                crate::trace::Span::new(0, SpanKind::Act, "codeact.exec", code.clone(), out.clone())
                    .with_microjoules(microjoules),
            );
            transcript.push_str(&format!("CODE:\n{code}\nOBSERVATION:\n{out}\n"));
        }
        Ok((last_output, StopReason::MaxIterations))
    }

    fn trace(&self) -> &crate::trace::Trace {
        &self.inner.trace
    }

    fn budget(&self) -> &Budget {
        &self.inner.budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_code_finds_block() {
        let s = "blah\n```python\nfoo(1)\nbar(2)\n```\ntail";
        let code = CodeActLoop::extract_code(s).unwrap();
        assert!(code.contains("foo(1)"));
        assert!(code.contains("bar(2)"));
    }

    #[test]
    fn extract_final_finds_answer() {
        assert_eq!(
            CodeActLoop::extract_final("FINAL: 42"),
            Some("42".into())
        );
        assert!(CodeActLoop::extract_final("not done").is_none());
    }

    #[test]
    fn parse_line_lenient() {
        let (n, a) = LineLangRuntime::parse_line("calc(1+1)").unwrap();
        assert_eq!(n, "calc");
        assert_eq!(a, "1+1");
        let (n, a) = LineLangRuntime::parse_line("noop").unwrap();
        assert_eq!(n, "noop");
        assert!(a.is_empty());
    }
}
