//! ReAct — Thought / Action / Observation.
//!
//! Loop:
//!
//! 1. Ask the LLM for the next *Thought* and *Action*.
//! 2. If the action is `Finish[answer]`, return.
//! 3. Otherwise, look up the named tool, run it, and feed the
//!    *Observation* back as part of the next prompt.
//! 4. Repeat until terminated by [`Budget`](crate::agent::Budget).
//!
//! The provider is expected to emit lines of the form:
//!
//! ```text
//! Thought: I should use the calc tool.
//! Action: calc[2+2]
//! ```
//!
//! Or the final form `Action: Finish[42]`. The format is intentionally
//! permissive — any trailing garbage after `Action:` is treated as the
//! action body.

use std::sync::Arc;

use async_trait::async_trait;

use crate::agent::{Agent, AgentLoop, Budget, LlmProvider, StopReason, ToolBox};
use crate::error::{AgentError, AgentResult};
use crate::trace::{Span, SpanKind};

/// One parsed ReAct step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReActStep {
    /// The model's chain-of-thought line (free text).
    pub thought: String,
    /// The action name (`"Finish"` if the agent is terminating).
    pub action: String,
    /// The action body — for `Finish[...]` this is the final answer.
    pub argument: String,
}

impl ReActStep {
    /// Parse a model completion into a step. Lenient — missing fields
    /// become empty strings.
    pub fn parse(text: &str) -> AgentResult<Self> {
        let mut thought = String::new();
        let mut action = String::new();
        let mut argument = String::new();
        for line in text.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("Thought:") {
                thought = rest.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("Action:") {
                let rest = rest.trim();
                // Parse `name[arg]` or bare `name`.
                if let (Some(lb), Some(rb)) = (rest.find('['), rest.rfind(']')) {
                    if lb < rb {
                        action = rest[..lb].trim().to_string();
                        argument = rest[lb + 1..rb].to_string();
                        continue;
                    }
                }
                action = rest.to_string();
            }
        }
        if action.is_empty() {
            return Err(AgentError::Parse(format!(
                "no Action: line in completion: {text:?}"
            )));
        }
        Ok(Self {
            thought,
            action,
            argument,
        })
    }
}

/// ReAct driver.
pub struct ReActLoop {
    provider: Arc<dyn LlmProvider>,
    tools: ToolBox,
    inner: AgentLoop,
    /// System preamble prepended to each prompt.
    pub system: String,
    transcript: String,
    last_output: String,
}

impl ReActLoop {
    /// Build a ReAct loop. `system` is prepended to every prompt; it
    /// should describe the available tools and the `Thought:` / `Action:`
    /// format the model is expected to emit.
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        tools: ToolBox,
        budget: Budget,
        system: impl Into<String>,
    ) -> Self {
        Self {
            provider,
            tools,
            inner: AgentLoop::new(budget),
            system: system.into(),
            transcript: String::new(),
            last_output: String::new(),
        }
    }

    fn build_prompt(&self, goal: &str) -> String {
        if self.transcript.is_empty() {
            format!("{sys}\n\nGoal: {goal}\n", sys = self.system)
        } else {
            format!(
                "{sys}\n\nGoal: {goal}\n{tr}",
                sys = self.system,
                tr = self.transcript
            )
        }
    }
}

#[async_trait]
impl Agent for ReActLoop {
    async fn run(&mut self, goal: &str) -> AgentResult<(String, StopReason)> {
        for iter in 0..self.inner.budget.max_iterations {
            if let Some(stop) = self.inner.iteration_check(iter) {
                return Ok((self.last_output.clone(), stop));
            }

            let prompt = self.build_prompt(goal);
            let reply = self.provider.complete(&prompt).await?;
            self.last_output = reply.text.clone();

            if let Some(stop) =
                self.inner.charge_llm_step(SpanKind::Think, "react.step", &prompt, &reply)
            {
                return Ok((self.last_output.clone(), stop));
            }

            let step = match ReActStep::parse(&reply.text) {
                Ok(s) => s,
                Err(e) => return Ok((self.last_output.clone(), StopReason::Error(e.to_string()))),
            };

            if step.action.eq_ignore_ascii_case("Finish") {
                return Ok((step.argument, StopReason::Done));
            }

            let tool = match self.tools.get(&step.action) {
                Some(t) => t,
                None => {
                    let obs = format!("ERROR: unknown tool `{}`", step.action);
                    self.transcript
                        .push_str(&format!("Observation: {obs}\n"));
                    self.inner.push_span(
                        Span::new(0, SpanKind::Observe, step.action.clone(), step.argument.clone(), obs.clone()),
                    );
                    continue;
                }
            };

            let args = serde_json::json!({ "input": step.argument });
            let (result, microjoules) = self
                .inner
                .measured(0, || async { tool.run(args).await })
                .await;
            let observation = match result {
                Ok(v) => v.to_string(),
                Err(e) => format!("ERROR: {e}"),
            };
            if let Some(stop) = self.inner.budget.charge_tool(microjoules) {
                return Ok((self.last_output.clone(), stop));
            }
            self.inner.push_span(
                Span::new(
                    0,
                    SpanKind::Act,
                    step.action.clone(),
                    step.argument.clone(),
                    observation.clone(),
                )
                .with_microjoules(microjoules),
            );
            self.transcript.push_str(&format!(
                "Thought: {th}\nAction: {ac}[{arg}]\nObservation: {obs}\n",
                th = step.thought,
                ac = step.action,
                arg = step.argument,
                obs = observation
            ));
        }
        Ok((self.last_output.clone(), StopReason::MaxIterations))
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
    fn parse_thought_action() {
        let s = ReActStep::parse("Thought: hi\nAction: calc[1+1]").unwrap();
        assert_eq!(s.thought, "hi");
        assert_eq!(s.action, "calc");
        assert_eq!(s.argument, "1+1");
    }

    #[test]
    fn parse_finish() {
        let s = ReActStep::parse("Action: Finish[42]").unwrap();
        assert_eq!(s.action, "Finish");
        assert_eq!(s.argument, "42");
    }

    #[test]
    fn parse_missing_action_errors() {
        let err = ReActStep::parse("Thought: nope\n").unwrap_err();
        assert!(matches!(err, AgentError::Parse(_)));
    }
}
